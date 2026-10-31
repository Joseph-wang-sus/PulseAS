"""Patient-level performance, uncertainty and matched clinical comparisons."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
from .data import write_json


def metrics(y, p, prediction=None):
    y, p = np.asarray(y, int), np.asarray(p, float)
    if len(y) != len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Finite, aligned probabilities in [0,1] are required')
    result = {'patients': len(y), 'cases': int(y.sum()), 'Brier': float(brier_score_loss(y, p))}
    if len(np.unique(y)) == 2:
        result.update(AUROC=float(roc_auc_score(y, p)), AP=float(average_precision_score(y, p)))
    else:
        result.update(AUROC=None, AP=None)
    if prediction is not None:
        tn, fp, fn, tp = confusion_matrix(y, np.asarray(prediction, int), labels=[0, 1]).ravel()
        sens = tp/(tp+fn) if tp+fn else None
        spec = tn/(tn+fp) if tn+fp else None
        result.update(TP=int(tp), FN=int(fn), TN=int(tn), FP=int(fp), sensitivity=sens, specificity=spec,
                      balanced_accuracy=(sens+spec)/2 if sens is not None and spec is not None else None,
                      F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    return result


def bootstrap_indices(y, iterations, seed, stratified=False):
    y = np.asarray(y); rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == k) for k in (0, 1)]
    for _ in range(iterations):
        ix = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups]) if stratified else rng.integers(0, len(y), len(y))
        if len(np.unique(y[ix])) == 2:
            yield ix


def bootstrap_performance(frame, iterations=2000, seed=42, stratified=False):
    if frame.patient_id.duplicated().any():
        raise ValueError('Bootstrap input must contain one OOF prediction per patient')
    y, p = frame.label.to_numpy(int), frame.probability.to_numpy(float)
    pred = frame.prediction.to_numpy(int) if 'prediction' in frame else None
    result = metrics(y, p, pred)
    samples = [metrics(y[ix], p[ix], None if pred is None else pred[ix]) for ix in bootstrap_indices(y, iterations, seed, stratified)]
    for key in ['AUROC', 'AP', 'Brier', 'sensitivity', 'specificity', 'balanced_accuracy']:
        values = [row[key] for row in samples if row.get(key) is not None]
        if values:
            result[key+'_95CI'] = np.quantile(values, [.025, .975]).tolist()
    result.update(bootstrap_valid=len(samples), bootstrap_requested=iterations, bootstrap_stratified=stratified)
    return result


def paired_delta(left, right, iterations=2000, seed=42, stratified=True):
    if left.patient_id.duplicated().any() or right.patient_id.duplicated().any() or set(left.patient_id) != set(right.patient_id):
        raise ValueError('Paired comparison requires identical unique patient sets')
    right = right.set_index('patient_id').loc[left.patient_id].reset_index()
    np.testing.assert_array_equal(left.label, right.label)
    y, a, b = left.label.to_numpy(int), left.probability.to_numpy(), right.probability.to_numpy()
    values = [roc_auc_score(y[ix], b[ix])-roc_auc_score(y[ix], a[ix]) for ix in bootstrap_indices(y, iterations, seed, stratified)]
    return {'delta_AUROC': float(roc_auc_score(y, b)-roc_auc_score(y, a)), 'CI95': np.quantile(values, [.025,.975]).tolist(), 'patients': len(y)}


def subgroup_performance(frame, patients):
    f = frame.merge(patients, on='patient_id', how='left', validate='one_to_one', suffixes=('', '_patient'))
    rows = []
    for name, column, cutoff in [('age', 'age', 65), ('SBP', 'sbp', 130), ('DBP', 'dbp', 80), ('sex', 'sex_female', .5)]:
        if column not in f:
            continue
        x = pd.to_numeric(f[column], errors='coerce')
        for label, mask in [('lower', x.lt(cutoff)), ('higher', x.ge(cutoff)), ('missing', x.isna())]:
            if mask.any():
                q = f.loc[mask]
                rows.append({'variable': name, 'stratum': label, 'cutoff': cutoff, **metrics(q.label, q.probability, q.prediction)})
    return pd.DataFrame(rows)


def reliability(frame):
    f = frame.sort_values('probability', kind='stable').copy()
    f['bin'] = np.minimum(np.arange(len(f))*10//len(f), 9)
    return f.groupby('bin').agg(patients=('patient_id','size'), mean_probability=('probability','mean'), observed_proportion=('label','mean')).reset_index()


def clinical_comparisons(data, output, folds, branch='main'):
    from .data import subset
    from .validation import nested_validate, check_splits
    output = Path(output)
    if branch == 'complete':
        keep = np.isfinite(data.patients[['age','sex_female','sbp','dbp']].to_numpy(float)).all(axis=1)
        indices = np.flatnonzero(keep); old_to_new = {int(v): i for i,v in enumerate(indices)}
        folds = [{'fold': f['fold'], 'train': [old_to_new[i] for i in f['train'] if i in old_to_new], 'test': [old_to_new[i] for i in f['test'] if i in old_to_new], 'inner': [{'train':[old_to_new[i] for i in j['train'] if i in old_to_new], 'valid':[old_to_new[i] for i in j['valid'] if i in old_to_new]} for j in f['inner']]} for f in folds]
        data = subset(data, indices); check_splits(data, folds)
    elif branch not in ('main','spline'):
        raise ValueError('Clinical branch must be main, complete or spline')
    if branch == 'spline':
        data.config = {**data.config, 'clinical_splines': True}
    baselines = {'dbp':['dbp'], 'pp':['pp'], 'age_dbp':['age','dbp'], 'age_pp':['age','pp'], 'clinical4':['age','sex_female','sbp','dbp']}
    rows = []
    for name, columns in baselines.items():
        nested_validate(data, ['clinical','clinical_fusion_logistic'], output/name, folds, columns)
        left = pd.read_csv(output/name/'clinical/oof.csv', dtype={'patient_id': str})
        right = pd.read_csv(output/name/'clinical_fusion_logistic/oof.csv', dtype={'patient_id': str})
        rows.append({'baseline': name, 'branch': branch, **paired_delta(left,right,data.config['bootstrap_iterations'],stratified=data.cohort=='MIMIC')})
    pd.DataFrame(rows).to_csv(output/'matched_delta.csv', index=False)
    return rows


def evaluate_file(predictions, output, config, cohort, patients=None):
    f = pd.read_csv(predictions, dtype={'patient_id': str}); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    write_json(output/'metrics.json', bootstrap_performance(f,config['bootstrap_iterations'],config['seeds']['bootstrap'],cohort=='MIMIC'))
    reliability(f).to_csv(output/'reliability.csv', index=False)
    if patients is not None:
        subgroup_performance(f,pd.read_csv(patients,dtype={'patient_id':str})).to_csv(output/'subgroups.csv',index=False)
