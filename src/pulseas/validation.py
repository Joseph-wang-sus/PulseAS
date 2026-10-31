"""Patient-grouped nested validation and source-compatible model bundles."""
from __future__ import annotations
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from .data import aggregate, write_json
from . import models

REGISTRY = {
    'full_pyppg': ('all_pyppg', 'extra_trees'), 'clinical': ('clinical', 'logistic_l2'),
    'anyppg': ('anyppg', 'xgboost'), 'fusion_top10': ('fusion', 'xgboost'),
    'neural_fusion': ('fusion', 'torch_mlp'), 'clinical_fusion': ('clinical_fusion', 'xgboost'),
    'clinical_neural_fusion': ('clinical_fusion', 'torch_mlp'),
    'clinical_fusion_logistic': ('clinical_fusion', 'logistic_l2'),
}
EDU_METHODS = {'neural_fusion': 'neural_fusion_top10', 'clinical_fusion': 'clinical_fusion_top10', 'clinical_neural_fusion': 'clinical_neural_fusion_top10', 'clinical_fusion_logistic': 'clinical_fusion_top10_logistic'}


def select_top10(x, y, schema):
    if len(np.unique(y)) != 2:
        raise ValueError('Feature selection needs both AS classes')
    fill = models.fit_imputation(x, ['pyppg'] * x.shape[1])
    ranks = rankdata(np.where(np.isfinite(x), x, fill), axis=0)
    n1, n0 = int(sum(y == 1)), int(sum(y == 0))
    auc = (ranks[y == 1].sum(axis=0) - n1 * (n1 + 1) / 2) / (n1 * n0)
    used, selected = set(), []
    for j in np.argsort(-np.abs(auc - .5), kind='stable'):
        domain = schema['domains'][schema['pyppg'][j]]
        if domain not in used:
            used.add(domain); selected.append(int(j))
        if len(selected) == 10:
            return selected
    raise ValueError('At least ten morphology domains are required')


def make_splits(data, outer=5, inner=3):
    y, groups = data.y, data.groups
    # Existing EDU study folds must be supplied: they inherited joint-endpoint strata.
    if data.config.get('require_frozen_EDU_folds') and data.cohort == 'EDU':
        raise ValueError('Study EDU validation requires --folds with the inherited patient-level folds')
    seed = data.config['seeds']['internal_split']
    result = []
    for fold, (tr, te) in enumerate(StratifiedGroupKFold(outer, shuffle=True, random_state=seed).split(data.ids, y, groups)):
        inn = []
        inner_seed = seed + fold if data.cohort == 'MIMIC' else 170 + fold + 1
        for a, b in StratifiedGroupKFold(inner, shuffle=True, random_state=inner_seed).split(tr, y[tr], groups[tr]):
            inn.append({'train': tr[a].tolist(), 'valid': tr[b].tolist()})
        result.append({'fold': fold if data.cohort == 'MIMIC' else fold + 1, 'train': tr.tolist(), 'test': te.tolist(), 'inner': inn})
    return result


def check_splits(data, folds):
    seen = []
    for fold in folds:
        tr, te = np.array(fold['train']), np.array(fold['test'])
        if len(set(tr)) != len(tr) or len(set(te)) != len(te) or set(tr) | set(te) != set(range(len(data.ids))):
            raise ValueError('Each outer fold must partition every analysis patient exactly once')
        if set(data.groups[tr]) & set(data.groups[te]):
            raise ValueError('Outer-fold patient/duplicate-group leakage')
        inner_seen = []
        for inner in fold['inner']:
            a, b = np.array(inner['train']), np.array(inner['valid'])
            if set(a) | set(b) != set(tr) or set(data.groups[a]) & set(data.groups[b]):
                raise ValueError('Inner fold must partition outer training groups only')
            if len(np.unique(data.y[a])) != 2 or len(np.unique(data.y[b])) != 2:
                raise ValueError('Both classes are needed in each inner train/validation fold')
            inner_seen.extend(b.tolist())
        if sorted(inner_seen) != sorted(tr.tolist()):
            raise ValueError('Inner validation folds must cover outer training patients once')
        seen.extend(te.tolist())
    if sorted(seen) != list(range(len(data.ids))):
        raise ValueError('Outer test folds must cover analysis patients exactly once')


def fit_model(data, name, params, train, seed, clinical_columns=None, preprocessing=None):
    representation, classifier = REGISTRY[name]
    train = np.asarray(train, int)
    if len(np.unique(data.y[train])) != 2:
        raise ValueError('Training requires both classes')
    selected = select_top10(data.medians[train], data.y[train], data.config['schema']) if representation in ('fusion', 'clinical_fusion') else []
    if preprocessing is not None:
        selected = preprocessing['selected']
    x, summaries, pi, weights, names = data.matrix(representation, selected, clinical_columns)
    mask = np.isin(pi, train)
    spline = None
    if data.config.get('clinical_splines'):
        fill = models.fit_imputation(summaries[train], names)
        complete = np.where(np.isfinite(x), x, fill)
        training = np.where(np.isfinite(summaries[train]), summaries[train], fill)
        x, knots = models.spline_design(training, complete, names)
        spline = {'fill': fill, 'knots': knots}
        summaries = models.apply_splines(np.where(np.isfinite(summaries), summaries, fill), knots)
        names = [f'basis_{i}' for i in range(x.shape[1])]
    if data.cohort == 'EDU' and not data.config.get('force_segments'):
        bundle = models.fit_bundle(classifier, params, x[mask], data.y[pi[mask]], names, seed)
    else:
        bundle = models.fit_weighted(classifier, params, x[mask], data.y[pi[mask]], weights[mask], summaries[train], data.y[train], names, seed)
    return {'name': name, 'bundle': bundle, 'selected': selected, 'clinical_columns': clinical_columns or list(data.config['clinical'][data.cohort]), 'train_ids': data.ids[train].tolist(), 'params': params, 'seed': int(seed), 'spline': spline}


def predict_raw(artifact, data, indices):
    rep = REGISTRY[artifact['name']][0]
    x, _, pi, _, _ = data.matrix(rep, artifact['selected'], artifact['clinical_columns'])
    indices = np.asarray(indices, int)
    mask = np.isin(pi, indices)
    if artifact.get('spline') is not None:
        s = artifact['spline']; x = models.apply_splines(np.where(np.isfinite(x),x,s['fill']),s['knots'])
    raw = models.predict_bundle(artifact['bundle'], x[mask])
    return aggregate(raw, pi[mask], len(data.ids))[indices]


def prediction_frame(artifact, data, indices, **metadata):
    raw = predict_raw(artifact, data, indices)
    prob = models.apply_platt(artifact['calibrator'], raw)
    return pd.DataFrame({'patient_id': data.ids[indices], 'label': data.y[indices], 'raw_score': raw, 'probability': prob, 'threshold_60spec': artifact['threshold'], 'prediction': (prob >= artifact['threshold']).astype(int), **metadata})


def nested_validate(data, model_names, output, folds=None, clinical_columns=None):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if isinstance(folds, (str, Path)):
        payload = json.loads(Path(folds).read_text(encoding='utf-8'))
        if isinstance(payload, dict):
            if payload['patient_id'] != data.ids.tolist():
                raise ValueError('Frozen-fold patient order differs from the supplied dataset')
            folds = payload['folds']
        else:
            raise ValueError('Frozen folds require {patient_id, folds} to validate patient order')
    folds = folds or make_splits(data, data.config['outer_folds'], data.config['inner_folds'])
    if len(folds)!=data.config['outer_folds'] or any(len(f['inner'])!=data.config['inner_folds'] for f in folds):
        raise ValueError('Fold counts differ from the explicitly selected analysis configuration')
    check_splits(data, folds)
    write_json(output / 'folds.json', {'patient_id': data.ids.tolist(), 'folds': folds})
    write_json(output / 'run_manifest.json', {'cohort':data.cohort,'models':model_names,'config_sha256':data.config['config_sha256'],'feature_schema_sha256':data.config['schema_sha256'],'input_hashes':getattr(data,'input_hashes',{}),'mode':'nested_internal_validation'})
    summaries = []
    for name in model_names:
        classifier = REGISTRY[name][1]
        grid = data.config.get('grids', {}).get(classifier, models.method_grid(classifier))
        folder = output/name; folder.mkdir(exist_ok=True)
        rows, receipts = [], []
        for split in folds:
            fold = split['fold']; tr, te = np.array(split['train']), np.array(split['test'])
            best, candidates = None, []
            for params in grid:
                raw = np.full(len(data.ids), np.nan)
                for k, inner in enumerate(split['inner']):
                    it, iv = np.array(inner['train']), np.array(inner['valid'])
                    if data.cohort == 'MIMIC':
                        seed = 17 + fold*100 + k
                    else:
                        seed = models.stable_seed('EDU', 'AS', EDU_METHODS.get(name, name), fold, k+1, json.dumps(params))
                    a = fit_model(data, name, params, it, seed, clinical_columns)
                    raw[iv] = predict_raw(a, data, iv)
                score = (roc_auc_score(data.y[tr], raw[tr]), average_precision_score(data.y[tr], raw[tr]))
                candidates.append({'params': params, 'AUROC': score[0], 'AP': score[1]})
                if best is None or score > best[0]:
                    best = score, params, raw.copy()
            _, params, raw = best
            cal = models.fit_platt(data.y[tr], raw[tr])
            threshold = models.threshold_at_minimum_specificity(data.y[tr], models.apply_platt(cal, raw[tr]), .6)
            seed = 17000+fold if data.cohort == 'MIMIC' else models.stable_seed('EDU', 'AS', EDU_METHODS.get(name, name), 'outer', fold)
            a = fit_model(data, name, params, tr, seed, clinical_columns)
            a.update(calibrator=cal, threshold=threshold, test_ids=data.ids[te].tolist())
            frame = prediction_frame(a, data, te, fold=fold)
            frame.to_csv(folder/f'fold_{fold}.csv', index=False)
            pd.DataFrame({'patient_id': data.ids[tr], 'label': data.y[tr], 'raw_score': raw[tr]}).to_csv(folder/f'fold_{fold}_inner_oof.csv', index=False)
            if classifier == 'torch_mlp':
                a['bundle']['estimator'].cpu()
            joblib.dump(a, folder/f'fold_{fold}.joblib', compress=3)
            receipts.append({'fold': fold, 'params': params, 'candidates': candidates, 'selected_features': [data.config['schema']['pyppg'][i] for i in a['selected']], 'train_ids': a['train_ids'], 'test_ids': a['test_ids'], 'seed': seed})
            rows.append(frame)
        oof = pd.concat(rows).set_index('patient_id').loc[data.ids].reset_index()
        oof.to_csv(folder/'oof.csv', index=False)
        write_json(folder/'selection_audit.json', receipts)
        from .evaluation import metrics
        result = {'model': name, **metrics(oof.label, oof.probability, oof.prediction)}
        write_json(folder/'metrics.json', result); summaries.append(result)
    pd.DataFrame(summaries).to_csv(output/'performance.csv', index=False)
    return pd.DataFrame(summaries)
