"""MIMIC-to-EDU source-only fitting, locked zero-shot evaluation and adaptation."""
from __future__ import annotations
from pathlib import Path
import copy, math, json
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from . import models
from .data import aggregate, write_json, subset
from .validation import REGISTRY, fit_model, predict_raw, prediction_frame


def anchored_loss(theta,x,y,weights,origin,penalty):
    score = x@theta[:-1]+theta[-1]; delta = theta-origin
    loss = np.dot(weights,np.logaddexp(0,score)-y*score)+penalty/2*np.dot(delta,delta)
    residual = weights*(expit(score)-y)
    return loss, np.r_[x.T@residual,residual.sum()]+penalty*delta


def make_roles(source,target,seed=61007):
    a = source.patients.set_index('patient_id').loc[sorted(source.ids)]
    b = target.patients.set_index('patient_id').loc[sorted(target.ids)]
    atv, ate = train_test_split(a.index.to_numpy(),test_size=.2,stratify=a.label,random_state=seed)
    atr, ava = train_test_split(atv,test_size=.25,stratify=a.loc[atv,'label'],random_state=seed+1)
    bad, bte = train_test_split(b.index.to_numpy(),train_size=math.ceil(.2*len(b)),stratify=b.label,random_state=seed)
    btr, bva = train_test_split(sorted(bad),test_size=math.ceil(.2*len(bad)),stratify=b.loc[sorted(bad),'label'],random_state=seed+4)
    return {k:sorted(v) for k,v in {'A_train':atr,'A_val':ava,'A_internal_test':ate,'B_adapt':bad,'B_test':bte,'B_train':btr,'B_val':bva}.items()}


def check_roles(source,target,roles):
    if set(source.ids) & set(target.ids):
        raise ValueError('Source and target patient identifiers must be globally distinct')
    for data,parts,all_ids in [(source,['A_train','A_val','A_internal_test'],set(source.ids)),(target,['B_train','B_val','B_test'],set(target.ids))]:
        seen, groups = set(),set()
        for key in parts:
            ix=data.index(roles[key]); current=set(data.groups[ix])
            if seen & set(roles[key]) or groups & current:
                raise ValueError('Patient or duplicate-group leakage across transfer roles')
            if len(np.unique(data.y[ix]))!=2:
                raise ValueError('Transfer fitting, validation and test roles require both classes')
            seen.update(roles[key]);groups.update(current)
        if seen != all_ids:
            raise ValueError('Transfer roles must partition the eligible cohort')
    if set(roles['B_train'])|set(roles['B_val']) != set(roles['B_adapt']):
        raise ValueError('Target adaptation roles disagree')


def calibrate(artifact,data,ids):
    ix=data.index(ids);raw=predict_raw(artifact,data,ix)
    cal=models.fit_platt(data.y[ix],raw)
    return {**artifact,'calibrator':cal,'threshold':models.threshold_at_minimum_specificity(data.y[ix],models.apply_platt(cal,raw),.6),'calibration_ids':list(ids)}


def update_model(source,data,train_ids,params,seed):
    torch = models.torch
    rep,kind=REGISTRY[source['name']]
    ix=data.index(train_ids)
    x,_,pi,w,_=data.matrix(rep,source['selected'],source['clinical_columns']);mask=np.isin(pi,ix)
    original=source['bundle'];x=np.where(np.isfinite(x[mask]),x[mask],original['fill'])
    if original['scaler'] is not None:x=original['scaler'].transform(x)
    y,yp,weight=data.y[pi[mask]],data.y[ix],w[mask]
    result=copy.deepcopy(source);bundle=result['bundle']
    result.update(train_ids=list(train_ids),seed=int(seed),adaptation_params=params)
    if kind=='xgboost':
        if not torch.cuda.is_available():raise RuntimeError('XGBoost CUDA is required')
        model=models.XGBClassifier(n_estimators=params['extra_trees'],learning_rate=.01,max_depth=2,subsample=.8,colsample_bytree=.8,objective='binary:logistic',eval_metric='logloss',scale_pos_weight=float(sum(yp==0)/sum(yp==1)),tree_method='hist',device='cuda',n_jobs=1,random_state=seed)
        base=original['estimator'].get_booster();before=base.get_dump(dump_format='json')
        model.fit(x.astype(np.float32),y,sample_weight=weight,xgb_model=base)
        after=model.get_booster().get_dump(dump_format='json')
        if after[:len(before)]!=before or len(after)!=len(before)+params['extra_trees']:
            raise AssertionError('Source trees changed during continuation')
        bundle['estimator']=model
    elif kind=='logistic_l2':
        estimator=original['estimator'];origin=np.r_[estimator.coef_[0],estimator.intercept_[0]]
        weights=weight*np.where(y==1,len(yp)/(2*sum(yp==1)),len(yp)/(2*sum(yp==0)));weights/=weights.sum()
        fit=minimize(anchored_loss,origin.copy(),args=(x,y,weights,origin,params['anchor']),method='L-BFGS-B',jac=True,options={'maxiter':2000,'ftol':1e-12,'gtol':1e-7})
        if not fit.success:raise RuntimeError(fit.message)
        bundle['estimator'].coef_=fit.x[None,:-1];bundle['estimator'].intercept_=fit.x[-1:]
    elif kind=='torch_mlp':
        if not torch.cuda.is_available():raise RuntimeError('Neural-head CUDA is required')
        import random
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        head=bundle['estimator'].cuda().train()
        tx=torch.from_numpy(np.asarray(x,dtype=np.float32)).cuda();ty=torch.from_numpy(y.astype(np.float32)).cuda();tw=torch.from_numpy(weight.astype(np.float32)).cuda()
        criterion=torch.nn.BCEWithLogitsLoss(reduction='none',pos_weight=torch.tensor(float(sum(yp==0)/sum(yp==1)),device='cuda'))
        optimizer=torch.optim.AdamW(head.parameters(),lr=params['lr'],weight_decay=1e-3)
        for _ in range(params['epochs']):
            optimizer.zero_grad(set_to_none=True);loss=(criterion(head(tx),ty)*tw).sum()/tw.sum();loss.backward();optimizer.step()
        head.eval()
    else:
        raise ValueError('ExtraTrees undergoes recalibration only')
    # No target feature selection, refitted scaler or refitted imputation.
    np.testing.assert_array_equal(bundle['fill'],original['fill'])
    return result


def transfer(source,target,model_names,output,roles=None,encoder_code=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if source.cohort!='MIMIC' or target.cohort!='EDU':raise ValueError('Study transfer direction is MIMIC to EDU')
    target.config={**target.config,'force_segments':True}
    seed=source.config['seeds']['transfer']
    if isinstance(roles,(str,Path)):
        payload=json.loads(Path(roles).read_text());roles=payload.get('splits',payload)
    roles=roles or make_roles(source,target,seed)
    check_roles(source,target,roles)
    write_json(output/'roles.json',roles)
    write_json(output/'run_manifest.json',{'config_sha256':source.config['config_sha256'],'source_input_hashes':getattr(source,'input_hashes',{}),'target_input_hashes':getattr(target,'input_hashes',{}),'mode':'MIMIC_to_EDU_20pct_adaptation','models':model_names})
    results=[]
    for name in model_names:
        if name=='net1d':
            results.extend(net1d_transfer(source,target,roles,output/name,encoder_code));continue
        kind=REGISTRY[name][1];best=None
        grid=source.config.get('grids',{}).get(kind,models.method_grid(kind))
        for params in grid:
            a=fit_model(source,name,params,source.index(roles['A_train']),seed,source.config['clinical']['MIMIC'])
            raw=predict_raw(a,source,source.index(roles['A_val']))
            score=(roc_auc_score(source.y[source.index(roles['A_val'])],raw),average_precision_score(source.y[source.index(roles['A_val'])],raw))
            if best is None or score>best[0]:best=(score,a)
        source_model=calibrate(best[1],source,roles['A_val'])
        folder=output/name;folder.mkdir(exist_ok=True)
        # Freeze and evaluate zero-shot BEFORE target adaptation/model selection.
        zero=prediction_frame(source_model,target,target.index(roles['B_test']),phase='Zero-shot')
        zero.to_csv(folder/'zero_shot.csv',index=False)
        joblib.dump(source_model,folder/'source.joblib',compress=3)
        if kind=='extra_trees':
            adapted=calibrate(copy.deepcopy(source_model),target,roles['B_adapt'])
            adapted.update(train_ids=[],adaptation='recalibration_only')
        else:
            grid=source.config['update_grids'][kind];best=None
            for params in grid:
                a=update_model(source_model,target,roles['B_train'],params,seed+1020)
                raw=predict_raw(a,target,target.index(roles['B_val']));y=target.y[target.index(roles['B_val'])]
                score=(roc_auc_score(y,raw),average_precision_score(y,raw))
                if best is None or score>best[0]:best=score,a
            adapted=calibrate(best[1],target,roles['B_val'])
        updated=prediction_frame(adapted,target,target.index(roles['B_test']),phase='Adapted')
        np.testing.assert_array_equal(zero.patient_id,updated.patient_id)
        updated.to_csv(folder/'adapted.csv',index=False)
        if kind=='torch_mlp':adapted['bundle']['estimator'].cpu()
        joblib.dump(adapted,folder/'adapted.joblib',compress=3)
        from .evaluation import bootstrap_performance,paired_delta
        for label,frame in [('Zero-shot',zero),('Adapted',updated)]:
            results.append({'model':name,'phase':label,**bootstrap_performance(frame,source.config['bootstrap_iterations'],seed,True)})
        write_json(folder/'paired_delta.json',paired_delta(zero,updated,source.config['bootstrap_iterations'],seed))
        write_json(folder/'roles_used.json',{'source_fitting':roles['A_train'],'source_selection_calibration':roles['A_val'],'target_adaptation':roles['B_adapt'],'test':roles['B_test'],'test_not_used_for_selection':True})
    write_json(output/'performance.json',results)
    return results


def net1d_transfer(source,target,roles,output,encoder_code):
    """Source-random Net1D, then full-network adaptation on target development only."""
    import torch
    if not encoder_code or source.wave is None or target.wave is None:
        raise ValueError('Net1D requires upstream --encoder-code and encoder_input arrays in both datasets')
    if not torch.cuda.is_available():raise RuntimeError('Net1D study training requires CUDA')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    cfg=source.config['net1d'];seed=source.config['seeds']['transfer']
    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__();self.encoder=models.upstream_encoder(encoder_code);self.head=torch.nn.Linear(512,1)
        def forward(self,x):return self.head(self.encoder(x)).squeeze(1)
    def predict(model,data,ids):
        ix=data.index(ids);seg=np.flatnonzero(np.isin(data.pi,ix));values=[];model.eval()
        with torch.inference_mode():
            for a in range(0,len(seg),64):values.append(torch.sigmoid(model(torch.from_numpy(data.wave[seg[a:a+64]]).unsqueeze(1).cuda())).cpu().numpy())
        return aggregate(np.concatenate(values),data.pi[seg],len(data.ids))[ix]
    def fit(data,train_ids,val_ids,initial,run_seed):
        import random
        random.seed(run_seed);np.random.seed(run_seed)
        torch.manual_seed(run_seed);torch.cuda.manual_seed_all(run_seed);rng=np.random.default_rng(run_seed)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True)
        model=Network().cuda()
        if initial is not None:model.load_state_dict(initial)
        it=data.index(train_ids);yv=data.y[data.index(val_ids)];y=data.y[it]
        by={i:np.flatnonzero(data.pi==i) for i in it};offset={i:int(rng.integers(len(by[i]))) for i in it}
        opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        lossfn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(sum(y==0)/sum(y==1)),device='cuda'))
        best=None
        for epoch in range(1,cfg['epochs']+1):
            model.train();order=rng.permutation(it)
            for start in range(0,len(order),cfg['batch_size']):
                patients=order[start:start+cfg['batch_size']];seg=np.array([by[i][(offset[i]+epoch-1)%len(by[i])] for i in patients])
                opt.zero_grad(set_to_none=True);loss=lossfn(model(torch.from_numpy(data.wave[seg]).unsqueeze(1).cuda()),torch.tensor(data.y[patients],device='cuda',dtype=torch.float32));loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip']);opt.step()
            if epoch%cfg['validate_every']==0 or epoch==cfg['epochs']:
                raw=predict(model,data,val_ids);score=(roc_auc_score(yv,raw),average_precision_score(yv,raw))
                if best is None or score>best[0]:best=(score,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        model.load_state_dict(best[1]);return model
    def calibrated(model,data,val_ids):
        raw=predict(model,data,val_ids);y=data.y[data.index(val_ids)];cal=models.fit_platt(y,raw)
        return cal,models.threshold_at_minimum_specificity(y,models.apply_platt(cal,raw),.6)
    original=fit(source,roles['A_train'],roles['A_val'],None,seed)
    initial={k:v.detach().cpu().clone() for k,v in original.state_dict().items()}
    cal,threshold=calibrated(original,source,roles['A_val'])
    def frame(model,cal,threshold,phase):
        raw=predict(model,target,roles['B_test']);prob=models.apply_platt(cal,raw)
        return pd.DataFrame({'patient_id':roles['B_test'],'label':target.y[target.index(roles['B_test'])],'raw_score':raw,'probability':prob,'threshold_60spec':threshold,'prediction':(prob>=threshold).astype(int),'phase':phase})
    zero=frame(original,cal,threshold,'Zero-shot');zero.to_csv(output/'zero_shot.csv',index=False)
    torch.save(initial,output/'source.pt')
    adapted=fit(target,roles['B_train'],roles['B_val'],initial,seed+1020)
    cal,threshold=calibrated(adapted,target,roles['B_val']);updated=frame(adapted,cal,threshold,'Adapted');updated.to_csv(output/'adapted.csv',index=False)
    torch.save({k:v.detach().cpu() for k,v in adapted.state_dict().items()},output/'adapted.pt')
    from .evaluation import bootstrap_performance
    return [{'model':'net1d','phase':phase,**bootstrap_performance(f,source.config['bootstrap_iterations'],seed,True)} for phase,f in [('Zero-shot',zero),('Adapted',updated)]]
