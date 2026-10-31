"""Patient-level morphology tests and complete-pulse templates."""
from __future__ import annotations
import contextlib, io, json, warnings
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis, mannwhitneyu, wilcoxon
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
def fit_one(y,post,ids):
    good = np.isfinite(y)
    y,post,ids = y[good],post[good],ids[good]
    counts = pd.crosstab(ids,post)
    pairs = int(((counts.get(0,0)>0)&(counts.get(1,0)>0)).sum())
    row = dict(n_obs=len(y),n_patients=len(np.unique(ids)),n_pre=int((post==0).sum()),
               n_post=int((post==1).sum()),n_pairs=pairs)
    for code,label in [(0,'pre'),(1,'post')]:
        v = y[post==code]
        row.update({label+'_mean':float(np.mean(v)),label+'_median':float(np.median(v)),
                    label+'_q25':float(np.quantile(v,.25)),label+'_q75':float(np.quantile(v,.75))})
    scale = float(np.std(y))
    if not np.isfinite(scale) or scale <= 0 or pairs < 5:
        return row | dict(status='untestable',p=np.nan)
    # Linear rescaling improves numerical conditioning; effects are returned in original units.
    yy = (y-np.mean(y))/scale
    X = np.column_stack([np.ones(len(y)),post])
    robust = sm.OLS(yy,X).fit(cov_type='cluster',cov_kwds={'groups':ids,'use_correction':True,'df_correction':True},use_t=True)
    ci = robust.conf_int()[1]*scale
    row.update(cluster_beta=float(robust.params[1]*scale),cluster_se=float(robust.bse[1]*scale),
        cluster_ci_low=float(ci[0]),cluster_ci_high=float(ci[1]),cluster_p=float(robust.pvalues[1]))
    attempts=[]
    result=None
    for method in ['bfgs','powell']:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                fitted = sm.MixedLM(yy,X,groups=ids).fit(reml=True,method=method,maxiter=1000,disp=False)
            notes = sorted({str(w.message) for w in caught})
            reliable = bool(fitted.converged and np.isfinite(fitted.fe_params).all()
                and np.isfinite(fitted.bse_fe).all() and (fitted.bse_fe>0).all()
                and np.isfinite(fitted.scale) and fitted.scale>0
                and not any('not positive definite' in s.lower() for s in notes))
            attempts.append(dict(method=method,converged=bool(fitted.converged),acceptable=reliable,warnings=notes))
            if reliable:
                result=fitted
                break
        except Exception as e:
            attempts.append(dict(method=method,error=type(e).__name__+': '+str(e)))
    row['fit_attempts']=json.dumps(attempts)
    if result is None:
        return row | dict(status='failed',p=np.nan)
    beta,se = result.fe_params[1]*scale,result.bse_fe[1]*scale
    p = float(2*norm.sf(abs(beta/se)))
    assert np.isclose(p,result.pvalues[1],atol=1e-12)
    var_random=float(np.asarray(result.cov_re)[0,0])
    var_error=float(result.scale)
    icc=var_random/(var_random+var_error)
    resid = yy-X@result.fe_params
    for pid in np.unique(ids):
        ix = ids==pid
        resid[ix] -= var_random*resid[ix].sum()/(var_error+ix.sum()*var_random)
    sd = np.std(resid,ddof=2)
    sk,ku,maxr = float(skew(resid)),float(kurtosis(resid,fisher=True)),float(np.max(np.abs(resid))/sd)
    variance_ratio=float(np.var(resid[post==1],ddof=1)/np.var(resid[post==0],ddof=1))
    row.update(status='ok',method=attempts[-1]['method'],beta=float(beta),se=float(se),
        ci_low=float(beta-1.959963984540054*se),ci_high=float(beta+1.959963984540054*se),p=p,
        random_variance=var_random*scale**2,residual_variance=var_error*scale**2,icc=icc,
        boundary=icc<1e-5,residual_skew=sk,residual_excess_kurtosis=ku,max_abs_standardized_residual=maxr,
        post_pre_residual_variance_ratio=variance_ratio,
        diagnostic_flag=bool(abs(sk)>2 or ku>7 or maxr>4 or variance_ratio<.25 or variance_ratio>4))
    return row


def bh_complete_family(pvalues, family_size=510):
    """Unestimable tests remain in the family as p=1; no candidate-only FDR."""
    p=np.asarray(pvalues,float)
    if len(p)!=family_size:
        raise ValueError(f'Expected the complete {family_size}-feature family')
    return multipletests(np.where(np.isfinite(p),p,1.),method='fdr_bh')[1]


def cross_sectional(data):
    rows=[];schema=data.config['schema'];family=schema['morphology_family_510']
    for name in family:
        x=data.medians[:,schema['pyppg'].index(name)]
        positive=x[(data.y==1)&np.isfinite(x)];negative=x[(data.y==0)&np.isfinite(x)]
        row={'feature':name,'positive_n':len(positive),'negative_n':len(negative),'p':np.nan,'delta':np.nan}
        if len(positive) and len(negative):
            u,p=mannwhitneyu(positive,negative,alternative='two-sided',method='asymptotic',use_continuity=True)
            row.update(p=float(p),delta=float(2*u/(len(positive)*len(negative))-1))
        rows.append(row)
    f=pd.DataFrame(rows);f['q']=bh_complete_family(f.p);return f


def longitudinal(frame,schema):
    if frame[['patient_id','stage']].duplicated().any():
        raise ValueError('Provide one patient-stage observation per feature; resolve repeated recordings first')
    if not frame.stage.isin(['pre','post','followup']).all():
        raise ValueError('Longitudinal stages must be pre, post or followup')
    family=schema['morphology_family_510']
    if not set(family)<=set(frame):
        raise ValueError('The complete 510 morphology features must be present, with NaN for unestimable values')
    pp=frame.loc[frame.stage.isin(['pre','post'])];rows=[]
    for name in family:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore');row=fit_one(pp[name].to_numpy(float),pp.stage.eq('post').to_numpy(int),pp.patient_id.to_numpy(str))
        except (ValueError,ZeroDivisionError,np.linalg.LinAlgError) as e:
            row={'status':'unestimable','p':np.nan,'reason':type(e).__name__}
        rows.append({'feature':name,**row})
    mixed=pd.DataFrame(rows);mixed['q']=bh_complete_family(mixed.p)
    tests=[]
    for left,right in [('pre','followup'),('post','followup')]:
        for name in family:
            pairs=frame.pivot(index='patient_id',columns='stage',values=name)
            pairs=pairs.reindex(columns=[left,right]).dropna()
            p=np.nan
            if len(pairs):
                delta=pairs[right].to_numpy()-pairs[left].to_numpy()
                p=1. if np.all(delta==0) else float(wilcoxon(pairs[left],pairs[right],alternative='two-sided',method='auto').pvalue)
            tests.append({'contrast':right+'-'+left,'feature':name,'pairs':len(pairs),'p':p})
    paired=pd.DataFrame(tests)
    paired['q']=paired.groupby('contrast',sort=False).p.transform(lambda x:bh_complete_family(x.to_numpy()))
    return mixed,paired


def pulse_template(channels,fiducials,fs,grid=None):
    """Onset alignment in real time, PPG amplitude scaling shared with derivatives."""
    grid=np.arange(91)/60 if grid is None else np.asarray(grid,float)
    beats=[]
    for row in np.asarray(fiducials):
        if not np.isfinite(row).all():continue
        on,peak,off=map(int,row)
        if not 0<=on<peak<off<channels.shape[1]:continue
        y=np.asarray(channels[:,on:off+1],float).copy();amplitude=np.ptp(y[0])
        if amplitude<=0 or not np.isfinite(y).all():continue
        y[0]-=y[0,0];y/=amplitude;t=np.arange(y.shape[1])/fs
        beats.append(np.stack([np.interp(grid,t,v,left=np.nan,right=np.nan) for v in y]))
    if not beats:return np.full((channels.shape[0],len(grid)),np.nan)
    beats=np.stack(beats)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');out=np.nanmedian(beats,axis=0)
    out[:,np.isfinite(beats[:,0]).sum(axis=0)<np.ceil(len(beats)*.5)]=np.nan
    return out


def population_waveform(patient_templates,iterations=2000,seed=20261022,indices=None):
    """One template per patient; paired stages may share the supplied bootstrap indices."""
    values=np.asarray(patient_templates,float)
    coverage=np.isfinite(values[:,0]).mean(axis=0)>=.9
    rng=np.random.default_rng(seed)
    if indices is None:indices=rng.integers(0,len(values),size=(iterations,len(values)))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');center=np.nanmedian(values,axis=0);boot=np.stack([np.nanmedian(values[ix],axis=0) for ix in indices]);low,high=np.nanpercentile(boot,[2.5,97.5],axis=0)
    for a in [center,low,high]:a[:,~coverage]=np.nan
    return center,low,high,coverage


def grouped_population(index,templates,iterations=2000,seed=20261022):
    """Common coverage across groups and shared patient draws for paired pre/post."""
    summaries={}
    for stage in index.stage.unique():
        summaries[str(stage)]=population_waveform(templates[index.stage.eq(stage)],iterations,seed)
    if {'pre','post'}<=set(index.stage):
        pre=index.loc[index.stage.eq('pre')].reset_index();post=index.loc[index.stage.eq('post')].reset_index()
        ids=sorted(set(pre.patient_id)&set(post.patient_id))
        if ids:
            a=pre.set_index('patient_id').loc[ids,'index'].to_numpy(int);b=post.set_index('patient_id').loc[ids,'index'].to_numpy(int)
            draws=np.random.default_rng(seed).integers(0,len(ids),size=(iterations,len(ids)))
            summaries['paired_pre']=population_waveform(templates[a],iterations,seed,draws)
            summaries['paired_post']=population_waveform(templates[b],iterations,seed,draws)
    common=np.logical_and.reduce([v[3] for v in summaries.values()])
    # Use the common contiguous display interval from onset, not disconnected islands.
    if (~common).any():common[np.flatnonzero(~common)[0]:]=False
    result={}
    for name,(center,low,high,coverage) in summaries.items():
        for a in [center,low,high]:a[:,~common]=np.nan
        result[name]=(center,low,high,common.copy())
    return result


def templates_from_segments(segments,folder):
    """Native complete pulses -> segment medians -> patient-stage medians."""
    from pathlib import Path
    from .features import build_pyppg_signal,fixed_pyppg_fiducials
    from .preprocessing import prepare_signal
    entries={}
    for row in segments.itertuples():
        x=prepare_signal(np.load(Path(folder)/row.signal_path,allow_pickle=False),int(row.fs),row.cohort,getattr(row,'processing_state','raw'))
        x=(x-x.mean())/x.std()
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            signal=build_pyppg_signal(x,int(row.fs),row.segment_id);f=fixed_pyppg_fiducials(signal)
        ppg=np.asarray(signal.ppg);channels=np.stack([ppg,np.gradient(ppg)*row.fs,np.gradient(np.gradient(ppg))*row.fs**2])
        template=pulse_template(channels,f[['on','sp','off']].to_numpy(float),int(row.fs))
        entries.setdefault((row.patient_id,row.stage),[]).append(template)
    rows=[];values=[]
    for (patient,stage),templates in entries.items():
        a=np.stack(templates)
        with warnings.catch_warnings():warnings.simplefilter('ignore');m=np.nanmedian(a,axis=0)
        m[:,np.isfinite(a[:,0]).sum(axis=0)<np.ceil(len(a)*.5)]=np.nan
        rows.append({'patient_id':patient,'stage':stage});values.append(m)
    return pd.DataFrame(rows),np.stack(values)
