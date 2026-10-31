from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from pulseas.morphology import bh_complete_family,pulse_template,population_waveform,grouped_population
from pulseas.stratification import capture_enrichment
from pulseas.evaluation import paired_delta,bootstrap_performance


def test_FDR_uses_510_and_retains_unestimable_tests():
    p=np.r_[.0001,.01,np.full(508,np.nan)]
    q=bh_complete_family(p)
    np.testing.assert_allclose(q,multipletests(np.nan_to_num(p,nan=1),method='fdr_bh')[1])
    assert np.isclose(q[0],.051) and np.all(q[2:]==1)

def test_capture_denominators_and_boundary_ties():
    f=pd.DataFrame({'patient_id':['a','b','c','d','e'],'label':[1,0,1,0,0],'probability':[.9,.8,.8,.3,.1]})
    events=pd.DataFrame({'patient_id':f.patient_id,'pre_tavi':[1,0,np.nan,1,0]})
    row=capture_enrichment(f,[.4],events).iloc[0]
    assert row.patients==3 and row.AS_capture==1 and row.pre_TAVI_capture==.5
    assert np.isclose(row.AS_enrichment,(2/3)/(2/5))

def test_pairing_is_by_patient_ID_and_label():
    f=pd.DataFrame({'patient_id':['a','b','c','d'],'label':[1,0,1,0],'probability':[.9,.8,.7,.1]})
    r=f.iloc[::-1].copy();d=paired_delta(f,r,50)
    assert d['delta_AUROC']==0 and d['CI95']==[0,0]

def test_bootstrap_is_patient_level():
    f=pd.DataFrame({'patient_id':['a','b','c','d'],'label':[1,0,1,0],'probability':[.9,.8,.7,.1]})
    result=bootstrap_performance(f,50,42,True)
    assert result['bootstrap_valid']==50 and result['patients']==4

def test_waveform_has_real_time_and_shared_derivative_scaling():
    t=np.arange(61)/60;wave=np.sin(np.pi*t)**2
    channels=np.stack([wave,np.gradient(wave)*60,np.gradient(np.gradient(wave))*60**2])
    template=pulse_template(channels,np.array([[0,30,60]]),60)
    np.testing.assert_allclose(template[:,:61],channels)
    assert np.isnan(template[:,61:]).all()
    median,low,high,coverage=population_waveform(np.stack([template,template]),20)
    np.testing.assert_allclose(median[:,:61],template[:,:61])
    assert coverage[:61].all() and not coverage[61:].any()


def test_paired_waveforms_share_patient_draws_and_common_interval():
    index=pd.DataFrame({'patient_id':['a','b','b','a'],'stage':['pre','pre','post','post']})
    x=np.stack([np.full((2,91),v) for v in [1.,3.,3.,1.]])
    x[0,:,75:]=np.nan
    groups=grouped_population(index,x,20)
    assert all(np.isnan(a[:,75:]).all() for values in groups.values() for a in values[:3])
    # Matching patient values and draws yield identical paired-stage confidence bands.
    for a,b in zip(groups['paired_pre'][:3],groups['paired_post'][:3]):np.testing.assert_allclose(a[:,:75],b[:,:75])
