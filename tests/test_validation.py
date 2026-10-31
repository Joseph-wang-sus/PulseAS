import copy,json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from pulseas.data import Dataset,aggregate,load_config
from pulseas.validation import select_top10,make_splits,check_splits,fit_model,predict_raw
from pulseas.transfer import make_roles,check_roles,anchored_loss,update_model
from pulseas.models import threshold_at_minimum_specificity

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def dataset():
    cfg=load_config(ROOT/'configs/demo.yaml');rng=np.random.default_rng(17);n=60
    patients=pd.DataFrame({'patient_id':[f'SYN_TEST_{i}' for i in range(n)],'label':np.arange(n)%2,'age':rng.normal(65,10,n),'sex_female':np.arange(n)%2,'sbp':rng.normal(130,10,n),'dbp':rng.normal(75,8,n),'hr':rng.normal(70,5,n)})
    pi=np.repeat(np.arange(n),1+np.arange(n)%3)
    segments=pd.DataFrame({'segment_id':[f'SYN_SEG_{i}' for i in range(len(pi))],'patient_id':patients.patient_id.to_numpy()[pi],'stage':'index'})
    return Dataset(patients,segments,rng.normal(size=(len(pi),918)),rng.normal(size=(len(pi),512)),'MIMIC',cfg)

def test_patient_equal_aggregation_and_weights(dataset):
    np.testing.assert_allclose(np.bincount(dataset.pi,weights=dataset.weight),1.)
    np.testing.assert_allclose(aggregate(np.array([.2,.3,.8]),np.array([0,1,1]),2),[.2,.55])

def test_patient_and_duplicate_groups_are_never_split(dataset):
    folds=make_splits(dataset,3,2);check_splits(dataset,folds)
    bad=copy.deepcopy(folds);bad[0]['inner'][0]['valid'].append(bad[0]['test'][0])
    with pytest.raises(ValueError):check_splits(dataset,bad)

def test_feature_selection_does_not_read_test_patients(dataset):
    train=np.arange(40);test=np.arange(40,60)
    a=fit_model(dataset,'clinical_fusion_logistic',{'C':.1},train,17)
    before=predict_raw(a,dataset,train)
    dataset.pyppg[dataset.pi>=40]=1e9;dataset.medians[test]=1e9;dataset.y[test]=1-dataset.y[test]
    b=fit_model(dataset,'clinical_fusion_logistic',{'C':.1},train,17)
    assert a['selected']==b['selected']
    np.testing.assert_array_equal(a['bundle']['fill'],b['bundle']['fill'])
    np.testing.assert_allclose(before,predict_raw(b,dataset,train),atol=0,rtol=0)
    domains=dataset.config['schema']['domains'];names=dataset.config['schema']['pyppg']
    assert len({domains[names[i]] for i in a['selected']})==10

def test_threshold_is_selected_on_development_scores():
    y=np.array([0,0,0,0,1,1]);p=np.array([.1,.2,.3,.8,.4,.9])
    t=threshold_at_minimum_specificity(y,p,.6)
    assert np.mean(p[y==0]<t)>=.6
    assert np.mean(p[y==1]>=t)==1.

def test_transfer_test_is_disjoint_from_adaptation(dataset):
    target=copy.deepcopy(dataset);target.cohort='EDU';target.patients.patient_id='TARGET_'+target.patients.patient_id;target.segments.patient_id='TARGET_'+target.segments.patient_id;target.__post_init__()
    roles=make_roles(dataset,target);check_roles(dataset,target,roles)
    assert len(roles['B_adapt'])==12
    bad=copy.deepcopy(roles);bad['B_train'].append(bad['B_test'][0])
    with pytest.raises(ValueError):check_roles(dataset,target,bad)

def test_anchored_loss_gradient():
    from scipy.optimize._numdiff import approx_derivative
    rng=np.random.default_rng(5);x=rng.normal(size=(12,3));y=np.arange(12)%2;w=np.ones(12)/12;origin=np.zeros(4);theta=rng.normal(size=4)
    loss,gradient=anchored_loss(theta,x,y,w,origin,.1)
    actual=approx_derivative(lambda t:anchored_loss(t,x,y,w,origin,.1)[0],theta)
    np.testing.assert_allclose(gradient,actual,rtol=1e-6,atol=1e-8)

def test_AF_is_not_a_model_predictor(dataset):
    with pytest.raises(ValueError):dataset.clinical(['age','af'])

@pytest.mark.gpu
def test_xgboost_continues_source_trees_and_keeps_fill(dataset):
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():pytest.skip('Study XGBoost requires CUDA')
    source=fit_model(dataset,'fusion_top10',{'max_depth':2,'learning_rate':.03},np.arange(40),17)
    before=source['bundle']['estimator'].get_booster().get_dump()
    updated=update_model(source,dataset,dataset.ids[40:].tolist(),{'extra_trees':20},61007)
    after=updated['bundle']['estimator'].get_booster().get_dump()
    assert after[:len(before)]==before and len(after)==len(before)+20
    np.testing.assert_array_equal(source['bundle']['fill'],updated['bundle']['fill'])

@pytest.mark.gpu
def test_neural_update_preserves_source_head_and_cached_encoder_outputs(dataset):
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():pytest.skip('Study neural update requires CUDA')
    source=fit_model(dataset,'neural_fusion',{'hidden':64,'dropout':.15},np.arange(40),17)
    state={k:v.detach().cpu().clone() for k,v in source['bundle']['estimator'].state_dict().items()};embedding=dataset.embedding.copy()
    update_model(source,dataset,dataset.ids[40:].tolist(),{'lr':.001,'epochs':2},61007)
    for k,v in source['bundle']['estimator'].state_dict().items():assert torch.equal(v.cpu(),state[k])
    np.testing.assert_array_equal(dataset.embedding,embedding)
