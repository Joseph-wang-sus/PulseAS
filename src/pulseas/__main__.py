"""CLI for the compact study pipeline; all paths are supplied by the user."""
from __future__ import annotations
import os
for _name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(_name,'1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from .data import load_config,load_dataset,audit_inputs,write_json


def main():
    parser=argparse.ArgumentParser(description='PulseAS core research analysis')
    parser.add_argument('--config',type=Path,help='YAML configuration (before the subcommand)')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--data',type=Path,required=True);p.add_argument('--cohort',choices=['EDU','MIMIC'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--upstream-extppg',type=Path)
    p=sub.add_parser('extract');p.add_argument('--data',type=Path,required=True);p.add_argument('--cohort',choices=['EDU','MIMIC'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--encoder-code',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p=sub.add_parser('validate');p.add_argument('--data',type=Path,required=True);p.add_argument('--cohort',choices=['EDU','MIMIC'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--models');p.add_argument('--folds',type=Path)
    p=sub.add_parser('transfer');p.add_argument('--source',type=Path,required=True);p.add_argument('--target',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--models');p.add_argument('--roles',type=Path);p.add_argument('--encoder-code',type=Path)
    p=sub.add_parser('evaluate');p.add_argument('--predictions',type=Path);p.add_argument('--patients',type=Path);p.add_argument('--cohort',choices=['EDU','MIMIC'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--data',type=Path);p.add_argument('--folds',type=Path);p.add_argument('--clinical-comparisons',choices=['main','complete','spline'])
    p=sub.add_parser('stratify');p.add_argument('--predictions',type=Path,required=True);p.add_argument('--patients',type=Path);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('morphology');p.add_argument('--data',type=Path);p.add_argument('--cohort',choices=['EDU','MIMIC']);p.add_argument('--longitudinal',type=Path);p.add_argument('--waveforms',type=Path,help='Folder containing waveform segments.csv and relative signal files');p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('demo');p.add_argument('--output',type=Path,default=Path('outputs/demo'))
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    # Editable installs use the repository configs. Non-editable use explicit --config.
    config_path=args.config or root/'configs'/('demo.yaml' if args.command=='demo' else 'study.yaml')
    cfg=load_config(config_path)
    if args.command=='prepare':
        prepare(args,cfg)
    elif args.command=='extract':
        from .features import extract_dataset
        extract_dataset(args.data,args.output,args.cohort,cfg,args.encoder_code,args.checkpoint)
    elif args.command=='validate':
        from .validation import nested_validate
        data=load_dataset(args.data,args.cohort,cfg);nested_validate(data,(args.models or ','.join(cfg['models'])).split(','),args.output,args.folds)
    elif args.command=='transfer':
        from .transfer import transfer
        transfer(load_dataset(args.source,'MIMIC',cfg),load_dataset(args.target,'EDU',cfg),(args.models or ','.join(cfg['transfer_models'])).split(','),args.output,args.roles,args.encoder_code)
    elif args.command=='evaluate':
        from .evaluation import evaluate_file,clinical_comparisons
        if args.clinical_comparisons:
            if not args.data or not args.folds:parser.error('Matched comparisons need --data and --folds')
            folds=json.loads(args.folds.read_text());data=load_dataset(args.data,args.cohort,cfg)
            if folds['patient_id']!=data.ids.tolist():parser.error('Fold patient order differs')
            clinical_comparisons(data,args.output,folds['folds'],args.clinical_comparisons)
        else:
            if not args.predictions:parser.error('--predictions is required')
            evaluate_file(args.predictions,args.output,cfg,args.cohort,args.patients)
    elif args.command=='stratify':
        from .stratification import capture_enrichment
        f=pd.read_csv(args.predictions,dtype={'patient_id':str});p=pd.read_csv(args.patients,dtype={'patient_id':str}) if args.patients else None
        args.output.parent.mkdir(parents=True,exist_ok=True);capture_enrichment(f,pre_tavi=p).to_csv(args.output,index=False)
    elif args.command=='morphology':
        from .morphology import longitudinal,cross_sectional,templates_from_segments,population_waveform
        args.output.mkdir(parents=True,exist_ok=True)
        if args.data:
            if not args.cohort:parser.error('--data needs --cohort')
            cross_sectional(load_dataset(args.data,args.cohort,cfg)).to_csv(args.output/'cross_sectional.csv',index=False)
        if args.longitudinal:
            mixed,paired=longitudinal(pd.read_csv(args.longitudinal,dtype={'patient_id':str}),cfg['schema'])
            mixed.to_csv(args.output/'pre_post_mixed.csv',index=False);paired.to_csv(args.output/'followup_paired.csv',index=False)
        if args.waveforms:
            rows,templates=templates_from_segments(pd.read_csv(args.waveforms/'segments.csv',dtype={'patient_id':str}),args.waveforms)
            rows.to_csv(args.output/'template_index.csv',index=False);np.savez_compressed(args.output/'templates.npz',templates=templates,time_seconds=np.arange(91)/60)
            for stage in rows.stage.unique():
                values=templates[rows.stage.eq(stage)];center,low,high,coverage=population_waveform(values,cfg['bootstrap_iterations'],cfg['seeds']['waveform'])
                np.savez_compressed(args.output/(stage+'_population.npz'),median=center,ci_low=low,ci_high=high,coverage=coverage)
        if not any([args.data,args.longitudinal,args.waveforms]):parser.error('Choose --data, --longitudinal or --waveforms')
    elif args.command=='demo':
        demo(args.output,cfg)


def prepare(args,cfg):
    if (args.data/'features.npz').exists():
        return audit_inputs(args.data,args.output,cfg,args.cohort)
    from .preprocessing import prepare_signal,evaluate_ppg_quality_v2,select_mimic_segments
    if args.data.resolve()==args.output.resolve():raise ValueError('Use a separate prepared-data output folder')
    args.output.mkdir(parents=True,exist_ok=True)
    if args.cohort=='MIMIC':
        if not args.upstream_extppg:raise ValueError('MIMIC parent preparation requires --upstream-extppg')
        parents=pd.read_csv(args.data/'parents.csv',dtype={'patient_id':str,'parent_id':str});rows=[]
        for patient,g in parents.groupby('patient_id',sort=False):
            g=g.sort_values(['parent_id'],kind='stable') if 'start_time' not in g else g.sort_values(['start_time','parent_id'],kind='stable')
            values=[(r.parent_id,np.load(args.data/r.signal_path,allow_pickle=False)) for r in g.itertuples()]
            for i,(_,half,raw) in enumerate(select_mimic_segments(values,args.upstream_extppg,cfg['seeds']['mimic_selection'])):
                sid=f'{patient}_segment_{i:02d}';name=sid+'.npy';np.save(args.output/name,raw)
                rows.append({'patient_id':patient,'segment_id':sid,'signal_path':name,'fs':125,'cohort':'MIMIC','stage':'index','processing_state':'raw'})
        pd.DataFrame(rows).to_csv(args.output/'segments.csv',index=False)
    else:
        f=pd.read_csv(args.data/'segments.csv',dtype={'patient_id':str,'segment_id':str});rows=[];audit=[]
        for r in f.itertuples():
            raw=np.load(args.data/r.signal_path,allow_pickle=False);clean=prepare_signal(raw,int(r.fs),'EDU',getattr(r,'processing_state','raw'))
            quality=evaluate_ppg_quality_v2(clean,raw)
            audit.append({'patient_id':r.patient_id,'segment_id':r.segment_id,**quality})
            if quality['quality_passed']:
                name=r.segment_id+'.npy';np.save(args.output/name,clean)
                rows.append({'patient_id':r.patient_id,'segment_id':r.segment_id,'signal_path':name,'fs':60,'cohort':'EDU','stage':'index','processing_state':'neurokit_cleaned'})
        pd.DataFrame(audit).to_csv(args.output/'signal_quality.csv',index=False)
        pd.DataFrame(rows).to_csv(args.output/'segments.csv',index=False)
    if not rows:raise ValueError('No signals passed preparation; inspect the quality inputs')
    p=pd.read_csv(args.data/'patients.csv',dtype={'patient_id':str});p['included']=p.patient_id.isin(pd.DataFrame(rows).patient_id)
    p.to_csv(args.output/'patient_flow.csv',index=False);p.loc[p.included].drop(columns='included').to_csv(args.output/'patients.csv',index=False)


def demo(output,cfg):
    import importlib.util
    root=Path(__file__).resolve().parents[2];spec=importlib.util.spec_from_file_location('pulseas_synthetic',root/'examples/make_synthetic_data.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    output=Path(output);module.generate(output/'data',cfg['schema'])
    from .validation import nested_validate
    from .transfer import transfer
    from .stratification import capture_enrichment
    from .morphology import longitudinal,cross_sectional
    from .evaluation import evaluate_file
    datasets={}
    for cohort in ['EDU','MIMIC']:
        datasets[cohort]=load_dataset(output/'data'/cohort,cohort,cfg)
        nested_validate(datasets[cohort],cfg['models'],output/'internal'/cohort)
        frame=output/'internal'/cohort/'clinical_fusion_logistic/oof.csv'
        capture_enrichment(pd.read_csv(frame),pre_tavi=datasets[cohort].patients).to_csv(output/(cohort+'_strata.csv'),index=False)
        evaluate_file(frame,output/'evaluation'/cohort,cfg,cohort,output/'data'/cohort/'patients.csv')
    transfer(datasets['MIMIC'],datasets['EDU'],cfg['transfer_models'],output/'transfer')
    cross_sectional(datasets['EDU']).to_csv(output/'cross_sectional.csv',index=False)
    mixed,paired=longitudinal(pd.read_csv(output/'data/longitudinal.csv'),cfg['schema'])
    mixed.to_csv(output/'pre_post.csv',index=False);paired.to_csv(output/'followup.csv',index=False)
    write_json(output/'DEMO_ONLY.json',{'synthetic':True,'clinical_validation':False,'description':'Synthetic feature-level workflow demonstration; embeddings are simulated and no clinical data or pretrained weights are used.'})
    print('Synthetic demo completed:',output)


if __name__=='__main__':
    main()
