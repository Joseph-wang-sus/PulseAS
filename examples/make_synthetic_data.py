"""Generate independent synthetic features and native-rate pulse examples."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd


def generate(output,schema,patients=80,seed=90123):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(seed)
    for cohort,fs in [('EDU',60),('MIMIC',125)]:
        folder=output/cohort;folder.mkdir(exist_ok=True)
        ids=np.array([f'SYN_{cohort}_{i:04d}' for i in range(patients)])
        y=np.r_[np.ones(patients//3,dtype=int),np.zeros(patients-patients//3,dtype=int)];rng.shuffle(y)
        f=pd.DataFrame({'patient_id':ids,'label':y,'age':rng.normal(68,10,patients),'sex_female':rng.integers(0,2,patients),'sbp':rng.normal(130,15,patients),'dbp':rng.normal(75,10,patients),'hr':rng.normal(75,8,patients),'pre_tavi':np.where(y & (rng.random(patients)<.5),1,0)})
        f.loc[rng.choice(patients,5,replace=False),'dbp']=np.nan
        f.to_csv(folder/'patients.csv',index=False)
        rows=[];xp=[];xe=[];waves=[]
        for i,pid in enumerate(ids):
            for k in range(1 if cohort=='EDU' else 2+(i%2)):
                sid=f'{pid}_s{k}';rows.append({'patient_id':pid,'segment_id':sid,'stage':'index'})
                x=rng.normal(size=918);x[:18]+=y[i]*.5;x[rng.random(918)<.025]=np.nan;xp.append(x)
                e=rng.normal(size=512);e[:8]+=y[i]*.3;xe.append(e)
                t=np.arange(1250)/125;waves.append((np.sin(2*np.pi*1.2*t)+.1*rng.normal(size=len(t))).astype(np.float32))
        s=pd.DataFrame(rows);s.to_csv(folder/'segments.csv',index=False)
        np.savez_compressed(folder/'features.npz',segment_id=s.segment_id.to_numpy(str),patient_id=s.patient_id.to_numpy(str),pyppg=np.array(xp),embedding=np.array(xe,dtype=np.float32),encoder_input=np.array(waves))
        # Separate raw examples exercise acquisition/feature adapters; no simulated embeddings are used by extract.
        rawfolder=output/(cohort+'_raw');rawfolder.mkdir(exist_ok=True)
        t=np.arange(15*fs)/fs;raw=65+np.sin(2*np.pi*1.2*t)+.2*np.sin(4*np.pi*1.2*t-.4)
        np.save(rawfolder/'example.npy',raw.astype(np.float32))
        f.iloc[:1].to_csv(rawfolder/'patients.csv',index=False)
        pd.DataFrame([{'patient_id':ids[0],'segment_id':ids[0]+'_raw','signal_path':'example.npy','fs':fs,'cohort':cohort,'stage':'index','processing_state':'raw'}]).to_csv(rawfolder/'segments.csv',index=False)
    rows=[];varying=schema['morphology_family_510'][:3]
    for i in range(12):
        base=rng.normal()
        for stage,shift in [('pre',0.),('post',-.4),('followup',-.35)]:
            if stage=='followup' and i>=8:continue
            row={'patient_id':f'SYN_LONG_{i:03d}','stage':stage,**{name:0. for name in schema['morphology_family_510']}}
            for name in varying:row[name]=base+shift+rng.normal(scale=.3)
            rows.append(row)
    pd.DataFrame(rows).to_csv(output/'longitudinal.csv',index=False)
    (output/'SYNTHETIC.json').write_text(json.dumps({'synthetic':True,'seed':seed,'patients_per_cohort':patients,'embeddings':'simulated features, not AnyPPG output'},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--schema',type=Path,default=Path('configs/feature_schema.json'));args=p.parse_args();generate(args.output,json.loads(args.schema.read_text()))
