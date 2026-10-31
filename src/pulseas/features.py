"""Native-rate pyPPG extraction using the study fiducial adapter."""
from __future__ import annotations
import contextlib, io, json, os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from dotmap import DotMap
from pyPPG import PPG, Fiducials
import pyPPG.fiducials as pyppg_fiducials
import pyPPG.preproc as pyppg_preproc
import pyPPG.biomarkers as pyppg_biomarkers
import pyPPG.ppg_sqi as pyppg_sqi
def add_numeric_stats(features, prefix, values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    features[f"{prefix}_count"] = int(len(values))
    if len(values) == 0:
        for name in ("mean", "median", "std", "min", "max", "p05", "p25", "p75", "p95"):
            features[f"{prefix}_{name}"] = np.nan
        return

    features[f"{prefix}_mean"] = float(np.mean(values))
    features[f"{prefix}_median"] = float(np.median(values))
    features[f"{prefix}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    features[f"{prefix}_min"] = float(np.min(values))
    features[f"{prefix}_max"] = float(np.max(values))
    for percentile in (5, 25, 75, 95):
        features[f"{prefix}_p{percentile:02d}"] = float(np.percentile(values, percentile))

def add_ibi_features(features, peaks, sampling_rate, prefix):
    peaks = np.asarray(peaks, dtype=float)
    peaks = peaks[np.isfinite(peaks)]
    features[f"{prefix}_peak_count"] = int(len(peaks))
    if len(peaks) < 2:
        add_numeric_stats(features, f"{prefix}_ibi_ms", [])
        add_numeric_stats(features, f"{prefix}_bpm_from_ibi", [])
        return np.array([])

    ibi_ms = np.diff(peaks) / sampling_rate * 1000.0
    bpm = 60000.0 / ibi_ms
    add_numeric_stats(features, f"{prefix}_ibi_ms", ibi_ms)
    add_numeric_stats(features, f"{prefix}_bpm_from_ibi", bpm)
    return ibi_ms

def flatten_dataframe(prefix, df):
    flat = {}
    if df is None or len(df) == 0:
        return flat

    if isinstance(df, pd.Series):
        df = df.to_frame().T

    for index, row in df.iterrows():
        index_name = str(index).strip().replace(" ", "_")
        for col, value in row.items():
            key = f"{prefix}_{index_name}_{col}"
            flat[key] = value
    return flat

def padded(values, length):
    output = np.full(length, np.nan)
    values = np.asarray(values, dtype=float)
    output[: min(len(values), length)] = values[: min(len(values), length)]
    return output

def fixed_pyppg_fiducials(s):
    extractor = pyppg_fiducials.FpCollection(s=s)
    peaks, onsets = extractor.get_peak_onset("PPGdet")
    length = len(peaks)
    if length == 0:
        return pd.DataFrame()

    dicrotic_notch = extractor.get_dicrotic_notch(peaks, onsets)
    vpg_fp = extractor.get_vpg_fiducials(onsets)
    apg_fp = extractor.get_apg_fiducials(onsets, peaks)
    jpg_fp = extractor.get_jpg_fiducials(onsets, apg_fp)
    diastolic_peak = extractor.get_diastolic_peak(onsets, dicrotic_notch, apg_fp.e)

    data = {
        "on": padded(onsets, length),
        "sp": padded(peaks, length),
        "dn": padded(dicrotic_notch, length),
        "dp": padded(diastolic_peak, length),
    }
    for temp_sig in (vpg_fp, apg_fp, jpg_fp):
        for key in temp_sig.keys():
            data[key] = padded(temp_sig[key].values, length)

    fiducials = pd.DataFrame(data)
    fiducials = extractor.correct_fiducials(fiducials, s.correction)
    fiducials = fiducials.astype("Int64")

    offsets = fiducials.on[1:].copy()
    offsets.index = offsets.index - 1
    fiducials = fiducials.drop(len(fiducials) - 1)
    fiducials = fiducials.rename_axis("pulse_index")
    fiducials.insert(4, "off", offsets)
    return fiducials

def build_pyppg_signal(values, sampling_rate, name):
    allow_short = os.environ.get("PYPPG_ALLOW_SHORT_SIGNAL", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    values = np.asarray(values, dtype=float)
    if allow_short and len(values) == sampling_rate * 10:
        # pyPPG's PPGdet uses a 10 s internal window and fails at exactly
        # fs*10 samples because no processing window is entered. Repeating
        # the last sample adds 1/fs second only for fiducial extraction.
        values = np.concatenate([values, values[-1:]])
    signal = DotMap()
    signal.v = values
    signal.fs = sampling_rate
    signal.name = name
    signal.start_sig = 0
    signal.end_sig = len(values)
    signal.filtering = True
    signal.fL = 0.5000001
    signal.fH = 12
    signal.order = 4
    signal.sm_wins = {"ppg": 50, "vpg": 10, "apg": 10, "jpg": 10}
    signal.correction = pd.DataFrame()

    prep = pyppg_preproc.Preprocess(
        fL=signal.fL,
        fH=signal.fH,
        order=signal.order,
        sm_wins=signal.sm_wins,
    )
    signal.ppg, signal.vpg, signal.apg, signal.jpg = prep.get_signals(s=signal)

    correction = pd.DataFrame()
    correction.loc[0, ["on", "dn", "dp", "v", "w", "f"]] = True
    correction.loc[0, ["dn"]] = False
    signal.correction = correction
    return PPG(s=signal, check_ppg_len=not allow_short)

def extract_pyppg_features(values, sampling_rate, sample_name, meta):
    features = dict(meta)
    signal = build_pyppg_signal(values, sampling_rate, sample_name)
    fiducials = fixed_pyppg_fiducials(signal)

    features["pyppg_fiducial_pulse_count"] = int(len(fiducials))
    if len(fiducials) > 0 and "sp" in fiducials:
        valid_peaks = pd.to_numeric(fiducials["sp"], errors="coerce").dropna().to_numpy()
        add_ibi_features(features, valid_peaks, sampling_rate, "pyppg")
        if os.environ.get("SKIP_PYPPG_SQI", "").strip().lower() in {"1", "true", "yes"}:
            features["pyppg_sqi_error"] = "skipped_by_SKIP_PYPPG_SQI"
        else:
            try:
                sqi = pyppg_sqi.get_ppgSQI(ppg=signal.ppg, fs=signal.fs, annotation=valid_peaks)
                add_numeric_stats(features, "pyppg_sqi", sqi)
            except Exception as exc:
                features["pyppg_sqi_error"] = str(exc)

    valid_fiducials = fiducials.dropna().reset_index(drop=True)
    if len(valid_fiducials) > 0:
        try:
            valid_fiducials = valid_fiducials.astype(int)
            fp = Fiducials(fp=valid_fiducials)
            _, _, biomarker_stats = pyppg_biomarkers.BmCollection(s=signal, fp=fp).get_biomarkers()
            for category, stats_df in biomarker_stats.items():
                features.update(flatten_dataframe(f"pyppg_{category}", stats_df))
        except Exception as exc:
            features["pyppg_biomarker_error"] = str(exc)

    return features


def native_features(values, fs, segment_id, schema):
    """No external pyPPG filter: z-score once, then pyPPG preprocessing once."""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or values.std() == 0:
        raise ValueError('Invalid feature input')
    normalized = (values-values.mean())/values.std(ddof=0)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        out = extract_pyppg_features(normalized,fs,segment_id,{})
    result = np.array([out.get(k,np.nan) for k in schema['pyppg']],dtype=float)
    result[~np.isfinite(result)] = np.nan
    return result, {'finite_features':int(np.isfinite(result).sum()), 'error':str(out.get('pyppg_biomarker_error',''))}


def extract_dataset(folder, output, cohort, config, encoder_code, checkpoint):
    from .preprocessing import prepare_signal, encoder_input
    from .models import load_encoder
    from .data import write_json, sha256
    import torch
    folder, output = Path(folder), Path(output); output.mkdir(parents=True,exist_ok=True)
    if folder.resolve() == output.resolve():
        raise ValueError('Extraction output must differ from the source folder')
    patients = pd.read_csv(folder/'patients.csv',dtype={'patient_id':str})
    segments = pd.read_csv(folder/'segments.csv',dtype={'patient_id':str,'segment_id':str})
    if 'stage' in segments and not segments.stage.eq('index').all():
        raise ValueError('Keep longitudinal and screening inputs separate')
    encoder = load_encoder(encoder_code,checkpoint,config.get('expected_encoder_code_sha256'),config.get('expected_checkpoint_sha256'))
    xp,xe,waves,kept,audit = [],[],[],[],[]
    for row in segments.itertuples():
        try:
            raw = np.load(folder/row.signal_path,allow_pickle=False)
            x = prepare_signal(raw,int(row.fs),cohort,getattr(row,'processing_state','raw'))
            features, info = native_features(x,int(row.fs),row.segment_id,config['schema'])
            if info['error'] or not info['finite_features']:
                raise ValueError('No usable pyPPG biomarkers: '+info['error'])
            wave = encoder_input(x,int(row.fs))
            with torch.inference_mode():
                embedding = encoder(torch.from_numpy(wave[None,None,:]).cuda()).cpu().numpy()[0]
            if embedding.shape != (512,) or not np.isfinite(embedding).all():
                raise ValueError('Invalid frozen encoder output')
            xp.append(features);xe.append(embedding);waves.append(wave);kept.append(row.segment_id)
            audit.append({'segment_id':row.segment_id,'patient_id':row.patient_id,'usable':True,**info})
        except Exception as e:
            audit.append({'segment_id':row.segment_id,'patient_id':row.patient_id,'usable':False,'error':type(e).__name__+': '+str(e)})
    pd.DataFrame(audit).to_csv(output/'extraction_flow.csv',index=False)
    if not kept:
        raise ValueError('No jointly usable segments; see extraction_flow.csv')
    selected = segments.set_index('segment_id').loc[kept].reset_index()
    included = patients.patient_id.isin(selected.patient_id)
    patients.assign(included=included).to_csv(output/'patient_flow.csv',index=False)
    patients.loc[included].to_csv(output/'patients.csv',index=False)
    selected.to_csv(output/'segments.csv',index=False)
    np.savez_compressed(output/'features.npz',segment_id=selected.segment_id.to_numpy(str),patient_id=selected.patient_id.to_numpy(str),pyppg=np.stack(xp),embedding=np.stack(xe),encoder_input=np.stack(waves))
    write_json(output/'extraction_manifest.json',{'cohort':cohort,'input_segments':len(segments),'included_segments':len(kept),'included_patients':int(included.sum()),'checkpoint_sha256':sha256(checkpoint),'encoder_sha256':sha256(encoder_code),'config_sha256':config['config_sha256']})
