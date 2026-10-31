"""AS and independently curated pre-TAVI concentration along OOF AS rankings."""
import math
import numpy as np
import pandas as pd


def capture_enrichment(frame, fractions=(.05,.1,.2,.3,.5,1.), pre_tavi=None):
    if frame.patient_id.duplicated().any():
        raise ValueError('Ranking requires one OOF probability per patient')
    if not np.isfinite(frame.probability).all():
        raise ValueError('Rankings require finite calibrated probabilities')
    f = frame.sort_values('probability', ascending=False, kind='stable').reset_index(drop=True)
    if pre_tavi is not None:
        # Only independently established positive membership is counted. Missing is not a nonoperative label.
        f = f.merge(pre_tavi[['patient_id','pre_tavi']], on='patient_id', how='left', validate='one_to_one')
        if not f.pre_tavi.dropna().isin([0,1]).all():
            raise ValueError('pre_tavi must be curated 0/1/unknown')
    N, E = len(f), int(f.label.sum())
    rows = []
    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError('Selected fractions must lie in (0,1]')
        cutoff = f.probability.iloc[math.ceil(fraction*N)-1]
        selected = f.loc[f.probability.ge(cutoff)]
        n, e = len(selected), int(selected.label.sum())
        row = {'fraction': fraction,'patients': n,'AS_cases_selected':e,'AS_cases_total':E,'AS_capture':e/E if E else np.nan,'AS_proportion':e/n,'AS_enrichment':(e/n)/(E/N) if E else np.nan}
        if pre_tavi is not None:
            T, t = int(f.pre_tavi.eq(1).sum()), int(selected.pre_tavi.eq(1).sum())
            row.update(pre_TAVI_cases_selected=t,pre_TAVI_cases_total=T,pre_TAVI_capture=t/T if T else np.nan,pre_TAVI_proportion=t/n,pre_TAVI_enrichment=(t/n)/(T/N) if T else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)
