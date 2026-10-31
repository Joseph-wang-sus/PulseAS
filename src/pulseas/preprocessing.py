"""Study EDU signal-quality checks; MIMIC uses a separate adapter below."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import warnings

import neurokit2 as nk
import numpy as np


QUALITY_SCHEME_VERSION = "edu_ppg_qc_v2_20260921"


@dataclass(frozen=True)
class PPGQualityThresholds:
    sampling_rate: int = 60
    window_seconds: int = 15
    template_sqi_min: float = 0.90
    sustained_amplitude_shift_max: float = 1.75
    three_second_amplitude_ratio_max: float = 2.25
    local_two_second_amplitude_floor: float = 0.40
    no_finger_value: float = 64.0
    no_finger_ratio_max: float = 0.50
    no_finger_run_max_samples: int = 59
    normalized_step_max: float = 0.50

    @property
    def expected_samples(self) -> int:
        return self.sampling_rate * self.window_seconds

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_ratio(high: float, low: float) -> float:
    if not np.isfinite(high) or not np.isfinite(low):
        return np.nan
    if low <= 1e-12:
        return np.inf if high > 1e-12 else 1.0
    return float(high / low)


def _max_true_run(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0
    padded = np.r_[False, mask, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return int(np.max(changes[1::2] - changes[::2]))


def _robust_amplitude(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan
    return float(np.percentile(finite, 95) - np.percentile(finite, 5))


def _template_sqi(cleaned: np.ndarray, sampling_rate: int) -> tuple[float, str]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            quality = np.asarray(
                nk.ppg_quality(
                    cleaned,
                    sampling_rate=sampling_rate,
                    method="templatematch",
                ),
                dtype=float,
            )
    except Exception as exc:
        return np.nan, f"template_sqi_error:{type(exc).__name__}"
    finite = quality[np.isfinite(quality)]
    if finite.size == 0:
        return np.nan, "template_sqi_all_nan"
    return float(np.mean(finite)), ""


def amplitude_stationarity_metrics(
    cleaned: np.ndarray,
    sampling_rate: int = 60,
) -> dict[str, Any]:
    cleaned = np.asarray(cleaned, dtype=float)
    block_samples = 3 * sampling_rate
    local_samples = 2 * sampling_rate
    local_step = sampling_rate

    block_amplitudes = np.asarray(
        [
            _robust_amplitude(cleaned[start : start + block_samples])
            for start in range(0, len(cleaned), block_samples)
            if start + block_samples <= len(cleaned)
        ],
        dtype=float,
    )
    rolling_amplitudes = np.asarray(
        [
            _robust_amplitude(cleaned[start : start + local_samples])
            for start in range(0, len(cleaned) - local_samples + 1, local_step)
        ],
        dtype=float,
    )

    if block_amplitudes.size:
        block_ratio = _safe_ratio(
            float(np.nanmax(block_amplitudes)),
            float(np.nanmin(block_amplitudes)),
        )
    else:
        block_ratio = np.nan

    sustained_ratios: list[float] = []
    if block_amplitudes.size == 5:
        for split in (2, 3):
            left = float(np.nanmedian(block_amplitudes[:split]))
            right = float(np.nanmedian(block_amplitudes[split:]))
            sustained_ratios.append(_safe_ratio(max(left, right), min(left, right)))
    sustained_ratio = float(np.nanmax(sustained_ratios)) if sustained_ratios else np.nan

    rolling_median = float(np.nanmedian(rolling_amplitudes)) if rolling_amplitudes.size else np.nan
    local_floor_ratio = (
        _safe_ratio(float(np.nanmin(rolling_amplitudes)), rolling_median)
        if rolling_amplitudes.size
        else np.nan
    )
    first_local_ratio = (
        _safe_ratio(float(rolling_amplitudes[0]), float(np.nanmedian(rolling_amplitudes[1:])))
        if rolling_amplitudes.size > 1
        else np.nan
    )
    last_local_ratio = (
        _safe_ratio(float(rolling_amplitudes[-1]), float(np.nanmedian(rolling_amplitudes[:-1])))
        if rolling_amplitudes.size > 1
        else np.nan
    )

    return {
        "three_second_amplitudes": block_amplitudes.tolist(),
        "three_second_amplitude_ratio": block_ratio,
        "sustained_amplitude_shift_ratio": sustained_ratio,
        "local_two_second_amplitude_floor_ratio": local_floor_ratio,
        "first_two_second_amplitude_ratio": first_local_ratio,
        "last_two_second_amplitude_ratio": last_local_ratio,
    }


def evaluate_ppg_quality_v2(
    cleaned_signal: np.ndarray,
    raw_signal: np.ndarray | None = None,
    precomputed_template_sqi: float | None = None,
    thresholds: PPGQualityThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or PPGQualityThresholds()
    cleaned = np.asarray(cleaned_signal, dtype=float)
    raw = cleaned if raw_signal is None else np.asarray(raw_signal, dtype=float)
    expected = thresholds.expected_samples

    cleaned_finite = bool(len(cleaned) == expected and np.isfinite(cleaned).all())
    raw_finite = bool(len(raw) == expected and np.isfinite(raw).all())
    cleaned_std = float(np.std(cleaned)) if cleaned_finite else np.nan
    raw_range = float(np.ptp(raw)) if raw_finite else np.nan
    raw_max_step = float(np.max(np.abs(np.diff(raw)))) if raw_finite else np.nan
    raw_normalized_step = (
        raw_max_step / raw_range if raw_finite and raw_range > 0 else np.nan
    )
    zero_ratio = float(np.mean(raw == 0.0)) if raw_finite else np.nan
    no_finger_ratio = (
        float(np.mean(raw == thresholds.no_finger_value)) if raw_finite else np.nan
    )
    no_finger_run = (
        _max_true_run(raw == thresholds.no_finger_value) if raw_finite else 0
    )

    if precomputed_template_sqi is None:
        if cleaned_finite and cleaned_std > 1e-12:
            template_sqi, template_error = _template_sqi(cleaned, thresholds.sampling_rate)
        else:
            template_sqi, template_error = np.nan, ""
    else:
        template_sqi = float(precomputed_template_sqi)
        template_error = ""

    if cleaned_finite and cleaned_std > 1e-12:
        stationarity = amplitude_stationarity_metrics(cleaned, thresholds.sampling_rate)
    else:
        stationarity = {
            "three_second_amplitudes": [],
            "three_second_amplitude_ratio": np.nan,
            "sustained_amplitude_shift_ratio": np.nan,
            "local_two_second_amplitude_floor_ratio": np.nan,
            "first_two_second_amplitude_ratio": np.nan,
            "last_two_second_amplitude_ratio": np.nan,
        }

    reasons: list[str] = []
    if len(cleaned) != expected or len(raw) != expected:
        reasons.append("not_900_samples")
    if not cleaned_finite or not raw_finite:
        reasons.append("nonfinite_signal")
    if cleaned_finite and cleaned_std <= 1e-12:
        reasons.append("constant_signal")
    if raw_finite and zero_ratio > 0:
        reasons.append("contains_zero")
    if raw_finite and no_finger_ratio > thresholds.no_finger_ratio_max:
        reasons.append("no_finger_ratio_over_0_5")
    if raw_finite and no_finger_run > thresholds.no_finger_run_max_samples:
        reasons.append("no_finger_run_at_least_60")
    if (
        raw_finite
        and np.isfinite(raw_normalized_step)
        and raw_normalized_step > thresholds.normalized_step_max
    ):
        reasons.append("abrupt_normalized_step_over_0_5")
    if template_error:
        reasons.append(template_error)
    elif not np.isfinite(template_sqi) or template_sqi < thresholds.template_sqi_min:
        reasons.append("template_sqi_below_0_90")
    if (
        np.isfinite(stationarity["sustained_amplitude_shift_ratio"])
        and stationarity["sustained_amplitude_shift_ratio"]
        > thresholds.sustained_amplitude_shift_max
    ):
        reasons.append("sustained_amplitude_shift")
    if (
        np.isfinite(stationarity["three_second_amplitude_ratio"])
        and stationarity["three_second_amplitude_ratio"]
        > thresholds.three_second_amplitude_ratio_max
    ):
        reasons.append("extreme_three_second_amplitude_imbalance")
    if (
        np.isfinite(stationarity["local_two_second_amplitude_floor_ratio"])
        and stationarity["local_two_second_amplitude_floor_ratio"]
        < thresholds.local_two_second_amplitude_floor
    ):
        reasons.append("local_two_second_signal_dropout")

    return {
        "quality_scheme": QUALITY_SCHEME_VERSION,
        "quality_passed": not reasons,
        "quality_exclusion_reasons": ";".join(reasons),
        "template_sqi_mean": template_sqi,
        "zero_ratio": zero_ratio,
        "no_finger_64_ratio": no_finger_ratio,
        "max_consecutive_64": no_finger_run,
        "signal_range": raw_range,
        "signal_std_cleaned": cleaned_std,
        "max_abs_step": raw_max_step,
        "max_abs_step_over_range": raw_normalized_step,
        **stationarity,
    }


def prepare_signal(values, fs, cohort, processing_state='raw'):
    """Return feature input without applying the acquisition cleaner twice."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) != round(15*fs) or not np.isfinite(values).all() or values.std() <= 0:
        raise ValueError('Expected a finite, nonconstant, native-rate 15-second recording')
    if cohort == 'MIMIC':
        if fs != 125 or processing_state != 'raw':
            raise ValueError('MIMIC study input is raw 125-Hz physical PLETH')
        return values.copy()
    if cohort != 'EDU' or fs != 60:
        raise ValueError('EDU study input is 60 Hz; use an explicit external adapter for other rates')
    if processing_state == 'raw':
        return np.asarray(nk.ppg_clean(values, sampling_rate=60, method='elgendi'))
    if processing_state == 'neurokit_cleaned':
        return values.copy()
    raise ValueError('processing_state must be raw or neurokit_cleaned')


def encoder_input(values, fs):
    """Preserved full-15s to 60 Hz, central 10s to 125 Hz checkpoint adapter."""
    from scipy.signal import resample_poly
    x = np.asarray(values, dtype=np.float32)
    if fs == 125:
        x = resample_poly(x, 12, 25).astype(np.float32)
    elif fs != 60:
        raise ValueError('Study encoder adapter accepts only 60 or 125 Hz')
    if x.shape != (900,) or not np.isfinite(x).all():
        raise ValueError('Expected a complete 15-second input')
    x = resample_poly(x[150:750], 25, 12).astype(np.float32)
    scale = x.std()
    return ((x-x.mean())/(scale if scale >= 1e-6 else 1.)).astype(np.float32)


def mimic_parent_quality(raw, fs, upstream_directory):
    """Call unchanged Ext-PPG SQI on all three native 10-second subwindows."""
    import importlib, sys
    from pathlib import Path
    upstream_directory = str(Path(upstream_directory).resolve())
    if upstream_directory not in sys.path:
        sys.path.insert(0, upstream_directory)
    utils = importlib.import_module('utils')
    sqi = importlib.import_module('sqi_utils')
    for module in [utils, sqi]:
        if Path(module.__file__).resolve().parent != Path(upstream_directory):
            raise ValueError('Conflicting upstream Ext-PPG module already imported')
    raw = np.asarray(raw)
    if fs != 125 or len(raw) != 30*fs or not np.isfinite(raw).all():
        return {'passed': False, 'reason': 'invalid_native_parent'}
    code = int(utils.validate_window(raw,fs,signal_type='PLETH'))
    if code != 4:
        return {'passed': False, 'reason': 'upstream_technical_failure','upstream_code':code}
    reports = [sqi.listen_sqi2(raw[i*10*fs:(i+1)*10*fs],signal_type='PLETH',fs=fs) for i in range(3)]
    return {'passed': all(int(v['sqi'])==1 for v in reports), 'subwindow_codes':[int(v['sqi']) for v in reports]}


def select_mimic_segments(parents, upstream_directory, seed=42, maximum=20):
    """Chronological endpoint/midpoint traversal; alternating native 15s halves."""
    import hashlib, json
    accepted = []
    for key, raw in parents:
        report = mimic_parent_quality(raw,125,upstream_directory)
        if report['passed']:
            accepted.append((key,np.asarray(raw,dtype=np.float32)))
    order, seen, level = [], set(), 1
    while len(order) < len(accepted):
        for i in np.linspace(0,len(accepted)-1,level,dtype=int):
            if int(i) not in seen:
                seen.add(int(i));order.append(int(i))
        level = max(2,2*level-1)
    output, hashes = [], set()
    for phase in (0,1):
        for i in order:
            key,raw = accepted[i]
            digest = hashlib.sha256(json.dumps([key,seed],sort_keys=True,default=str).encode()).hexdigest()
            half = (int(digest[:8],16)%2)^phase
            x = raw[half*1875:(half+1)*1875]
            whash = hashlib.sha256(x.astype('<f4').tobytes()).hexdigest()
            if whash not in hashes and len(output)<maximum:
                hashes.add(whash);output.append((key,half,x))
    return output
