"""Portable input contracts and patient/segment alignment."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib, json, warnings
import numpy as np
import pandas as pd
import yaml


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x), allow_nan=False) + '\n', encoding='utf-8')


def load_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    schema_path = (path.parent / cfg['feature_schema']).resolve()
    cfg['schema'] = json.loads(schema_path.read_text(encoding='utf-8'))
    cfg['config_sha256'] = sha256(path)
    cfg['schema_sha256'] = sha256(schema_path)
    return cfg


def patient_medians(matrix, patient_index, count):
    result = np.full((count, matrix.shape[1]), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for i in range(count):
            result[i] = np.nanmedian(matrix[patient_index == i], axis=0)
    return result


def aggregate(scores, patient_index, count):
    n = np.bincount(patient_index, minlength=count)
    result = np.full(count, np.nan)
    np.divide(np.bincount(patient_index, weights=scores, minlength=count), n, out=result, where=n > 0)
    return result


@dataclass
class Dataset:
    patients: pd.DataFrame
    segments: pd.DataFrame
    pyppg: np.ndarray
    embedding: np.ndarray
    cohort: str
    config: dict
    wave: np.ndarray | None = None

    def __post_init__(self):
        self.patients = self.patients.reset_index(drop=True).copy()
        self.segments = self.segments.reset_index(drop=True).copy()
        if self.patients.patient_id.isna().any() or self.segments[['patient_id','segment_id']].isna().any().any():
            raise ValueError('Patient and segment IDs cannot be missing')
        self.ids = self.patients.patient_id.astype(str).to_numpy()
        if not self.patients.patient_id.is_unique or not self.segments.segment_id.is_unique:
            raise ValueError('Patient and segment identifiers must each be unique in their tables')
        labels = pd.to_numeric(self.patients.label, errors='raise').to_numpy(dtype=float, copy=True)
        if not np.isfinite(labels).all() or not set(labels) <= {0, 1}:
            raise ValueError('AS labels must be observed binary values; exclude unknown AS before analysis')
        self.y = labels.astype(int)
        self.pi = pd.Index(self.ids).get_indexer(self.segments.patient_id.astype(str))
        if len(self.pi) != len(self.pyppg) or len(self.pi) != len(self.embedding) or (self.pi < 0).any():
            raise ValueError('Segment arrays and patient table are not aligned')
        counts = np.bincount(self.pi, minlength=len(self.ids))
        if (counts == 0).any():
            raise ValueError('Every analysis patient needs at least one jointly usable index segment')
        schema = self.config['schema']['pyppg']
        if self.pyppg.shape != (len(self.pi), len(schema)) or self.embedding.shape != (len(self.pi), 512):
            raise ValueError('Expected 918 morphology columns in schema order and 512 embedding columns')
        if not np.isfinite(self.embedding).all() or np.isinf(self.pyppg).any():
            raise ValueError('Embeddings must be finite; morphology missing values must be NaN, not infinity')
        if not np.isfinite(self.pyppg).any(axis=1).all():
            raise ValueError('Every included segment needs at least one usable morphology feature')
        self.weight = 1.0 / counts[self.pi]
        self.medians = patient_medians(self.pyppg, self.pi, len(self.ids))
        self.emb_medians = patient_medians(self.embedding, self.pi, len(self.ids))
        self.groups = self.patients.get('duplicate_group', self.patients.patient_id).astype(str).to_numpy()

    def index(self, ids):
        result = pd.Index(self.ids).get_indexer(np.asarray(ids, dtype=str))
        if (result < 0).any():
            raise ValueError('Requested patients outside this dataset')
        return result

    def clinical(self, columns=None):
        columns = columns or self.config['clinical'][self.cohort]
        if any('af' in c.lower() for c in columns):
            raise ValueError('AF and AF missingness are not predictors in this study')
        values = []
        for name in columns:
            if name == 'pp':
                values.append(pd.to_numeric(self.patients.sbp, errors='coerce').to_numpy() - pd.to_numeric(self.patients.dbp, errors='coerce').to_numpy())
            else:
                values.append(pd.to_numeric(self.patients[name], errors='coerce').to_numpy(float))
        x = np.column_stack(values)
        return np.hstack([x, (~np.isfinite(x)).astype(float)]), list(columns) + [c + '_missing' for c in columns]

    def matrix(self, representation, selected, clinical_columns=None):
        clinical, cnames = self.clinical(clinical_columns)
        # EDU classifiers use patient-level features; MIMIC PPG models use segments.
        pooled = self.cohort == 'EDU' and not self.config.get('force_segments', False)
        xp = self.medians if pooled else self.pyppg
        xe = self.emb_medians if pooled else self.embedding
        pi = np.arange(len(self.ids)) if pooled else self.pi
        weight = np.ones(len(self.ids)) if pooled else self.weight
        if representation == 'clinical':
            return clinical, clinical, np.arange(len(self.ids)), np.ones(len(self.ids)), cnames
        if representation == 'all_pyppg':
            return xp, self.medians, pi, weight, self.config['schema']['pyppg']
        parts, summaries = [xe], [self.emb_medians]
        names = [f'embedding_{i}' for i in range(512)]
        if representation in ('fusion', 'clinical_fusion'):
            parts.append(xp[:, selected]); summaries.append(self.medians[:, selected])
            names += [self.config['schema']['pyppg'][i] for i in selected]
        if representation == 'clinical_fusion':
            parts.append(clinical[pi]); summaries.append(clinical); names += cnames
        return np.hstack(parts), np.hstack(summaries), pi, weight, names


def load_dataset(folder, cohort, config):
    folder = Path(folder)
    patients = pd.read_csv(folder / 'patients.csv', dtype={'patient_id': str})
    segments = pd.read_csv(folder / 'segments.csv', dtype={'patient_id': str, 'segment_id': str})
    if 'stage' in segments and not segments.stage.eq('index').all():
        raise ValueError('Screening inputs must contain only index segments; use separate longitudinal inputs')
    with np.load(folder / 'features.npz', allow_pickle=False) as z:
        if not np.array_equal(z['segment_id'].astype(str), segments.segment_id.to_numpy(str)):
            raise ValueError('features.npz segment order differs from segments.csv')
        if not np.array_equal(z['patient_id'].astype(str), segments.patient_id.to_numpy(str)):
            raise ValueError('Feature-to-patient mapping differs from segments.csv')
        wave = z['encoder_input'].copy() if 'encoder_input' in z else None
        data = Dataset(patients, segments, z['pyppg'].copy(), z['embedding'].copy(), cohort, config, wave)
    data.input_hashes = {name:sha256(folder/name) for name in ['patients.csv','segments.csv','features.npz']}
    return data


def subset(data, indices):
    indices = np.asarray(indices, int)
    ids = data.ids[indices]
    mask = np.isin(data.pi, indices)
    return Dataset(data.patients.iloc[indices], data.segments.loc[mask], data.pyppg[mask], data.embedding[mask], data.cohort, data.config, None if data.wave is None else data.wave[mask])


def audit_inputs(folder, output, config, cohort):
    folder, output = Path(folder), Path(output)
    data = load_dataset(folder, cohort, config)
    write_json(output, {'cohort': cohort, 'patients': len(data.ids), 'AS_cases': int(data.y.sum()), 'segments': len(data.pi), 'patient_equal_weights': bool(np.allclose(np.bincount(data.pi, weights=data.weight), 1)), 'input_hashes': {name: sha256(folder/name) for name in ['patients.csv', 'segments.csv', 'features.npz']}, 'config_sha256': config['config_sha256']})
    return data
