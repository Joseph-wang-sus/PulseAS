"""Preserved model primitives extracted from the study analysis code."""
from __future__ import annotations
import hashlib, json, random, warnings
from typing import Any
import numpy as np
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
try:
    import torch
except ImportError:
    torch = None
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
RANDOM_STATE = 42


def upstream_encoder(code_path):
    """Load the user's upstream Net1D implementation without vendoring it."""
    from pathlib import Path
    import importlib.util, sys
    code_path = Path(code_path).resolve()
    spec = importlib.util.spec_from_file_location('_pulseas_upstream_net1d',code_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module; spec.loader.exec_module(module)
    return module.Net1D(in_channels=1,base_filters=64,ratio=1.,filter_list=[64,160,160,400,400,512],m_blocks_list=[2,2,2,3,3,1],kernel_size=3,stride=2,groups_width=16,use_bn=True,use_do=True,verbose=False)


def load_encoder(code_path,checkpoint,expected_code=None,expected_checkpoint=None):
    from .data import sha256
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError('CUDA PyTorch is required for the study encoder')
    if expected_code and sha256(code_path)!=expected_code:
        raise ValueError('Encoder implementation does not match the study hash')
    if expected_checkpoint and sha256(checkpoint)!=expected_checkpoint:
        raise ValueError('AnyPPG checkpoint does not match the study hash')
    model = upstream_encoder(code_path)
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
    model.cuda().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model

if torch is not None:
    class TorchMLP(torch.nn.Module):
        def __init__(self, dimensions: int, hidden: int, dropout: float):
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Linear(dimensions, hidden),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden, 1),
            )

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return self.network(values).squeeze(1)
else:
    class TorchMLP:
        def __init__(self, *args, **kwargs):
            raise ImportError("Install the neural extra to use neural prediction heads")

def method_grid(classifier: str) -> list[dict[str, Any]]:
    if classifier == "extra_trees":
        return [
            {"max_features": "sqrt", "min_samples_leaf": 2},
            {"max_features": 0.2, "min_samples_leaf": 5},
        ]
    if classifier == "xgboost":
        return [
            {"max_depth": 2, "learning_rate": 0.03},
            {"max_depth": 3, "learning_rate": 0.05},
        ]
    if classifier == "logistic_l2":
        return [{"C": 0.1}, {"C": 1.0}]
    if classifier == "torch_mlp":
        return [{"hidden": 64, "dropout": 0.15}]
    raise ValueError(classifier)

def fit_imputation(x: np.ndarray, names: list[str]) -> np.ndarray:
    fill = np.zeros(x.shape[1], dtype=float)
    for column in range(x.shape[1]):
        finite = x[:, column][np.isfinite(x[:, column])]
        if not len(finite):
            fill[column] = 0.0
        elif names[column] == "sex_female":
            counts = np.bincount(finite.astype(int), minlength=2)
            fill[column] = float(np.argmax(counts))
        else:
            fill[column] = float(np.median(finite))
    return fill

def fit_bundle(
    classifier: str,
    params: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    names: list[str],
    seed: int,
) -> dict[str, Any]:
    if classifier in {"xgboost", "torch_mlp"} and (torch is None or not torch.cuda.is_available()):
        raise RuntimeError("CUDA is required for the specified model; CPU fallback is prohibited")
    fill = fit_imputation(x_train, names)
    train = np.where(np.isfinite(x_train), x_train, fill)
    scaler: StandardScaler | None = None
    if classifier in {"logistic_l2", "torch_mlp"}:
        scaler = StandardScaler().fit(train)
        train = scaler.transform(train)
    if classifier == "extra_trees":
        estimator: Any = ExtraTreesClassifier(
            n_estimators=250,
            max_features=params["max_features"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=seed,
        ).fit(train, y_train)
    elif classifier == "xgboost":
        negative = max(1, int(np.sum(y_train == 0)))
        positive = max(1, int(np.sum(y_train == 1)))
        estimator = XGBClassifier(
            n_estimators=180,
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=negative / positive,
            tree_method="hist",
            device="cuda",
            n_jobs=1,
            random_state=seed,
        ).fit(train.astype(np.float32), y_train)
        actual_device = json.loads(estimator.get_booster().save_config())["learner"]["generic_param"]["device"]
        if not str(actual_device).startswith("cuda"):
            raise RuntimeError(f"XGBoost silently changed device to {actual_device}")
    elif classifier == "logistic_l2":
        estimator = LogisticRegression(
            C=float(params["C"]),
            class_weight="balanced",
            solver="liblinear",
            max_iter=5000,
            random_state=seed,
        ).fit(train, y_train)
    elif classifier == "torch_mlp":
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        estimator = TorchMLP(train.shape[1], int(params["hidden"]), float(params["dropout"])).cuda()
        tensor_x = torch.from_numpy(train.astype(np.float32)).cuda()
        tensor_y = torch.from_numpy(y_train.astype(np.float32)).cuda()
        positive = max(1, int(np.sum(y_train == 1)))
        negative = max(1, int(np.sum(y_train == 0)))
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive, device="cuda"))
        optimizer = torch.optim.AdamW(estimator.parameters(), lr=1e-3, weight_decay=1e-3)
        estimator.train()
        for _ in range(80):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(estimator(tensor_x), tensor_y)
            loss.backward()
            optimizer.step()
        estimator.eval()
    else:
        raise ValueError(classifier)
    return {"classifier": classifier, "params": params, "fill": fill, "scaler": scaler, "estimator": estimator, "names": names}

def predict_bundle(bundle: dict[str, Any], x: np.ndarray) -> np.ndarray:
    values = np.where(np.isfinite(x), x, bundle["fill"])
    if bundle["scaler"] is not None:
        values = bundle["scaler"].transform(values)
    if bundle["classifier"] == "torch_mlp":
        model: TorchMLP = bundle["estimator"]
        with torch.no_grad():
            logits = model(torch.from_numpy(values.astype(np.float32)).to(next(model.parameters()).device)).cpu().numpy()
        return expit(logits)
    estimator = bundle["estimator"]
    if bundle["classifier"] == "xgboost":
        values = values.astype(np.float32)
    return estimator.predict_proba(values)[:, 1]

def fit_platt(y: np.ndarray, raw_score: np.ndarray) -> LogisticRegression:
    calibrator = LogisticRegression(C=1e6, solver="liblinear", max_iter=5000, random_state=RANDOM_STATE)
    calibrator.fit(raw_score.reshape(-1, 1), y)
    return calibrator

def apply_platt(calibrator: LogisticRegression, raw_score: np.ndarray) -> np.ndarray:
    return calibrator.predict_proba(raw_score.reshape(-1, 1))[:, 1]


def spline_design(train, values, names):
    """Restricted cubic splines at training 10th/50th/90th percentiles."""
    knots = {}
    for j, name in enumerate(names):
        if name in {'age', 'sbp', 'dbp', 'pp'}:
            k = np.quantile(train[:, j], [.1, .5, .9])
            if len(np.unique(k)) == 3:
                knots[j] = k
    return apply_splines(values, knots), knots


def apply_splines(values, knots):
    columns = []
    for j in range(values.shape[1]):
        x = values[:, j]; columns.append(x)
        if j in knots:
            a,b,c = knots[j]
            term = (np.maximum(x-a,0)**3 - (c-a)/(c-b)*np.maximum(x-b,0)**3 + (b-a)/(c-b)*np.maximum(x-c,0)**3)/(c-a)**2
            columns.append(term)
    return np.column_stack(columns)

def fit_weighted(classifier, params, x, y, weights, summaries, summary_y, names, seed):
    fill = fit_imputation(summaries, names)
    train = np.where(np.isfinite(x), x, fill).astype(np.float32)
    complete = np.where(np.isfinite(summaries), summaries, fill)
    scaler = None
    if classifier in {"logistic_l2", "torch_mlp"}:
        scaler = StandardScaler().fit(complete)
        train = scaler.transform(train).astype(np.float32)
    n0, n1 = int(sum(summary_y == 0)), int(sum(summary_y == 1))
    balanced = weights * np.where(y == 1, len(summary_y)/(2*n1), len(summary_y)/(2*n0))
    if classifier == "extra_trees":
        estimator = ExtraTreesClassifier(n_estimators=250, max_features=params["max_features"],
            min_samples_leaf=params["min_samples_leaf"], n_jobs=8, random_state=seed,
            class_weight=None).fit(train, y, sample_weight=balanced)
    elif classifier == "logistic_l2":
        estimator = LogisticRegression(C=params["C"], solver="liblinear", max_iter=5000,
            random_state=seed).fit(train, y, sample_weight=balanced)
    elif classifier == "xgboost":
        if (torch is None or not torch.cuda.is_available()):
            raise RuntimeError("XGBoost CUDA required")
        estimator = XGBClassifier(n_estimators=180, max_depth=params["max_depth"],
            learning_rate=params["learning_rate"], subsample=.8, colsample_bytree=.8,
            objective="binary:logistic", eval_metric="logloss", scale_pos_weight=n0/n1,
            tree_method="hist", device="cuda", n_jobs=1, random_state=seed)
        estimator.fit(train, y, sample_weight=weights)
        actual = json.loads(estimator.get_booster().save_config())["learner"]["generic_param"]["device"]
        if not actual.startswith("cuda"):
            raise RuntimeError(f"XGBoost device fallback: {actual}")
    elif classifier == "torch_mlp":
        if (torch is None or not torch.cuda.is_available()):
            raise RuntimeError("Neural fusion CUDA required")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        estimator = TorchMLP(train.shape[1], params["hidden"], params["dropout"]).cuda()
        tx = torch.from_numpy(train).cuda()
        ty = torch.from_numpy(y.astype(np.float32)).cuda()
        tw = torch.from_numpy(weights.astype(np.float32)).cuda()
        criterion = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(n0/n1, device="cuda"))
        optimizer = torch.optim.AdamW(estimator.parameters(), lr=1e-3, weight_decay=1e-3)
        estimator.train()
        for _ in range(80):
            optimizer.zero_grad(set_to_none=True)
            loss = (criterion(estimator(tx), ty)*tw).sum()/tw.sum()
            loss.backward()
            optimizer.step()
        estimator.eval()
    else:
        raise ValueError(classifier)
    return {"classifier": classifier, "params": params, "fill": fill, "scaler": scaler,
            "estimator": estimator, "names": names, "patient_weighted": True,
            "preprocessing_fit_patients": len(summary_y)}

def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()
    return RANDOM_STATE + int(digest[:8], 16) % 1_000_000

def threshold_at_minimum_specificity(
    y_true: np.ndarray, scores: np.ndarray, target_specificity: float = 0.60
) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    specificity = 1.0 - fpr
    candidates = np.flatnonzero(np.isfinite(thresholds) & (specificity >= target_specificity))
    if not len(candidates):
        return float(np.nextafter(np.nanmax(scores), np.inf))
    best = sorted(
        candidates,
        key=lambda index: (tpr[index], specificity[index], -thresholds[index]),
        reverse=True,
    )[0]
    return float(thresholds[best])
