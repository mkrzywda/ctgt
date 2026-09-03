"""Auto-generated from the CTGP research monolith — do not edit by hand."""

import math
import time
import numpy as np
import torch
import torch.nn.functional as F
from ctgp_lib.config import CONV_NAMES, INIT_OPTIONS, SURROGATE_FEATURE_DIM
from ctgp_lib.device import get_device
from ctgp_lib.train.evaluate import evaluate_loader_metrics, evaluate_solution_with_budget
try:
    import xgboost as xgb
except Exception:
    xgb = None
try:
    import lightgbm as lgb
except Exception:
    lgb = None

# ============================================================================
# Surrogate Model (from graph.py)
# ============================================================================

class PairwiseMLP(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim * 2, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, a, b):
        return self.net(torch.cat([a, b], dim=1))


class GraphSurrogateManager:
    def __init__(self, mode='mlp', feature_dim=SURROGATE_FEATURE_DIM):
        self.mode = mode
        self.feature_dim = feature_dim
        self.X = []
        self.y = []

        if mode == 'mlp':
            self.model = PairwiseMLP(feature_dim).to(get_device())
            self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        elif mode == 'xgb':
            if xgb is None:
                raise RuntimeError("xgboost not available; use 'mlp' or 'none'")
            self.model = xgb.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, eval_metric='logloss',
            )
            self.opt = None
        elif mode == 'lgbm':
            if lgb is None:
                raise RuntimeError("lightgbm not available; use 'mlp' or 'none'")
            self.model = lgb.LGBMClassifier(
                n_estimators=300, learning_rate=0.05, num_leaves=31,
                subsample=0.9, colsample_bytree=0.9,
            )
            self.opt = None
        else:
            self.model = None
            self.opt = None

    def ready(self, warmup):
        return self.model is not None and len(self.y) >= int(warmup)

    def add_pair(self, child_feat, parent_feat, label_child_better):
        x1 = np.concatenate([child_feat, parent_feat], axis=0)
        x2 = np.concatenate([parent_feat, child_feat], axis=0)
        y1 = int(label_child_better)
        y2 = 1 - y1
        self.X.append(x1)
        self.y.append(y1)
        self.X.append(x2)
        self.y.append(y2)

    def train(self, epochs_mlp=60):
        if self.model is None or len(self.y) < 20:
            return

        X = np.asarray(self.X, dtype=np.float32)
        y = np.asarray(self.y, dtype=np.int64)

        if self.mode == 'mlp':
            Xt = torch.tensor(X, dtype=torch.float32).to(get_device())
            yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(get_device())
            self.model.train()
            for _ in range(int(epochs_mlp)):
                pred = self.model(Xt[:, :self.feature_dim], Xt[:, self.feature_dim:])
                loss = F.binary_cross_entropy(pred, yt)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
        else:
            self.model.fit(X, y)

    def score_child_vs_parent(self, child_feat, parent_feat):
        if self.model is None:
            return 0.5

        if self.mode == 'mlp':
            with torch.no_grad():
                a = torch.tensor(child_feat, dtype=torch.float32).unsqueeze(0).to(get_device())
                b = torch.tensor(parent_feat, dtype=torch.float32).unsqueeze(0).to(get_device())
                return float(self.model(a, b).item())

        x = np.concatenate([child_feat, parent_feat], axis=0).reshape(1, -1).astype(np.float32)
        return float(self.model.predict_proba(x)[0, 1])


def _safe_mean(values):
    if not values:
        return 0.0
    return float(np.mean(values))


def _safe_std(values):
    if not values:
        return 0.0
    return float(np.std(values))


def extract_solution_features(sol, num_params, auc, acc, f1):
    """Extract feature vector from solution for surrogate"""
    convs = sol['conv']
    hiddens = sol['hidden']
    heads = sol['heads']
    cls_hidden = sol['cls_hidden']
    n_layers = len(convs)

    conv_counts = [convs.count(name) / max(1, n_layers) for name in CONV_NAMES]
    features = [
        float(n_layers),
        float(_safe_mean(heads)),
        float(_safe_mean(hiddens)),
        float(_safe_std(hiddens)),
        float(sol['drop']),
        float(math.log10(max(sol['lr'], 1e-12))),
        float(INIT_OPTIONS.index(sol['init'])) / max(1, len(INIT_OPTIONS) - 1),
        float(len(cls_hidden)),
        float(_safe_mean(cls_hidden)),
        float(_safe_std(cls_hidden)),
        float(num_params),
        float(auc),
        float(acc),
        float(f1),
    ]
    features.extend(float(v) for v in conv_counts[:2])
    return np.asarray(features, dtype=np.float32)


def quick_evaluate_solution(sol, tr, vl, in_f, out_f, quick_epochs=12):
    """Quick evaluation with limited epochs"""
    acc, auc, _, params, model = evaluate_solution_with_budget(
        sol, tr, vl, in_f, out_f,
        max_epochs=max(1, int(quick_epochs)),
        patience=max(2, int(quick_epochs) + 1),
    )
    val_loader = vl if vl is not None else tr
    val_metrics = evaluate_loader_metrics(model, val_loader, split='val')
    return {
        'auc': float(val_metrics['auc']),
        'acc': float(val_metrics['acc']),
        'f1': float(val_metrics['f1']),
        'params': int(params),
    }


def collect_final_metrics(model, tr, vl, ts):
    """Collect final metrics on val and test sets, plus inference time on test."""
    val_loader = vl if vl is not None else tr
    test_loader = ts if ts is not None else val_loader
    val_metrics = evaluate_loader_metrics(model, val_loader, split='val')

    # Measure inference time on test set
    model.eval()
    n_samples = 0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    infer_start = time.time()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(get_device())
            _ = model(batch)
            n_samples += batch.y.numel()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    infer_elapsed = time.time() - infer_start
    inference_ms_per_sample = (infer_elapsed / max(1, n_samples)) * 1000.0

    test_metrics = evaluate_loader_metrics(model, test_loader, split='test')
    test_metrics['inference_ms_per_sample'] = inference_ms_per_sample
    test_metrics['inference_total_sec'] = infer_elapsed
    return val_metrics, test_metrics
