"""Auto-generated from the CTGP research monolith — do not edit by hand."""

import copy
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from ctgp_lib.runtime import process
from ctgp_lib.device import get_device

# ============================================================================
# GNN Model (from jarvis.py + graph.py)
# ============================================================================

from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, SGConv, global_mean_pool

CONV_MAP = {
    'GCN': GCNConv,
    'GraphSAGE': SAGEConv,
    'GIN': lambda in_c, out_c: GINConv(torch.nn.Sequential(
        torch.nn.Linear(in_c, out_c), torch.nn.ReLU(), torch.nn.Linear(out_c, out_c)
    )),
    'SG': SGConv
}


def apply_init(module, init_type):
    """Initialize weights"""
    for m in module.modules():
        if hasattr(m, 'weight'):
            w = m.weight
            if init_type == 'uniform':
                torch.nn.init.uniform_(w)
            elif init_type == 'xavier':
                torch.nn.init.xavier_uniform_(w)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_uniform_(w)
            elif init_type == 'normal':
                torch.nn.init.normal_(w)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class GNN(torch.nn.Module):
    def __init__(self, in_feats, sol, out_feats, task='graph'):
        super().__init__()
        self.sol = sol
        self.task = task
        layers, last = [], in_feats
        for typ, heads, hid in zip(sol['conv'], sol['heads'], sol['hidden']):
            if typ == 'GAT':
                layers.append(GATConv(last, hid, heads=heads, concat=False))
            else:
                layers.append(CONV_MAP[typ](last, hid))
            last = hid
        self.layers = torch.nn.ModuleList(layers)
        cls_seq = []
        for hid in sol['cls_hidden']:
            cls_seq += [torch.nn.Linear(last, hid), torch.nn.ReLU(), torch.nn.Dropout(sol['drop'])]
            last = hid
        cls_seq.append(torch.nn.Linear(last, out_feats))
        self.classifier = torch.nn.Sequential(*cls_seq)
        apply_init(self, sol['init'])
        self.param_count = sum(p.numel() for p in self.parameters())

    def forward(self, data):
        x, edge = data.x, data.edge_index
        for conv in self.layers:
            x = F.relu(conv(x, edge))
            x = F.dropout(x, self.sol['drop'], training=self.training)
        if self.task == 'node':
            return self.classifier(x)
        return self.classifier(global_mean_pool(x, data.batch))


def safe_roc_auc(y_true, y_prob):
    """Safe ROC-AUC with fallback"""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.size == 0:
        return 0.5
    n_classes = y_prob.shape[1] if y_prob.ndim == 2 else 1
    if n_classes < 2:
        return 0.5
    try:
        if n_classes == 2:
            return float(roc_auc_score(y_true, y_prob[:, 1]))
        return float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'))
    except ValueError:
        return 0.5


def evaluate_loader_metrics(model, loader, split='val', loss_fn=None):
    """Evaluate model on loader"""
    if loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss()

    model.eval()
    total_loss = 0.0
    y_t, y_p, y_prob = [], [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(get_device())
            out = model(batch)
            logits = out
            true = batch.y.view(-1)

            if true.numel() == 0:
                continue

            true_cls = true.long()
            total_loss += loss_fn(logits, true_cls).item() * true_cls.size(0)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            y_t += true_cls.cpu().tolist()
            y_p += preds.cpu().tolist()
            y_prob += probs.cpu().tolist()

    n = len(y_t)
    if n == 0:
        return {'loss': float('inf'), 'acc': 0.0, 'f1': 0.0, 'auc': 0.5}

    avg_loss = total_loss / n
    acc = accuracy_score(y_t, y_p)
    f1 = f1_score(y_t, y_p, average='macro', zero_division=0)
    auc = safe_roc_auc(y_t, y_prob)
    return {'loss': avg_loss, 'acc': acc, 'f1': f1, 'auc': auc}


def collect_predictions_with_jids(model, loader):
    """Run inference and return list of (jid, predicted_class) tuples.

    Used to produce leaderboard submission CSV files.
    Graphs must have a `.jid` attribute (set by :func:`get_jarvis_loaders`).
    """
    model.eval()
    results = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(get_device())
            preds = model(batch).argmax(dim=1).cpu().tolist()
            jids = getattr(batch, 'jid', None)
            if jids is None:
                jids = ['NA'] * len(preds)
            elif isinstance(jids, torch.Tensor):
                jids = jids.cpu().tolist()
            for jid, pred in zip(jids, preds):
                results.append((str(jid), int(pred)))
    return results


def save_leaderboard_csv(predictions, csv_path):
    """Save leaderboard-format CSV: id,prediction.

    Args:
        predictions: list of (jid, class_int) tuples from
                     :func:`collect_predictions_with_jids`
        csv_path: output file path (will be created/overwritten)
    """
    import csv as _csv
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = _csv.writer(f)
        writer.writerow(['id', 'prediction'])
        writer.writerows(predictions)
    logger.info(f"Leaderboard submission CSV saved: {csv_path} ({len(predictions)} rows)")

