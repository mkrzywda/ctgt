"""Auto-generated from the CTGP research monolith — do not edit by hand."""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from ctgp_lib.models.gnn import safe_roc_auc

def extract_graph_features(graphs, method='graph_concat_stats'):
    """Extract global features from graphs"""
    features = []
    for graph in graphs:
        x = graph.x.numpy() if hasattr(graph.x, 'numpy') else graph.x
        
        if method == 'graph_mean':
            # Mean of node features
            feat = np.mean(x, axis=0)
        elif method == 'graph_max':
            # Max of node features
            feat = np.max(x, axis=0)
        elif method == 'graph_std':
            # Std of node features
            feat = np.std(x, axis=0)
        elif method == 'graph_concat_stats':
            # Concatenate mean, max, std
            feat = np.concatenate([
                np.mean(x, axis=0),
                np.max(x, axis=0),
                np.std(x, axis=0)
            ])
        else:
            feat = np.mean(x, axis=0)
        
        features.append(feat)
    
    return np.asarray(features, dtype=np.float32)


def train_baseline_mlp(tr_graphs, vl_graphs, ts_graphs, epochs=100):
    """Train simple MLP baseline"""
    from sklearn.preprocessing import StandardScaler
    
    X_tr = extract_graph_features(tr_graphs, method='graph_concat_stats')
    X_vl = extract_graph_features(vl_graphs, method='graph_concat_stats')
    X_ts = extract_graph_features(ts_graphs, method='graph_concat_stats')
    
    y_tr = np.array([int(g.y.item()) for g in tr_graphs], dtype=int)
    y_vl = np.array([int(g.y.item()) for g in vl_graphs], dtype=int)
    y_ts = np.array([int(g.y.item()) for g in ts_graphs], dtype=int)
    
    # Normalize
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_vl = scaler.transform(X_vl)
    X_ts = scaler.transform(X_ts)
    
    # PyTorch MLP
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(get_device())
    X_vl_t = torch.tensor(X_vl, dtype=torch.float32).to(get_device())
    X_ts_t = torch.tensor(X_ts, dtype=torch.float32).to(get_device())
    y_tr_t = torch.tensor(y_tr, dtype=torch.long).to(get_device())
    y_vl_t = torch.tensor(y_vl, dtype=torch.long).to(get_device())
    y_ts_t = torch.tensor(y_ts, dtype=torch.long).to(get_device())
    
    model = torch.nn.Sequential(
        torch.nn.Linear(X_tr.shape[1], 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(64, 2)
    ).to(get_device())
    
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    best_val_auc = -float('inf')
    patience = 0
    
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(X_tr_t)
        loss = loss_fn(logits, y_tr_t)
        loss.backward()
        opt.step()
        
        model.eval()
        with torch.no_grad():
            val_logits = model(X_vl_t)
            val_probs = torch.softmax(val_logits, dim=1)
            val_preds = val_logits.argmax(dim=1)
            val_auc = safe_roc_auc(y_vl.tolist(), val_probs.cpu().numpy())
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        
        if patience >= 15:
            break
    
    model.load_state_dict(best_state)
    model.eval()
    
    with torch.no_grad():
        val_logits = model(X_vl_t)
        val_probs = torch.softmax(val_logits, dim=1)
        val_preds = val_logits.argmax(dim=1)
        
        ts_logits = model(X_ts_t)
        ts_probs = torch.softmax(ts_logits, dim=1)
        ts_preds = ts_logits.argmax(dim=1)
    
    val_metrics = {
        'auc': safe_roc_auc(y_vl.tolist(), val_probs.cpu().numpy()),
        'acc': accuracy_score(y_vl, val_preds.cpu().numpy()),
        'f1': f1_score(y_vl, val_preds.cpu().numpy(), average='macro', zero_division=0),
    }
    
    test_metrics = {
        'auc': safe_roc_auc(y_ts.tolist(), ts_probs.cpu().numpy()),
        'acc': accuracy_score(y_ts, ts_preds.cpu().numpy()),
        'f1': f1_score(y_ts, ts_preds.cpu().numpy(), average='macro', zero_division=0),
    }
    
    return val_metrics, test_metrics, 'baseline_mlp'


def train_baseline_sklearn(method_name, tr_graphs, vl_graphs, ts_graphs):
    """Train sklearn baseline (RandomForest, SVM, etc.)"""
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    
    X_tr = extract_graph_features(tr_graphs, method='graph_concat_stats')
    X_vl = extract_graph_features(vl_graphs, method='graph_concat_stats')
    X_ts = extract_graph_features(ts_graphs, method='graph_concat_stats')
    
    y_tr = np.array([int(g.y.item()) for g in tr_graphs], dtype=int)
    y_vl = np.array([int(g.y.item()) for g in vl_graphs], dtype=int)
    y_ts = np.array([int(g.y.item()) for g in ts_graphs], dtype=int)
    
    # Normalize
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_vl = scaler.transform(X_vl)
    X_ts = scaler.transform(X_ts)
    
    if method_name == 'baseline_rf':
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    elif method_name == 'baseline_dt':
        model = DecisionTreeClassifier(max_depth=10, random_state=42)
    elif method_name == 'baseline_svm':
        model = SVC(kernel='rbf', probability=True, random_state=42)
    elif method_name == 'baseline_lr':
        model = LogisticRegression(max_iter=10000, random_state=42)
    else:
        raise ValueError(f"Unknown baseline: {method_name}")
    
    model.fit(X_tr, y_tr)
    
    if hasattr(model, 'predict_proba'):
        val_probs = model.predict_proba(X_vl)
        ts_probs = model.predict_proba(X_ts)
    else:
        val_probs = model.decision_function(X_vl)
        ts_probs = model.decision_function(X_ts)
    
    val_preds = model.predict(X_vl)
    ts_preds = model.predict(X_ts)
    
    val_metrics = {
        'auc': safe_roc_auc(y_vl.tolist(), val_probs),
        'acc': accuracy_score(y_vl, val_preds),
        'f1': f1_score(y_vl, val_preds, average='macro', zero_division=0),
    }
    
    test_metrics = {
        'auc': safe_roc_auc(y_ts.tolist(), ts_probs),
        'acc': accuracy_score(y_ts, ts_preds),
        'f1': f1_score(y_ts, ts_preds, average='macro', zero_division=0),
    }
    
