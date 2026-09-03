"""Auto-generated from the CTGP research monolith — do not edit by hand."""

import copy
import time
import numpy as np
import torch
import torch.nn.functional as F
from ctgp_lib.runtime import process
from ctgp_lib.device import get_device
from ctgp_lib.models.gnn import GNN, evaluate_loader_metrics

# ============================================================================
# Solution Evaluation
# ============================================================================

def evaluate_solution_with_budget(sol, tr, vl, in_f, out_f, max_epochs=100, patience=10):
    """Evaluate solution with early stopping"""
    model = GNN(in_f, sol, out_f, task='graph').to(get_device())
    opt = torch.optim.Adam(model.parameters(), lr=sol['lr'])
    loss_fn = torch.nn.CrossEntropyLoss()
    mem0 = process.memory_info().rss if process is not None else 0

    val_loader = vl if vl is not None else tr
    best_auc = -float('inf')
    best_acc = 0.0
    patience_counter = 0
    best_state = None
    train_start = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in tr:
            batch = batch.to(get_device())
            opt.zero_grad()
            out = model(batch)
            logits = out
            true = batch.y.view(-1).long()
            loss = loss_fn(logits, true)
            loss.backward()
            opt.step()

        val_metrics = evaluate_loader_metrics(model, val_loader, split='val', loss_fn=loss_fn)
        val_auc = val_metrics['auc']

        if val_auc > best_auc:
            best_auc = val_auc
            best_acc = val_metrics['acc']
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    train_time_sec = time.time() - train_start

    if best_state is not None:
        model.load_state_dict(best_state)

    mem1 = process.memory_info().rss if process is not None else mem0
    model.train_time_sec = train_time_sec
    return best_acc, best_auc, mem1 - mem0, model.param_count, model

