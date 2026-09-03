"""Auto-generated from the CTGP research monolith — do not edit by hand."""

import json
import os
import random
import numpy as np
import torch

def set_run_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)


def build_run_seed(base_seed, dataset, method, run):
    token = f"{dataset}:{method}:{run}:{base_seed}"
    return int(base_seed + sum(ord(ch) for ch in token) % 100000)


def save_best_artifacts(run_dir, result, dataset, method, run):
    model_path = os.path.join(run_dir, 'best_model.pth')
    meta_path = os.path.join(run_dir, 'best_model_info.json')
    sol_path = os.path.join(run_dir, 'best_solution.json')
    torch.save(result['best_model'].state_dict(), model_path)
    best_model = result['best_model']
    train_time_sec = getattr(best_model, 'train_time_sec', None)
    meta = {
        'dataset': dataset,
        'method': method,
        'run': run,
        'best_auc': result['best_auc'],
        'best_acc': result['best_acc'],
        'best_params': result['best_params'],
        'train_time_sec': train_time_sec,
        'search_time_sec': result['elapsed_time_sec'],
        'search_full_evaluations': result.get('surrogate_full_evaluations'),
        'val_metrics': result['val_metrics'],
        'test_metrics': result['test_metrics'],
        'elapsed_sec': result['elapsed_time_sec'],
    }
    with open(meta_path, 'w') as fp:
        json.dump(meta, fp, indent=2)
    with open(sol_path, 'w') as fp:
        json.dump(result['best_solution'], fp, indent=2)
    if result.get('best_history'):
        hist_path = os.path.join(run_dir, 'best_architectures_history.json')
        with open(hist_path, 'w') as fp:
            json.dump(result['best_history'], fp, indent=2)
