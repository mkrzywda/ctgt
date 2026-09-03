"""Auto-generated from the CTGP research monolith — do not edit by hand."""

import copy
import logging
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from ctgp_lib.runtime import logger
from ctgp_lib.genotype.concat_tree import *
from ctgp_lib.train.evaluate import evaluate_solution_with_budget, collect_final_metrics
from ctgp_lib.search.surrogate import GraphSurrogateManager, extract_solution_features
from ctgp_lib.utils.io import append_csv_row, save_history_csv

def plot_search_history(history_rows, dataset, output_dir):
    if not history_rows:
        return

    gens = [row['generation'] for row in history_rows]
    best_auc = [row['best_auc'] for row in history_rows]
    current_auc = [row['current_auc'] for row in history_rows]
    rejected = [row['surrogate_rejected_total'] for row in history_rows]

    plt.figure(figsize=(11, 6))
    plt.plot(gens, best_auc, label='Best val AUC', linewidth=2.2)
    plt.plot(gens, current_auc, label='Current val AUC', linewidth=1.8, alpha=0.85)
    plt.xlabel('Generation')
    plt.ylabel('AUC-ROC')
    plt.title(f'JARVIS Metal Classification | CTGP+Surrogate | {dataset}')
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'auc_history_{dataset}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(11, 6))
    plt.step(gens, rejected, where='post', label='Rejected by surrogate', linewidth=2.2)
    plt.xlabel('Generation')
    plt.ylabel('Cumulative rejected models')
    plt.title(f'JARVIS Surrogate Rejections | {dataset}')
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'rejections_{dataset}.png'), dpi=300, bbox_inches='tight')
    plt.close()


def run_ctgp_1plus1(task, dataset, loaders, args, use_surrogate=False):
    """CTGP 1+1 evolution with optional surrogate for JARVIS metal classification"""
    tr, vl, ts, in_f, out_f = loaders
    start_time = time.time()

    # Initialize
    current_prog = _random_gp_program()
    current_sol = _gp_to_solution(current_prog)
    current_acc, current_auc, _, current_params, current_model = evaluate_solution_with_budget(
        current_sol, tr, vl, in_f, out_f, max_epochs=50, patience=5
    )
    current_val_metrics, current_test_metrics = collect_final_metrics(current_model, tr, vl, ts)

    best_prog = copy.deepcopy(current_prog)
    best_sol = copy.deepcopy(current_sol)
    best_auc = float(current_auc)
    best_acc = float(current_acc)
    best_params = int(current_params)
    best_model = current_model
    best_val_metrics = current_val_metrics
    best_test_metrics = current_test_metrics

    surrogate = GraphSurrogateManager('mlp' if use_surrogate else 'none')
    rejected_total = 0
    evaluated_total = 1
    history_rows = []

    current_feat = extract_solution_features(
        current_sol, current_params,
        current_auc, current_val_metrics['acc'], current_val_metrics['f1']
    )

    logger.info(f"Initial: AUC={best_auc:.4f}, ACC={best_acc:.4f}, Params={best_params}")

    best_history = [{
        'generation': 0,
        'val_auc': best_auc,
        'val_acc': best_acc,
        'val_f1': best_val_metrics['f1'],
        'test_auc': best_test_metrics['auc'],
        'test_acc': best_test_metrics['acc'],
        'test_f1': best_test_metrics['f1'],
        'params': best_params,
        'sol': copy.deepcopy(best_sol),
    }]

    for gen in range(1, args.generations + 1):
        # Mutate
        child_prog = _gp_mutate(current_prog)
        child_sol = _gp_to_solution(child_prog)
        child_feat = extract_solution_features(
            child_sol, 0, 0.5, 0.5, 0.5  # placeholder
        )

        # Surrogate check
        accept_via_surrogate = False
        if surrogate.ready(args.surrogate_warmup):
            surrogate_score = surrogate.score_child_vs_parent(child_feat, current_feat)
            if surrogate_score < args.surrogate_acceptance_threshold:
                rejected_total += 1
                history_rows.append({
                    'generation': gen,
                    'best_auc': best_auc,
                    'current_auc': current_auc,
                    'surrogate_rejected_total': rejected_total,
                    'surrogate_score': surrogate_score,
                    'was_rejected': 1,
                })
                continue
            accept_via_surrogate = True

        # Evaluate child
        try:
            child_acc, child_auc, _, child_params, child_model = evaluate_solution_with_budget(
                child_sol, tr, vl, in_f, out_f, max_epochs=50, patience=5
            )
            evaluated_total += 1
        except Exception as e:
            logger.warning(f"Gen {gen}: evaluation failed - {e}")
            continue

        child_val_metrics, child_test_metrics = collect_final_metrics(child_model, tr, vl, ts)
        child_feat = extract_solution_features(
            child_sol, child_params, child_auc, child_val_metrics['acc'], child_val_metrics['f1']
        )

        # Accept/reject
        if child_auc > current_auc:
            current_prog = copy.deepcopy(child_prog)
            current_sol = copy.deepcopy(child_sol)
            current_auc = float(child_auc)
            current_acc = float(child_acc)
            current_params = int(child_params)
            current_feat = child_feat
            current_model = child_model
            current_val_metrics = child_val_metrics
            accept_label = 1
        else:
            accept_label = 0

        # Update best
        if child_auc > best_auc:
            best_prog = copy.deepcopy(child_prog)
            best_sol = copy.deepcopy(child_sol)
            best_auc = float(child_auc)
            best_acc = float(child_val_metrics['acc'])
            best_params = int(child_params)
            best_model = child_model
            best_val_metrics = child_val_metrics
            best_test_metrics = child_test_metrics
            logger.info(f"Gen {gen}: New best AUC={best_auc:.4f}, ACC={best_acc:.4f}")
            best_history.append({
                'generation': gen,
                'val_auc': best_auc,
                'val_acc': best_acc,
                'val_f1': best_val_metrics['f1'],
                'test_auc': best_test_metrics['auc'],
                'test_acc': best_test_metrics['acc'],
                'test_f1': best_test_metrics['f1'],
                'params': best_params,
                'sol': copy.deepcopy(best_sol),
            })

        # Train surrogate
        if use_surrogate and gen % args.surrogate_train_every == 0:
            surrogate.add_pair(child_feat, current_feat, accept_label)
            surrogate.train(epochs_mlp=args.surrogate_train_epochs)

        history_rows.append({
            'generation': gen,
            'best_auc': best_auc,
            'current_auc': current_auc,
            'surrogate_rejected_total': rejected_total,
            'surrogate_score': surrogate.score_child_vs_parent(child_feat, current_feat) if surrogate.model is not None else 0.5,
            'was_rejected': 0,
        })

    elapsed = time.time() - start_time
    logger.info(
        f"Search complete: Best AUC={best_auc:.4f}, ACC={best_acc:.4f}, "
        f"Params={best_params}, Rejected={rejected_total}/{evaluated_total}, Time={elapsed:.1f}s"
    )

    return {
        'best_solution': best_sol,
        'best_model': best_model,
        'best_acc': best_acc,
        'best_auc': best_auc,
        'best_params': best_params,
        'val_metrics': best_val_metrics,
        'test_metrics': best_test_metrics,
        'history': history_rows,
        'best_history': best_history,
        'surrogate_rejected_total': rejected_total,
        'surrogate_full_evaluations': evaluated_total,
        'elapsed_time_sec': elapsed,
    }


def run_ctgp_1plus4(task, dataset, loaders, args, use_surrogate=False):
    """CTGP 1+4 evolution: 1 parent, 4 offspring per generation"""
    tr, vl, ts, in_f, out_f = loaders
    start_time = time.time()

    # Initialize
    current_prog = _random_gp_program()
    current_sol = _gp_to_solution(current_prog)
    current_acc, current_auc, _, current_params, current_model = evaluate_solution_with_budget(
        current_sol, tr, vl, in_f, out_f, max_epochs=50, patience=5
    )
    current_val_metrics, current_test_metrics = collect_final_metrics(current_model, tr, vl, ts)

    best_prog = copy.deepcopy(current_prog)
    best_sol = copy.deepcopy(current_sol)
    best_auc = float(current_auc)
    best_acc = float(current_acc)
    best_params = int(current_params)
    best_model = current_model
    best_val_metrics = current_val_metrics
    best_test_metrics = current_test_metrics

    surrogate = GraphSurrogateManager('mlp' if use_surrogate else 'none')
    rejected_total = 0
    evaluated_total = 1
    history_rows = []

    current_feat = extract_solution_features(
        current_sol, current_params,
        current_auc, current_val_metrics['acc'], current_val_metrics['f1']
    )

    logger.info(f"Initial: AUC={best_auc:.4f}, ACC={best_acc:.4f}, Params={best_params}")

    best_history = [{
        'generation': 0,
        'val_auc': best_auc,
        'val_acc': best_acc,
        'val_f1': best_val_metrics['f1'],
        'test_auc': best_test_metrics['auc'],
        'test_acc': best_test_metrics['acc'],
        'test_f1': best_test_metrics['f1'],
        'params': best_params,
        'sol': copy.deepcopy(best_sol),
    }]

    for gen in range(1, args.generations + 1):
        # Generate 4 offspring
        offspring = []
        for _ in range(4):
            child_prog = _gp_mutate(current_prog)
            child_sol = _gp_to_solution(child_prog)
            child_feat = extract_solution_features(
                child_sol, 0, 0.5, 0.5, 0.5  # placeholder
            )

            # Surrogate check
            if surrogate.ready(args.surrogate_warmup):
                surrogate_score = surrogate.score_child_vs_parent(child_feat, current_feat)
                if surrogate_score < args.surrogate_acceptance_threshold:
                    rejected_total += 1
                    continue
            
            offspring.append({
                'prog': child_prog,
                'sol': child_sol,
                'feat': child_feat,
                'surrogate_score': surrogate_score if surrogate.ready(args.surrogate_warmup) else 0.5
            })

        # Evaluate best offspring
        best_offspring = None
        best_offspring_auc = -float('inf')
        for off in offspring:
            try:
                off_acc, off_auc, _, off_params, off_model = evaluate_solution_with_budget(
                    off['sol'], tr, vl, in_f, out_f, max_epochs=50, patience=5
                )
                evaluated_total += 1
                off_val_metrics, off_test_metrics = collect_final_metrics(off_model, tr, vl, ts)
                off['acc'] = off_acc
                off['auc'] = off_auc
                off['params'] = off_params
                off['model'] = off_model
                off['val_metrics'] = off_val_metrics
                off['test_metrics'] = off_test_metrics
                off['feat'] = extract_solution_features(
                    off['sol'], off_params, off_auc, off_val_metrics['acc'], off_val_metrics['f1']
                )
                
                if off_auc > best_offspring_auc:
                    best_offspring_auc = off_auc
                    best_offspring = off
            except Exception as e:
                logger.warning(f"Gen {gen}: offspring evaluation failed - {e}")

        if best_offspring is None:
            # No valid offspring, continue with current
            history_rows.append({
                'generation': gen,
                'best_auc': best_auc,
                'current_auc': current_auc,
                'surrogate_rejected_total': rejected_total,
                'surrogate_score': 0.0,
                'was_rejected': 1,
                'offspring_count': len(offspring),
            })
            continue

        accept_label = 0
        if best_offspring['auc'] > current_auc:
            current_prog = copy.deepcopy(best_offspring['prog'])
            current_sol = copy.deepcopy(best_offspring['sol'])
            current_auc = float(best_offspring['auc'])
            current_acc = float(best_offspring['acc'])
            current_params = int(best_offspring['params'])
            current_feat = best_offspring['feat']
            current_model = best_offspring['model']
            current_val_metrics = best_offspring['val_metrics']
            accept_label = 1

        # Update best
        if best_offspring['auc'] > best_auc:
            best_prog = copy.deepcopy(best_offspring['prog'])
            best_sol = copy.deepcopy(best_offspring['sol'])
            best_auc = float(best_offspring['auc'])
            best_acc = float(best_offspring['val_metrics']['acc'])
            best_params = int(best_offspring['params'])
            best_model = best_offspring['model']
            best_val_metrics = best_offspring['val_metrics']
            best_test_metrics = best_offspring['test_metrics']
            logger.info(f"Gen {gen}: New best AUC={best_auc:.4f}, ACC={best_acc:.4f}")
            best_history.append({
                'generation': gen,
                'val_auc': best_auc,
                'val_acc': best_acc,
                'val_f1': best_val_metrics['f1'],
                'test_auc': best_test_metrics['auc'],
                'test_acc': best_test_metrics['acc'],
                'test_f1': best_test_metrics['f1'],
                'params': best_params,
                'sol': copy.deepcopy(best_sol),
            })

        # Train surrogate
        if use_surrogate and gen % args.surrogate_train_every == 0:
            surrogate.add_pair(best_offspring['feat'], current_feat, accept_label)
            surrogate.train(epochs_mlp=args.surrogate_train_epochs)

        history_rows.append({
            'generation': gen,
            'best_auc': best_auc,
            'current_auc': current_auc,
            'surrogate_rejected_total': rejected_total,
            'surrogate_score': best_offspring['surrogate_score'],
            'was_rejected': 0,
            'offspring_count': len(offspring),
        })

    elapsed = time.time() - start_time
    logger.info(
        f"Search complete: Best AUC={best_auc:.4f}, ACC={best_acc:.4f}, "
        f"Params={best_params}, Rejected={rejected_total}/{evaluated_total}, Time={elapsed:.1f}s"
    )

    return {
        'best_solution': best_sol,
        'best_model': best_model,
        'best_acc': best_acc,
        'best_auc': best_auc,
        'best_params': best_params,
        'val_metrics': best_val_metrics,
        'test_metrics': best_test_metrics,
        'history': history_rows,
        'best_history': best_history,
        'surrogate_rejected_total': rejected_total,
        'surrogate_full_evaluations': evaluated_total,
        'elapsed_time_sec': elapsed,
    }


def run_ctgp_1plus10(task, dataset, loaders, args, use_surrogate=False):
    """CTGP 1+10 evolution: 1 parent, 10 offspring per generation
    
    More exploration than 1+4, encourages discovery of diverse heterogeneous architectures.
    Each generation evaluates 10 candidates, providing richer selection for layer type mixing.
    """
    tr, vl, ts, in_f, out_f = loaders
    start_time = time.time()

    # Initialize
    current_prog = _random_gp_program()
    current_sol = _gp_to_solution(current_prog)
    current_acc, current_auc, _, current_params, current_model = evaluate_solution_with_budget(
        current_sol, tr, vl, in_f, out_f, max_epochs=50, patience=5
    )
    current_val_metrics, current_test_metrics = collect_final_metrics(current_model, tr, vl, ts)

    best_prog = copy.deepcopy(current_prog)
    best_sol = copy.deepcopy(current_sol)
    best_auc = float(current_auc)
    best_acc = float(current_acc)
    best_params = int(current_params)
    best_model = current_model
    best_val_metrics = current_val_metrics
    best_test_metrics = current_test_metrics

    surrogate = GraphSurrogateManager('mlp' if use_surrogate else 'none')
    rejected_total = 0
    evaluated_total = 1
    history_rows = []

    current_feat = extract_solution_features(
        current_sol, current_params,
        current_auc, current_val_metrics['acc'], current_val_metrics['f1']
    )

    logger.info(f"Initial: AUC={best_auc:.4f}, ACC={best_acc:.4f}, Params={best_params}")

    best_history = [{
        'generation': 0,
        'val_auc': best_auc,
        'val_acc': best_acc,
        'val_f1': best_val_metrics['f1'],
        'test_auc': best_test_metrics['auc'],
        'test_acc': best_test_metrics['acc'],
        'test_f1': best_test_metrics['f1'],
        'params': best_params,
        'sol': copy.deepcopy(best_sol),
    }]

    for gen in range(1, args.generations + 1):
        # Generate 10 offspring
        offspring = []
        for _ in range(10):
            child_prog = _gp_mutate(current_prog)
            child_sol = _gp_to_solution(child_prog)
            child_feat = extract_solution_features(
                child_sol, 0, 0.5, 0.5, 0.5  # placeholder
            )

            # Surrogate check
            if surrogate.ready(args.surrogate_warmup):
                surrogate_score = surrogate.score_child_vs_parent(child_feat, current_feat)
                if surrogate_score < args.surrogate_acceptance_threshold:
                    rejected_total += 1
                    continue
            
            offspring.append({
                'prog': child_prog,
                'sol': child_sol,
                'feat': child_feat,
                'surrogate_score': surrogate_score if surrogate.ready(args.surrogate_warmup) else 0.5
            })

        # Evaluate best offspring
        best_offspring = None
        best_offspring_auc = -float('inf')
        for off in offspring:
            try:
                off_acc, off_auc, _, off_params, off_model = evaluate_solution_with_budget(
                    off['sol'], tr, vl, in_f, out_f, max_epochs=50, patience=5
                )
                evaluated_total += 1
                off_val_metrics, off_test_metrics = collect_final_metrics(off_model, tr, vl, ts)
                off['acc'] = off_acc
                off['auc'] = off_auc
                off['params'] = off_params
                off['model'] = off_model
                off['val_metrics'] = off_val_metrics
                off['test_metrics'] = off_test_metrics
                off['feat'] = extract_solution_features(
                    off['sol'], off_params, off_auc, off_val_metrics['acc'], off_val_metrics['f1']
                )
                
                if off_auc > best_offspring_auc:
                    best_offspring_auc = off_auc
                    best_offspring = off
            except Exception as e:
                logger.warning(f"Gen {gen}: offspring evaluation failed - {e}")

        if best_offspring is None:
            # No valid offspring, continue with current
            history_rows.append({
                'generation': gen,
                'best_auc': best_auc,
                'current_auc': current_auc,
                'surrogate_rejected_total': rejected_total,
                'surrogate_score': 0.0,
                'was_rejected': 1,
                'offspring_count': len(offspring),
            })
            continue

        accept_label = 0
        if best_offspring['auc'] > current_auc:
            current_prog = copy.deepcopy(best_offspring['prog'])
            current_sol = copy.deepcopy(best_offspring['sol'])
            current_auc = float(best_offspring['auc'])
            current_acc = float(best_offspring['acc'])
            current_params = int(best_offspring['params'])
            current_feat = best_offspring['feat']
            current_model = best_offspring['model']
            current_val_metrics = best_offspring['val_metrics']
            accept_label = 1

        # Update best
        if best_offspring['auc'] > best_auc:
            best_prog = copy.deepcopy(best_offspring['prog'])
            best_sol = copy.deepcopy(best_offspring['sol'])
            best_auc = float(best_offspring['auc'])
            best_acc = float(best_offspring['val_metrics']['acc'])
            best_params = int(best_offspring['params'])
            best_model = best_offspring['model']
            best_val_metrics = best_offspring['val_metrics']
            best_test_metrics = best_offspring['test_metrics']
            logger.info(f"Gen {gen}: New best AUC={best_auc:.4f}, ACC={best_acc:.4f}")
            best_history.append({
                'generation': gen,
                'val_auc': best_auc,
                'val_acc': best_acc,
                'val_f1': best_val_metrics['f1'],
                'test_auc': best_test_metrics['auc'],
                'test_acc': best_test_metrics['acc'],
                'test_f1': best_test_metrics['f1'],
                'params': best_params,
                'sol': copy.deepcopy(best_sol),
            })

        # Train surrogate
        if use_surrogate and gen % args.surrogate_train_every == 0:
            surrogate.add_pair(best_offspring['feat'], current_feat, accept_label)
            surrogate.train(epochs_mlp=args.surrogate_train_epochs)

        history_rows.append({
            'generation': gen,
            'best_auc': best_auc,
            'current_auc': current_auc,
            'surrogate_rejected_total': rejected_total,
            'surrogate_score': best_offspring['surrogate_score'],
            'was_rejected': 0,
            'offspring_count': len(offspring),
        })

    elapsed = time.time() - start_time
    logger.info(
        f"Search complete: Best AUC={best_auc:.4f}, ACC={best_acc:.4f}, "
        f"Params={best_params}, Rejected={rejected_total}/{evaluated_total}, Time={elapsed:.1f}s"
    )

    return {
        'best_solution': best_sol,
        'best_model': best_model,
        'best_acc': best_acc,
        'best_auc': best_auc,
        'best_params': best_params,
        'val_metrics': best_val_metrics,
        'test_metrics': best_test_metrics,
        'history': history_rows,
        'best_history': best_history,
        'surrogate_rejected_total': rejected_total,
        'surrogate_full_evaluations': evaluated_total,
        'elapsed_time_sec': elapsed,
    }
