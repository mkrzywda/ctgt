"""Main experiment loop and CLI argument definitions."""

import argparse
import copy
import json
import logging
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ctgp_lib.baselines.sklearn_ml import train_baseline_mlp, train_baseline_sklearn
from ctgp_lib.config import LEADERBOARD_BENCHMARK_MAP
from ctgp_lib.data.jarvis import download_leaderboard_splits, get_jarvis_loaders
from ctgp_lib.device import get_device
from ctgp_lib.models.gnn import GNN, collect_predictions_with_jids, save_leaderboard_csv
from ctgp_lib.runtime import SAVE_DIR, logger
from ctgp_lib.search.evolution import (
    plot_search_history,
    run_ctgp_1plus1,
    run_ctgp_1plus4,
    run_ctgp_1plus10,
)
from ctgp_lib.utils.io import append_csv_row, save_history_csv
from ctgp_lib.utils.seed import build_run_seed, save_best_artifacts, set_run_seed

def run_experiments(args):
    """Main experiment loop with graph configurations and baseline methods"""
    import pandas as pd

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-7s | %(message)s'
    )

    global SAVE_DIR, logger
    logger = logging.getLogger(__name__)

    # Embed repr_level in output dir so results stay separate per level
    repr_level = getattr(args, 'repr_level', 0)
    if repr_level != 0 and not args.output_dir.endswith(f'_L{repr_level}'):
        args.output_dir = args.output_dir.rstrip('/') + f'_L{repr_level}'
    logger.info(f"Output directory: {args.output_dir}  (repr_level={repr_level})")

    os.makedirs(args.output_dir, exist_ok=True)
    detailed_csv = os.path.join(args.output_dir, 'detailed_results.csv')
    history_root = os.path.join(args.output_dir, 'run_artifacts')
    all_rows = []

    # Determine which methods to run
    run_all = args.run_all or (not args.run_baselines and not args.run_ctgp)
    
    ctgp_methods = []
    baseline_methods = []
    
    if args.run_ctgp or run_all:
        ctgp_methods = ['ctgp_1plus1', 'ctgp_surrogate_1plus1', 'ctgp_1plus4', 'ctgp_surrogate_1plus4', 'ctgp_1plus10', 'ctgp_surrogate_1plus10']
        if args.methods:
            ctgp_methods = [m for m in ctgp_methods if m in args.methods]
            if not ctgp_methods:
                raise ValueError(f"No CTGP methods matched --methods {args.methods}")
    
    if args.run_baselines or run_all:
        baseline_methods = ['baseline_mlp', 'baseline_rf', 'baseline_dt', 'baseline_svm']
    
    # Baselines run first (fast reference), then CTGP
    all_methods = baseline_methods + ctgp_methods
    task = 'graph'

    # Graph construction configurations
    graph_cutoffs = args.graph_cutoff_configs
    graph_neighbors = args.graph_maxneighbors_configs
    
    # Apply comprehensive test configs if requested
    if args.test_configs:
        graph_cutoffs = [6.0, 8.0, 10.0]
        graph_neighbors = [8, 12, 16]
        logger.info("Using comprehensive test configurations: cutoffs=[6.0, 8.0, 10.0], neighbors=[8, 12, 16]")
    
    graph_configs = [
        (cutoff, max_neighbors)
        for cutoff in graph_cutoffs
        for max_neighbors in graph_neighbors
    ]

    logger.info(
        f"JARVIS Experiment | methods={len(all_methods)} | graph_configs={len(graph_configs)} | "
        f"runs={args.runs} | generations={args.generations}"
    )
    logger.info(f"  Methods: {all_methods}")
    logger.info(f"  Graph configs: {graph_configs}")

    # ------------------------------------------------------------------
    # LEADERBOARD MODE: download pre-determined splits
    # ------------------------------------------------------------------
    leaderboard_splits = None
    leaderboard_bench_name = None
    if getattr(args, 'leaderboard', False):
        if args.class_target not in LEADERBOARD_BENCHMARK_MAP:
            raise ValueError(
                f"Target '{args.class_target}' is not available in the JARVIS Leaderboard. "
                f"Supported targets: {list(LEADERBOARD_BENCHMARK_MAP.keys())}"
            )
        leaderboard_bench_name = LEADERBOARD_BENCHMARK_MAP[args.class_target]
        cache_dir = getattr(args, 'leaderboard_cache_dir', '.jarvis_lb_cache')
        logger.info(
            f"LEADERBOARD MODE enabled: benchmark='{leaderboard_bench_name}', "
            f"cache_dir='{cache_dir}'"
        )
        leaderboard_splits = download_leaderboard_splits(leaderboard_bench_name, cache_dir)
        # Full dataset is required — override any sampling limit
        args.max_jarvis_samples = 0

    # Process each graph configuration
    for config_idx, (cutoff, max_neighbors) in enumerate(graph_configs):
        logger.info(f"\n{'='*70}")
        logger.info(f"Graph Configuration [{config_idx+1}/{len(graph_configs)}]: cutoff={cutoff}Å, max_neighbors={max_neighbors}")
        logger.info(f"{'='*70}")
        
        # Load JARVIS data for this configuration
        logger.info(f"Loading JARVIS DFT3D data (classification target: {args.class_target})...")
        loaders = get_jarvis_loaders(
            batch_size=args.batch_size,
            jarvis_dataset='dft_3d',
            class_target=args.class_target,
            class_threshold=0.01,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            max_samples=args.max_jarvis_samples,
            leaderboard_splits=leaderboard_splits,
            repr_level=args.repr_level,
        )
        tr, vl, ts, in_f, out_f = loaders

        # Run CTGP methods with this graph configuration
        for method in ctgp_methods:
            use_surrogate = 'surrogate' in method
            
            # Extract offspring count from method name (1+1, 1+4, 1+10)
            if '1plus10' in method:
                offspring_count = 10
                ctgp_runner = run_ctgp_1plus10
            elif '1plus4' in method:
                offspring_count = 4
                ctgp_runner = run_ctgp_1plus4
            elif '1plus1' in method:
                offspring_count = 1
                ctgp_runner = run_ctgp_1plus1
            else:
                offspring_count = 1  # default fallback
                ctgp_runner = run_ctgp_1plus1
            
            run_start = args.start_run
            run_end = args.start_run + args.runs
            for run in range(run_start, run_end):
                run_seed = build_run_seed(args.seed, f'dft_3d_c{cutoff}_m{max_neighbors}', method, run)
                set_run_seed(run_seed)

                run_dir = os.path.join(history_root, f'dft_3d_c{cutoff}_m{max_neighbors}', method, f'run_{run}')
                os.makedirs(run_dir, exist_ok=True)
                SAVE_DIR = run_dir

                logger.info(
                    f"Run {run} ({run - run_start + 1}/{args.runs}) | method={method} | seed={run_seed}"
                )

                try:
                    result = ctgp_runner(task, 'dft_3d', loaders, args, use_surrogate=use_surrogate)
                    
                    save_history_csv(os.path.join(run_dir, 'history.csv'), result['history'])
                    plot_search_history(result['history'], 'dft_3d', run_dir)
                    save_best_artifacts(run_dir, result, 'dft_3d', method, run)

                    row = {
                        'task': task,
                        'dataset': 'jarvis_dft_3d',
                        'method': method,
                        'strategy': f'1+{offspring_count}',
                        'uses_surrogate': int(use_surrogate),
                        'run': run,
                        'seed': run_seed,
                        'generations': args.generations,
                        'best_val_auc': result['val_metrics']['auc'],
                        'best_val_acc': result['val_metrics']['acc'],
                        'best_val_f1': result['val_metrics']['f1'],
                        'best_test_auc': result['test_metrics']['auc'],
                        'best_test_acc': result['test_metrics']['acc'],
                        'best_test_f1': result['test_metrics']['f1'],
                        'best_params': result['best_params'],
                        'elapsed_time_sec': result['elapsed_time_sec'],
                        'surrogate_rejected_total': result['surrogate_rejected_total'],
                        'surrogate_full_evaluations': result['surrogate_full_evaluations'],
                        'artifact_dir': run_dir,
                        'graph_cutoff': cutoff,
                        'graph_max_neighbors': max_neighbors,
                    }
                    append_csv_row(detailed_csv, row)
                    all_rows.append(row)

                    logger.info(
                        f"Run complete | method={method} | val_auc={row['best_val_auc']:.4f} | "
                        f"test_auc={row['best_test_auc']:.4f} | rejected={row['surrogate_rejected_total']}"
                    )
                except Exception as e:
                    logger.error(f"Run failed: {e}", exc_info=True)
        
        # Run baseline methods with this graph configuration
        if baseline_methods:
            logger.info(f"\nRunning Baseline Methods for this graph config...")
            
            for method in baseline_methods:
                for run in range(1, args.runs + 1):
                    run_seed = build_run_seed(args.seed, f'dft_3d_c{cutoff}_m{max_neighbors}', method, run)
                    set_run_seed(run_seed)
                    
                    run_dir = os.path.join(history_root, f'dft_3d_c{cutoff}_m{max_neighbors}', method, f'run_{run}')
                    os.makedirs(run_dir, exist_ok=True)
                    
                    logger.info(f"Baseline: {method} | run {run}/{args.runs}")
                    
                    try:
                        if method == 'baseline_mlp':
                            val_metrics, test_metrics, method_id = train_baseline_mlp(
                                [tr.dataset[i] for i in range(len(tr.dataset))],
                                [vl.dataset[i] for i in range(len(vl.dataset))],
                                [ts.dataset[i] for i in range(len(ts.dataset))],
                                epochs=100
                            )
                        else:
                            val_metrics, test_metrics, method_id = train_baseline_sklearn(
                                method,
                                [tr.dataset[i] for i in range(len(tr.dataset))],
                                [vl.dataset[i] for i in range(len(vl.dataset))],
                                [ts.dataset[i] for i in range(len(ts.dataset))]
                            )
                        
                        row = {
                            'task': task,
                            'dataset': 'jarvis_dft_3d',
                            'method': method,
                            'strategy': 'baseline',
                            'uses_surrogate': 0,
                            'run': run,
                            'seed': run_seed,
                            'generations': 0,
                            'best_val_auc': val_metrics['auc'],
                            'best_val_acc': val_metrics['acc'],
                            'best_val_f1': val_metrics['f1'],
                            'best_test_auc': test_metrics['auc'],
                            'best_test_acc': test_metrics['acc'],
                            'best_test_f1': test_metrics['f1'],
                            'best_params': 0,
                            'elapsed_time_sec': 0,
                            'surrogate_rejected_total': 0,
                            'surrogate_full_evaluations': 1,
                            'artifact_dir': run_dir,
                            'graph_cutoff': cutoff,
                            'graph_max_neighbors': max_neighbors,
                        }
                        append_csv_row(detailed_csv, row)
                        all_rows.append(row)
                        
                        logger.info(f"  → test_auc={test_metrics['auc']:.4f}, test_acc={test_metrics['acc']:.4f}")
                    except Exception as e:
                        logger.error(f"Baseline {method} failed: {e}", exc_info=True)

    if all_rows:
        df = pd.DataFrame(all_rows)
        
        # Summary by method
        summary = (
            df.groupby(['method'], as_index=False)
              .agg(
                  runs=('run', 'count'),
                  mean_test_auc=('best_test_auc', 'mean'),
                  std_test_auc=('best_test_auc', 'std'),
                  mean_test_f1=('best_test_f1', 'mean'),
                  mean_rejected=('surrogate_rejected_total', 'mean'),
                  mean_time=('elapsed_time_sec', 'mean'),
              )
        )
        
        # Summary by strategy
        summary_by_strategy = (
            df[df['strategy'] != 'baseline'].groupby(['strategy'], as_index=False)
              .agg(
                  runs=('run', 'count'),
                  mean_test_auc=('best_test_auc', 'mean'),
                  std_test_auc=('best_test_auc', 'std'),
                  mean_test_f1=('best_test_f1', 'mean'),
              )
        )
        
        # Summary by graph config (if multiple)
        summary_by_graph = None
        if len(graph_configs) > 1:
            summary_by_graph = (
                df.groupby(['graph_cutoff', 'graph_max_neighbors'], as_index=False)
                  .agg(
                      runs=('run', 'count'),
                      mean_test_auc=('best_test_auc', 'mean'),
                      std_test_auc=('best_test_auc', 'std'),
                  )
            )
        
        logger.info("\n" + "="*70)
        logger.info("Summary by Method")
        logger.info("="*70)
        logger.info(summary.to_string())
        
        if summary_by_strategy is not None and len(summary_by_strategy) > 0:
            logger.info("\n" + "="*70)
            logger.info("Summary by Strategy")
            logger.info("="*70)
            logger.info(summary_by_strategy.to_string())
        
        if summary_by_graph is not None:
            logger.info("\n" + "="*70)
            logger.info("Summary by Graph Configuration")
            logger.info("="*70)
            logger.info(summary_by_graph.to_string())

        # ------------------------------------------------------------------
        # LEADERBOARD MODE: save submission CSV for the best method/run
        # ------------------------------------------------------------------
        if leaderboard_splits is not None and leaderboard_bench_name is not None:
            # Pick the run with the highest val_auc across all methods/configs
            best_row = df.loc[df['best_val_auc'].idxmax()]
            artifact_dir = best_row['artifact_dir']
            logger.info(
                f"\nGenerating leaderboard submission CSV for best run: "
                f"method={best_row['method']}, val_auc={best_row['best_val_auc']:.4f}, "
                f"test_acc={best_row['best_test_acc']:.4f}"
            )
            # The best model checkpoint is stored as 'best_model.pt' in the artifact dir
            best_ckpt = os.path.join(artifact_dir, 'best_model.pth')
            if os.path.exists(best_ckpt):
                try:
                    best_sol_path = os.path.join(artifact_dir, 'best_solution.json')
                    if os.path.exists(best_sol_path):
                        with open(best_sol_path) as f:
                            best_sol = json.load(f)
                        # Re-load matching config loaders
                        _cutoff = float(best_row['graph_cutoff'])
                        _maxnb  = int(best_row['graph_max_neighbors'])
                        _loaders = get_jarvis_loaders(
                            batch_size=args.batch_size,
                            jarvis_dataset='dft_3d',
                            class_target=args.class_target,
                            class_threshold=0.01,
                            cutoff=_cutoff,
                            max_neighbors=_maxnb,
                            max_samples=0,
                            leaderboard_splits=leaderboard_splits,
                            repr_level=args.repr_level,
                        )
                        _tr, _vl, _ts, _in_f, _out_f = _loaders
                        _model = GNN(_in_f, best_sol, _out_f, task='graph').to(get_device())
                        _model.load_state_dict(torch.load(best_ckpt, map_location=get_device()))
                        preds = collect_predictions_with_jids(_model, _ts)
                        csv_name = (
                            f"AI-SinglePropertyClass-{leaderboard_bench_name}"
                            f"-dft_3d-test-acc.csv"
                        )
                        save_leaderboard_csv(preds, os.path.join(args.output_dir, csv_name))
                    else:
                        logger.warning(f"best_solution.json not found in {artifact_dir} — skipping CSV export")
                except Exception as exc:
                    logger.warning(f"Could not generate leaderboard CSV: {exc}")
            else:
                logger.warning(f"best_model.pt not found in {artifact_dir} — skipping CSV export")

    logger.info(f"\nResults saved to {args.output_dir}")
    
    # Save architecture visualizations
    if args.save_architecture_viz:
        logger.info("\nGenerating architecture visualizations...")
        viz_dir = os.path.join(args.output_dir, 'architecture_visualizations')
        try:
            save_architectures_summary(args.output_dir, viz_dir)
            logger.info(f"Architecture visualizations saved to {viz_dir}")
        except Exception as e:
            logger.warning(f"Failed to generate architecture visualizations: {e}")


def visualize_architecture(sol, output_path):
    """Visualize heterogeneous GNN architecture from CTGP solution"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 15)
    ax.axis('off')
    
    # Title
    conv_types = sol.get('conv', ['?'])
    n_layers = len(conv_types)
    conv_str = ' → '.join(conv_types) if n_layers <= 5 else f"{', '.join(conv_types[:3])}... ({n_layers} layers)"
    fig.suptitle(f"Heterogeneous GNN Architecture: {conv_str}\nParams: {sol.get('best_params', 0):,}", 
                 fontsize=16, weight='bold', y=0.98)
    
    # Input node
    input_box = FancyBboxPatch((3.8, 13.5), 2.4, 0.7, boxstyle="round,pad=0.1", 
                               edgecolor='black', facecolor='#e8f4f8', linewidth=2.5)
    ax.add_patch(input_box)
    ax.text(5, 13.85, 'Input Graph', ha='center', va='center', fontsize=12, weight='bold')
    
    # Convolution layers - HETEROGENEOUS
    conv_types = sol.get('conv', ['GCN'])
    heads = sol.get('heads', [1] * len(conv_types))
    hiddens = sol.get('hidden', [64] * len(conv_types))
    
    # Color map for different conv types
    conv_colors = {
        'GCN': '#a0bee5',      # Blue
        'GraphSAGE': '#c8e6c9', # Green
        'GAT': '#fff9c4',       # Yellow
        'GIN': '#ffccbc',       # Orange
        'SG': '#f8bbd0',        # Pink
    }
    
    y_pos = 12.5
    for i, (conv_type, heads_val, hidden_val) in enumerate(zip(conv_types, heads, hiddens)):
        color = conv_colors.get(conv_type, '#e0e0e0')
        
        # Layer box
        conv_box = FancyBboxPatch((2.5, y_pos - 0.75), 5, 0.85, 
                                  boxstyle="round,pad=0.08",
                                  edgecolor='#333333', facecolor=color, 
                                  linewidth=2.2, alpha=0.9)
        ax.add_patch(conv_box)
        
        # Layer label with type, hidden dim, heads
        heads_str = f"({heads_val}h)" if heads_val > 1 else ""
        label = f"Layer {i+1}: {conv_type} | {hidden_val}D {heads_str}"
        ax.text(5, y_pos - 0.35, label, ha='center', va='center', 
                fontsize=10, family='monospace', weight='bold')
        
        # Arrow to next
        if i < len(conv_types) - 1:
            arrow = FancyArrowPatch((5, y_pos - 0.75), (5, y_pos - 1.5),
                                   arrowstyle='->', mutation_scale=22, 
                                   linewidth=2.5, color='#555555')
            ax.add_patch(arrow)
        
        y_pos -= 1.8
    
    # Dropout layer
    dropout = sol.get('drop', 0.3)
    dropout_box = FancyBboxPatch((3.5, y_pos - 0.65), 3, 0.8,
                                 boxstyle="round,pad=0.08",
                                 edgecolor='#d32f2f', facecolor='#ffcccc', 
                                 linewidth=2, alpha=0.85)
    ax.add_patch(dropout_box)
    ax.text(5, y_pos - 0.25, f'Dropout ({dropout:.2f})', ha='center', va='center', 
            fontsize=10, weight='bold')
    
    # Arrow
    arrow = FancyArrowPatch((5, y_pos - 0.65), (5, y_pos - 1.4),
                           arrowstyle='->', mutation_scale=22, 
                           linewidth=2.5, color='#555555')
    ax.add_patch(arrow)
    y_pos -= 1.8
    
    # Classification layers
    cls_hiddens = sol.get('cls_hidden', [])
    for i, cls_hid in enumerate(cls_hiddens):
        cls_box = FancyBboxPatch((3.5, y_pos - 0.65), 3, 0.8,
                                 boxstyle="round,pad=0.08",
                                 edgecolor='#6d597a', facecolor='#e8d5f2', 
                                 linewidth=2, alpha=0.85)
        ax.add_patch(cls_box)
        ax.text(5, y_pos - 0.25, f'FC({cls_hid}D)', ha='center', va='center', 
                fontsize=9, weight='bold')
        
        if i < len(cls_hiddens) - 1:
            arrow = FancyArrowPatch((5, y_pos - 0.65), (5, y_pos - 1.4),
                                   arrowstyle='->', mutation_scale=20, 
                                   linewidth=2, color='#666666')
            ax.add_patch(arrow)
        else:
            arrow = FancyArrowPatch((5, y_pos - 0.65), (5, y_pos - 1.4),
                                   arrowstyle='->', mutation_scale=22, 
                                   linewidth=2.5, color='#555555')
            ax.add_patch(arrow)
        y_pos -= 1.6
    
    # Output
    output_box = FancyBboxPatch((4, y_pos - 0.65), 2, 0.8,
                                boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor='#c8e6c9', 
                                linewidth=2.5)
    ax.add_patch(output_box)
    ax.text(5, y_pos - 0.25, 'Output (2)', ha='center', va='center', 
            fontsize=11, weight='bold')
    
    # Legend with architecture details
    legend_text = f"""
HETEROGENEOUS ARCHITECTURE SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Convolutions:
  {chr(10).join([f"  • Layer {i+1}: {t} dim={h} h={heads_val}" 
                  for i, (t, h, heads_val) in enumerate(zip(conv_types, hiddens, heads))])}

Global Pooling: Mean

Classification:
  {f"Hidden layers: {', '.join(map(str, cls_hiddens)) if cls_hiddens else 'None'}"}

Regularization:
  Dropout: {dropout:.2f}
  Init: {sol.get('init', 'unknown')}

Performance:
  Test AUC: {sol.get('best_test_auc', 0.0):.4f}
  Params: {sol.get('best_params', 0):,}
"""
    
    ax.text(0.2, 7, legend_text, fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='#fffacd', alpha=0.4, pad=0.8),
            verticalalignment='top')
    
    # Legend for conv types
    legend_y = 1.5
    legend_items = []
    for conv_type, color in conv_colors.items():
        sample_box = FancyBboxPatch((0.2, legend_y), 0.3, 0.3, 
                                    boxstyle="round,pad=0.02",
                                    facecolor=color, edgecolor='#333', linewidth=1)
        ax.add_patch(sample_box)
        ax.text(0.7, legend_y + 0.15, conv_type, fontsize=9, va='center')
        legend_y -= 0.5
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def get_best_architectures(result_dir, topk=3):
    """Get top-K best architectures from experiment results"""
    import pandas as pd
    
    detailed_csv = os.path.join(result_dir, 'detailed_results.csv')
    if not os.path.exists(detailed_csv):
        return []
    
    df = pd.read_csv(detailed_csv)
    # Only CTGP methods (not baselines)
    ctgp_df = df[~df['method'].str.contains('baseline_')]
    # Sort by test AUC, take top-K
    best = ctgp_df.nlargest(topk, 'best_test_auc')
    return best[['method', 'run', 'best_test_auc', 'artifact_dir']].to_dict('records')


def save_architectures_summary(result_dir, output_dir):
    """Load and save visualizations of best architectures"""
    best_archs = get_best_architectures(result_dir, topk=5)
    
    if not best_archs:
        logger.warning("No best architectures found")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, arch_info in enumerate(best_archs):
        artifact_dir = arch_info['artifact_dir']
        sol_file = os.path.join(artifact_dir, 'best_solution.json')
        
        if os.path.exists(sol_file):
            try:
                with open(sol_file, 'r') as f:
                    sol = json.load(f)
                
                # Add metrics to solution
                sol['best_test_auc'] = arch_info['best_test_auc']
                sol['method'] = arch_info['method']
                sol['run'] = arch_info['run']
                
                output_path = os.path.join(output_dir, f'architecture_rank_{idx+1}_auc_{arch_info["best_test_auc"]:.4f}.png')
                visualize_architecture(sol, output_path)
                logger.info(f"Saved architecture visualization #{idx+1} (AUC={arch_info['best_test_auc']:.4f})")
            except Exception as e:
                logger.error(f"Failed to visualize architecture: {e}")


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='CTGP + Surrogate for JARVIS Metal Property Classification'
    )
    
    # Basic experiment settings
    parser.add_argument('--output_dir', type=str, default='jarvis_ctgp_experiments',
                        help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--runs', type=int, default=3,
                        help='Number of independent runs per method')
    parser.add_argument('--generations', type=int, default=50,
                        help='Number of evolution generations')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--max_jarvis_samples', type=int, default=5000,
                        help='Max samples from JARVIS dataset')

    # Surrogate model settings
    parser.add_argument('--surrogate_warmup', type=int, default=8,
                        help='Warmup generations before using surrogate')
    parser.add_argument('--surrogate_train_every', type=int, default=3,
                        help='Retrain surrogate every N generations')
    parser.add_argument('--surrogate_train_epochs', type=int, default=40,
                        help='Epochs for surrogate training')
    parser.add_argument('--surrogate_acceptance_threshold', type=float, default=0.5,
                        help='Surrogate acceptance threshold')

    # Graph construction options
    parser.add_argument('--graph_cutoff_configs', type=float, nargs='+', default=[8.0],
                        help='Graph cutoff radii (Ångströms) to test')
    parser.add_argument('--graph_maxneighbors_configs', type=int, nargs='+', default=[12],
                        help='Max neighbors for k-NN graph construction')
    
    # Classification target (from JARVIS-DFT ALIGNN paper: 10 tasks)
    parser.add_argument('--class_target', type=str, default='metal_opt',
                        choices=['metal_opt', 'metal_mbj', 'magnetic', 'stable', 'slme', 
                                'spillage', 'seebeck_n', 'seebeck_p', 'powerfactor_n', 'powerfactor_p'],
                        help='Classification target (from JARVIS-DFT ALIGNN paper)')
    
    # Multiple test configurations (comprehensive testing)
    parser.add_argument('--test_configs', action='store_true',
                        help='Run comprehensive test configs: cutoff=[6,8,10], neighbors=[8,12,16]')
    parser.add_argument('--test_different_mutations', action='store_true',
                        help='Test different CTGP mutation rates')
    parser.add_argument('--test_population_sizes', action='store_true',
                        help='Test different evolution population sizes')
    
    # Which methods to run
    parser.add_argument('--run_baselines', action='store_true',
                        help='Run baseline methods (MLP, RandomForest, etc)')
    parser.add_argument('--run_ctgp', action='store_true',
                        help='Run CTGP methods')
    parser.add_argument('--run_all', action='store_true',
                        help='Run all methods (CTGP + baselines)')
    parser.add_argument(
        '--methods', nargs='+', default=None,
        help='Subset of CTGP methods to run (with --run_ctgp). E.g. ctgp_1plus4 ctgp_surrogate_1plus4',
    )
    parser.add_argument(
        '--start_run', type=int, default=1,
        help='First run index (run_{start_run}). Use to append without overwriting existing runs.',
    )
    
    # Visualization and output
    parser.add_argument('--save_architecture_viz', action='store_true',
                        help='Save visualizations of best found architectures')
    parser.add_argument('--num_best_architectures', type=int, default=5,
                        help='Number of best architectures to visualize')

    # ---- JARVIS Leaderboard mode ----
    parser.add_argument(
        '--leaderboard', action='store_true',
        help=(
            'Use pre-determined JARVIS Leaderboard train/val/test JID splits '
            'instead of random 80/10/10 split.  Enables apple-to-apple comparison '
            'with published leaderboard results.  Supported targets: '
            + ', '.join(LEADERBOARD_BENCHMARK_MAP.keys())
        )
    )
    parser.add_argument(
        '--leaderboard_cache_dir', type=str, default='.jarvis_lb_cache',
        help='Local directory to cache downloaded leaderboard benchmark JSON files'
    )
    parser.add_argument(
        '--repr_level', type=int, default=0, choices=[0, 1, 2, 3],
        help=(
            'Graph representation level: '
            '0=atomic_number[1D] (default/original), '
            '1=cgcnn[92D] (ALIGNN-style atom embeddings), '
            '2=cgcnn[92D]+RBF_dist[10D] (bond distances as Gaussian basis), '
            '3=cgcnn[92D]+RBF_dist[10D]+angle_stats[4D] (bond angles approx)'
        )
    )

