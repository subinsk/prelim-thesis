"""
Generate all publication-quality figures for the thesis.

Produces 6-8 figures from existing experiment results:
1. Accuracy by Condition (2-hop, all models)
2. Behavior Breakdown (CFR/POR stacked bar)
3. Model Comparison (grouped bar, 4 models)
4. 2-Hop vs 3-Hop Accuracy (line chart)
5. Conflict Type Comparison (grouped bar)
6. Entity Popularity Effect (grouped bar by popularity bin)
7. Error Propagation Rate (line chart by hop position)
8. Comprehensive Statistics Table (markdown)

Usage:
    python experiments/generate_all_figures.py
    python experiments/generate_all_figures.py --skip-popularity  # skip Wikipedia API calls
"""

import json
import os
import sys
import argparse
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

from src.analysis.statistics import compute_full_statistics, binomial_ci, cohens_h, effect_size_label

RESULTS_BASE = "outputs/results"
FIGURES_DIR = "outputs/figures"
COLORS = {
    'baseline': '#2ecc71',
    'hop1': '#e74c3c',
    'hop2': '#f39c12',
    'hop3': '#9b59b6',
}
MODEL_COLORS = ['#3498db', '#e67e22', '#2ecc71', '#9b59b6']

MODELS = [
    {"id": "llama-3.3-70b-versatile", "label": "Llama-3.3-70B"},
    {"id": "llama-3.1-8b-instant",    "label": "Llama-3.1-8B"},
    {"id": "gemini-2.5-flash-lite",    "label": "Gemini-2.5-Flash-Lite"},
    {"id": "qwen/qwen3-32b",          "label": "Qwen3-32B"},
]


def load_model_results(experiment_name="experiment"):
    """Load results for all models."""
    all_data = {}
    for model in MODELS:
        path = os.path.join(RESULTS_BASE, model['id'], f"{experiment_name}.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                all_data[model['id']] = {'label': model['label'], 'data': json.load(f)}
    return all_data


def fig1_accuracy_by_condition(all_data):
    """Figure 1: Accuracy by condition for all models (grouped bar)."""
    fig, ax = plt.subplots(figsize=(12, 7))
    conditions = ['no_conflict', 'conflict_hop1', 'conflict_hop2']
    labels = ['No Conflict\n(Baseline)', 'Conflict\n@Hop 1', 'Conflict\n@Hop 2']
    x = np.arange(len(labels))
    n_models = len(all_data)
    width = 0.7 / n_models

    for idx, (model_id, entry) in enumerate(all_data.items()):
        metrics = entry['data']['metrics']
        accs = [metrics[c]['accuracy'] * 100 for c in conditions]
        ns = [metrics[c]['n'] for c in conditions]
        # Error bars from binomial CI
        errs_lo = []
        errs_hi = []
        for c in conditions:
            lo, hi = binomial_ci(metrics[c]['accuracy'], metrics[c]['n'])
            errs_lo.append((metrics[c]['accuracy'] - lo) * 100)
            errs_hi.append((hi - metrics[c]['accuracy']) * 100)

        offset = (idx - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, accs, width, label=f"{entry['label']} (n={ns[0]})",
                      color=MODEL_COLORS[idx % len(MODEL_COLORS)], edgecolor='black', linewidth=0.8,
                      yerr=[errs_lo, errs_hi], capsize=3, error_kw={'linewidth': 1})

    ax.set_ylabel('Accuracy (%)', fontsize=13)
    ax.set_title('Impact of Knowledge Conflicts on Multi-Hop QA (2-Hop)', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig1_accuracy_by_condition.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def fig2_behavior_breakdown(all_data):
    """Figure 2: CFR/POR/Other stacked bar for all models."""
    fig, axes = plt.subplots(1, len(all_data), figsize=(5 * len(all_data), 6), squeeze=False)

    for ax_idx, (model_id, entry) in enumerate(all_data.items()):
        ax = axes[0][ax_idx]
        metrics = entry['data']['metrics']
        conditions = ['conflict_hop1', 'conflict_hop2']
        labels = ['Hop 1', 'Hop 2']

        cfr = [metrics[c]['context_following_rate'] * 100 for c in conditions]
        por = [metrics[c]['parametric_override_rate'] * 100 for c in conditions]
        other = [100 - cfr[i] - por[i] for i in range(len(conditions))]

        x = np.arange(len(labels))
        w = 0.5
        ax.bar(x, cfr, w, label='Followed Context (CFR)', color='#e74c3c')
        ax.bar(x, por, w, bottom=cfr, label='Parametric Override (POR)', color='#2ecc71')
        ax.bar(x, other, w, bottom=[cfr[i] + por[i] for i in range(len(conditions))],
               label='Other', color='#95a5a6')

        ax.set_title(entry['label'], fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 100)
        ax.set_ylabel('Percentage (%)' if ax_idx == 0 else '')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if ax_idx == 0:
            ax.legend(fontsize=8, loc='upper right')

    plt.suptitle('Model Behavior Under Knowledge Conflict', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig2_behavior_breakdown.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def fig3_model_comparison_summary(all_data):
    """Figure 3: Summary — baseline accuracy vs accuracy drop per model."""
    fig, ax = plt.subplots(figsize=(10, 7))

    model_labels = []
    baselines = []
    avg_drops = []

    for model_id, entry in all_data.items():
        metrics = entry['data']['metrics']
        baseline = metrics['no_conflict']['accuracy'] * 100
        conflict_accs = []
        for c in ['conflict_hop1', 'conflict_hop2']:
            if c in metrics:
                conflict_accs.append(metrics[c]['accuracy'] * 100)
        avg_conflict = np.mean(conflict_accs)
        drop = baseline - avg_conflict

        model_labels.append(entry['label'])
        baselines.append(baseline)
        avg_drops.append(drop)

    x = np.arange(len(model_labels))
    width = 0.35
    ax.bar(x - width/2, baselines, width, label='Baseline Accuracy', color='#3498db', edgecolor='black')
    ax.bar(x + width/2, avg_drops, width, label='Avg Accuracy Drop (pp)', color='#e74c3c', edgecolor='black')

    for i, (b, d) in enumerate(zip(baselines, avg_drops)):
        ax.annotate(f'{b:.1f}%', xy=(i - width/2, b), xytext=(0, 4),
                    textcoords="offset points", ha='center', fontsize=10, fontweight='bold')
        ax.annotate(f'{d:.1f}pp', xy=(i + width/2, d), xytext=(0, 4),
                    textcoords="offset points", ha='center', fontsize=10, fontweight='bold', color='darkred')

    ax.set_ylabel('Percentage', fontsize=13)
    ax.set_title('Baseline Accuracy vs Conflict Impact by Model', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig3_model_comparison_summary.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def fig4_hop_comparison():
    """Figure 4: 2-hop vs 3-hop baseline and conflict accuracy."""
    musique_data = load_model_results("musique_experiment")
    hotpot_data = load_model_results("experiment")

    if not musique_data:
        print("  SKIP fig4: no MuSiQue results")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    for idx, model in enumerate(MODELS):
        mid = model['id']
        if mid not in hotpot_data or mid not in musique_data:
            continue

        h_metrics = hotpot_data[mid]['data']['metrics']
        m_metrics = musique_data[mid]['data']['metrics']

        # 2-hop: baseline, conflict avg
        h_base = h_metrics['no_conflict']['accuracy'] * 100
        h_conflict = np.mean([h_metrics[c]['accuracy'] * 100
                              for c in ['conflict_hop1', 'conflict_hop2']])

        # 3-hop: baseline, conflict avg
        m_base = m_metrics['no_conflict']['accuracy'] * 100
        m_conflicts = [m_metrics[c]['accuracy'] * 100
                       for c in ['conflict_hop1', 'conflict_hop2', 'conflict_hop3']
                       if c in m_metrics]
        m_conflict = np.mean(m_conflicts) if m_conflicts else 0

        ax.plot([2, 3], [h_base, m_base], 'o-', color=MODEL_COLORS[idx],
                label=f"{model['label']} (baseline)", linewidth=2, markersize=8)
        ax.plot([2, 3], [h_conflict, m_conflict], 's--', color=MODEL_COLORS[idx],
                linewidth=2, markersize=8, alpha=0.6)

    ax.set_xlabel('Number of Hops', fontsize=13)
    ax.set_ylabel('Accuracy (%)', fontsize=13)
    ax.set_title('Accuracy by Chain Length: 2-Hop vs 3-Hop', fontsize=15, fontweight='bold')
    ax.set_xticks([2, 3])
    ax.set_xlim(1.5, 3.5)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add annotation explaining solid=baseline, dashed=conflict
    ax.annotate('Solid = baseline, Dashed = avg conflict', xy=(0.02, 0.02),
                xycoords='axes fraction', fontsize=9, style='italic', color='gray')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig4_hop_comparison.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def fig5_conflict_types():
    """Figure 5: Accuracy drop by conflict type."""
    conflict_types = ['factual', 'temporal', 'numerical']
    type_colors = {'factual': '#3498db', 'temporal': '#e67e22', 'numerical': '#2ecc71'}

    # Load conflict type results for all models
    all_type_data = {}
    for model in MODELS:
        model_data = {}
        for ctype in conflict_types:
            path = os.path.join(RESULTS_BASE, model['id'], f"conflict_type_{ctype}.json")
            if os.path.exists(path):
                with open(path, 'r') as f:
                    model_data[ctype] = json.load(f)['metrics']
        if model_data:
            all_type_data[model['id']] = {'label': model['label'], 'types': model_data}

    if not all_type_data:
        print("  SKIP fig5: no conflict type results")
        return

    # Summary: accuracy drop by type per model
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(all_type_data))
    width = 0.25

    for t_idx, ctype in enumerate(conflict_types):
        drops = []
        for model_id, entry in all_type_data.items():
            if ctype in entry['types']:
                m = entry['types'][ctype]
                baseline = m.get('no_conflict', {}).get('accuracy', 0)
                conflict_accs = [m[c]['accuracy'] for c in ['conflict_hop1', 'conflict_hop2'] if c in m]
                drop = (baseline - np.mean(conflict_accs)) * 100 if conflict_accs else 0
            else:
                drop = 0
            drops.append(drop)

        offset = (t_idx - 1) * width
        bars = ax.bar(x + offset, drops, width, label=ctype.capitalize(),
                      color=type_colors[ctype], edgecolor='black', linewidth=0.8)
        for bar, d in zip(bars, drops):
            ax.annotate(f'{d:.1f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')

    ax.set_ylabel('Accuracy Drop (pp)', fontsize=13)
    ax.set_title('Accuracy Drop by Conflict Type', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([entry['label'] for entry in all_type_data.values()], fontsize=11)
    ax.legend(fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig5_conflict_types.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def fig6_entity_popularity():
    """Figure 6: Entity popularity effect on conflict resolution."""
    analysis_path = os.path.join(RESULTS_BASE, "entity_popularity_analysis.json")
    if not os.path.exists(analysis_path):
        print("  SKIP fig6: run entity_popularity.py first")
        return

    with open(analysis_path, 'r') as f:
        results = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    bins = ['high', 'medium', 'low']
    bin_labels = ['High\nPopularity', 'Medium\nPopularity', 'Low\nPopularity']
    x = np.arange(len(bins))
    colors = ['#2ecc71', '#f39c12', '#e74c3c']

    # Panel 1: Accuracy by popularity
    for cond_idx, (cond, label) in enumerate([('no_conflict', 'Baseline'),
                                               ('conflict_hop1', 'Conflict@Hop1'),
                                               ('conflict_hop2', 'Conflict@Hop2')]):
        accs = []
        for b in bins:
            acc = results.get(b, {}).get(cond, {}).get('accuracy', 0) * 100
            accs.append(acc)
        width = 0.25
        offset = (cond_idx - 1) * width
        ax1.bar(x + offset, accs, width, label=label, color=colors[cond_idx], edgecolor='black', linewidth=0.8)

    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Accuracy by Entity Popularity', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(bin_labels, fontsize=10)
    ax1.legend(fontsize=9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel 2: CFR and POR by popularity
    for metric, color, label in [('context_following_rate', '#e74c3c', 'CFR'),
                                  ('parametric_override_rate', '#2ecc71', 'POR')]:
        vals = []
        for b in bins:
            # Average across hop1 and hop2
            v1 = results.get(b, {}).get('conflict_hop1', {}).get(metric, 0)
            v2 = results.get(b, {}).get('conflict_hop2', {}).get(metric, 0)
            vals.append(((v1 + v2) / 2) * 100)
        ax2.plot(bin_labels, vals, 'o-', color=color, label=label, linewidth=2, markersize=8)

    ax2.set_ylabel('Rate (%)', fontsize=12)
    ax2.set_title('CFR/POR by Entity Popularity', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.suptitle('Entity Popularity Effect on Conflict Resolution', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig6_entity_popularity.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def fig7_error_propagation():
    """Figure 7: Error propagation in 3-hop chains."""
    musique_data = load_model_results("musique_experiment")
    if not musique_data:
        print("  SKIP fig7: no MuSiQue results")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    for idx, model in enumerate(MODELS):
        mid = model['id']
        if mid not in musique_data:
            continue

        m = musique_data[mid]['data']['metrics']
        baseline = m.get('no_conflict', {}).get('accuracy', 0) * 100
        hops = []
        accs = []
        for hop_num, cond in enumerate(['conflict_hop1', 'conflict_hop2', 'conflict_hop3'], 1):
            if cond in m:
                hops.append(hop_num)
                accs.append(m[cond]['accuracy'] * 100)

        if hops:
            ax.plot(hops, accs, 'o-', color=MODEL_COLORS[idx],
                    label=f"{model['label']} (baseline={baseline:.0f}%)",
                    linewidth=2, markersize=8)
            ax.axhline(y=baseline, color=MODEL_COLORS[idx], linestyle=':', alpha=0.3)

    ax.set_xlabel('Conflict Position (Hop)', fontsize=13)
    ax.set_ylabel('Accuracy (%)', fontsize=13)
    ax.set_title('Error Propagation: Accuracy by Conflict Position (3-Hop)', fontsize=15, fontweight='bold')
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Hop 1', 'Hop 2', 'Hop 3'])
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig7_error_propagation.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def fig8_comprehensive_stats_table(all_data):
    """Figure 8: Generate comprehensive statistics markdown table."""
    from src.analysis.statistics import generate_statistics_table

    full_table = "# Comprehensive Statistics\n\n"
    full_table += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"

    for model in MODELS:
        path = os.path.join(RESULTS_BASE, model['id'], "experiment.json")
        if os.path.exists(path):
            full_table += generate_statistics_table(path, model['label'])
            full_table += "\n"

    out_path = os.path.join(RESULTS_BASE, "comprehensive_statistics.md")
    with open(out_path, 'w') as f:
        f.write(full_table)
    print(f"  Saved: {out_path}")


def main(skip_popularity=False):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    all_data = load_model_results()

    if not all_data:
        print("No results found. Run experiments first.")
        return

    print(f"\nGenerating figures from {len(all_data)} models...")
    print(f"Models: {', '.join(entry['label'] for entry in all_data.values())}\n")

    print("Fig 1: Accuracy by Condition (2-hop)")
    fig1_accuracy_by_condition(all_data)

    print("Fig 2: Behavior Breakdown (CFR/POR)")
    fig2_behavior_breakdown(all_data)

    print("Fig 3: Model Comparison Summary")
    fig3_model_comparison_summary(all_data)

    print("Fig 4: 2-Hop vs 3-Hop Comparison")
    fig4_hop_comparison()

    print("Fig 5: Conflict Type Comparison")
    fig5_conflict_types()

    if not skip_popularity:
        print("Fig 6: Entity Popularity Effect")
        # Run popularity analysis first if not done
        analysis_path = os.path.join(RESULTS_BASE, "entity_popularity_analysis.json")
        if not os.path.exists(analysis_path):
            print("  Running entity popularity analysis first...")
            from src.analysis.entity_popularity import analyze_popularity_effect
            primary_path = os.path.join(RESULTS_BASE, "llama-3.3-70b-versatile", "experiment.json")
            if os.path.exists(primary_path):
                results, _ = analyze_popularity_effect(primary_path)
                with open(analysis_path, 'w') as f:
                    json.dump(results, f, indent=2)
        fig6_entity_popularity()
    else:
        print("Fig 6: SKIPPED (--skip-popularity)")

    print("Fig 7: Error Propagation (3-hop)")
    fig7_error_propagation()

    print("Fig 8: Comprehensive Statistics Table")
    fig8_comprehensive_stats_table(all_data)

    print(f"\nDone! All figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-popularity', action='store_true',
                        help='Skip entity popularity analysis (avoids Wikipedia API calls)')
    args = parser.parse_args()
    main(skip_popularity=args.skip_popularity)
