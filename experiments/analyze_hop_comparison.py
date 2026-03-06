"""
2-hop vs 3-hop comparison analysis.

Loads results from both HotpotQA (2-hop) and MuSiQue (3-hop) experiments,
generates comparison figures and summary tables.

Usage:
    python experiments/analyze_hop_comparison.py
"""

import json
import os
import sys
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

MODELS = [
    {"id": "llama-3.3-70b-versatile", "label": "Llama-3.3-70B"},
    {"id": "llama-3.1-8b-instant",    "label": "Llama-3.1-8B"},
    {"id": "gemini-2.5-flash-lite",    "label": "Gemini-2.5-Flash-Lite"},
    {"id": "qwen/qwen3-32b",           "label": "Qwen3-32B"},
]

RESULTS_BASE = "outputs/results"
FIGURES_DIR = "outputs/figures"


def load_experiment(model_id, experiment_name):
    """Load experiment results for a model. Returns metrics dict or None."""
    path = os.path.join(RESULTS_BASE, model_id, f"{experiment_name}.json")
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        data = json.load(f)
    return data.get('metrics', {})


def plot_accuracy_panels(data_2hop, data_3hop):
    """Create side-by-side panels: 2-hop vs 3-hop accuracy by condition."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    colors = ['#3498db', '#e67e22', '#2ecc71', '#9b59b6']

    # 2-hop panel
    conditions_2hop = ['no_conflict', 'conflict_hop1', 'conflict_hop2']
    labels_2hop = ['No Conflict', 'Conflict\n@Hop1', 'Conflict\n@Hop2']
    x = np.arange(len(labels_2hop))
    n_models = len(data_2hop)
    width = 0.7 / max(n_models, 1)

    for idx, (model_id, entry) in enumerate(data_2hop.items()):
        metrics = entry['metrics']
        accs = [metrics[c]['accuracy'] * 100 if c in metrics else 0 for c in conditions_2hop]
        offset = (idx - (n_models - 1) / 2) * width
        bars = ax1.bar(x + offset, accs, width, label=entry['label'],
                       color=colors[idx % len(colors)], edgecolor='black', linewidth=1)
        for bar in bars:
            h = bar.get_height()
            ax1.annotate(f'{h:.0f}%', xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax1.set_ylabel('Accuracy (%)', fontsize=13)
    ax1.set_title('2-Hop (HotpotQA)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_2hop, fontsize=11)
    ax1.set_ylim(0, 100)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 3-hop panel
    conditions_3hop = ['no_conflict', 'conflict_hop1', 'conflict_hop2', 'conflict_hop3']
    labels_3hop = ['No Conflict', 'Conflict\n@Hop1', 'Conflict\n@Hop2', 'Conflict\n@Hop3']
    x3 = np.arange(len(labels_3hop))
    n_models_3 = len(data_3hop)
    width3 = 0.7 / max(n_models_3, 1)

    for idx, (model_id, entry) in enumerate(data_3hop.items()):
        metrics = entry['metrics']
        accs = [metrics[c]['accuracy'] * 100 if c in metrics else 0 for c in conditions_3hop]
        offset = (idx - (n_models_3 - 1) / 2) * width3
        bars = ax2.bar(x3 + offset, accs, width3, label=entry['label'],
                       color=colors[idx % len(colors)], edgecolor='black', linewidth=1)
        for bar in bars:
            h = bar.get_height()
            ax2.annotate(f'{h:.0f}%', xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax2.set_title('3-Hop (MuSiQue)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x3)
    ax2.set_xticklabels(labels_3hop, fontsize=11)
    ax2.legend(fontsize=9, loc='upper right')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.suptitle('Impact of Knowledge Conflicts: 2-Hop vs 3-Hop Reasoning',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, 'hop_comparison_accuracy.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_error_propagation(data_3hop):
    """Line chart: accuracy drop vs conflict position for 3-hop experiments."""

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#3498db', '#e67e22', '#2ecc71', '#9b59b6']
    markers = ['o', 's', '^', 'D']

    for idx, (model_id, entry) in enumerate(data_3hop.items()):
        metrics = entry['metrics']
        baseline = metrics.get('no_conflict', {}).get('accuracy', 0) * 100

        positions = [1, 2, 3]
        drops = []
        for hop in positions:
            cond = f'conflict_hop{hop}'
            acc = metrics.get(cond, {}).get('accuracy', 0) * 100
            drops.append(baseline - acc)

        ax.plot(positions, drops, marker=markers[idx % len(markers)],
                color=colors[idx % len(colors)], linewidth=2.5, markersize=10,
                label=f"{entry['label']} (baseline: {baseline:.0f}%)")

    ax.set_xlabel('Conflict Position (Hop)', fontsize=13)
    ax.set_ylabel('Accuracy Drop (pp)', fontsize=13)
    ax.set_title('Error Propagation: Accuracy Drop by Conflict Position (3-Hop)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Hop 1', 'Hop 2', 'Hop 3'], fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'error_propagation.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def generate_summary_table(data_2hop, data_3hop):
    """Generate markdown summary table comparing 2-hop and 3-hop results."""

    table = "# 2-Hop vs 3-Hop Comparison\n\n"
    table += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"

    # Overview table
    table += "## Overview\n\n"
    table += "| Model | Dataset | Hops | Baseline Acc | Avg Conflict Acc | Max Drop (pp) | Avg CFR | Avg POR |\n"
    table += "|-------|---------|------|-------------|-----------------|--------------|---------|--------|\n"

    for model in MODELS:
        mid = model['id']
        label = model['label']

        # 2-hop
        if mid in data_2hop:
            m = data_2hop[mid]['metrics']
            baseline = m.get('no_conflict', {}).get('accuracy', 0)
            conflict_accs = []
            cfrs = []
            pors = []
            for cond in ['conflict_hop1', 'conflict_hop2']:
                if cond in m:
                    conflict_accs.append(m[cond]['accuracy'])
                    cfrs.append(m[cond].get('context_following_rate', 0))
                    pors.append(m[cond].get('parametric_override_rate', 0))
            avg_conflict = np.mean(conflict_accs) if conflict_accs else 0
            max_drop = (baseline - min(conflict_accs)) * 100 if conflict_accs else 0
            avg_cfr = np.mean(cfrs) if cfrs else 0
            avg_por = np.mean(pors) if pors else 0
            table += f"| {label} | HotpotQA | 2 | {baseline:.1%} | {avg_conflict:.1%} | {max_drop:.1f} | {avg_cfr:.1%} | {avg_por:.1%} |\n"

        # 3-hop
        if mid in data_3hop:
            m = data_3hop[mid]['metrics']
            baseline = m.get('no_conflict', {}).get('accuracy', 0)
            conflict_accs = []
            cfrs = []
            pors = []
            for cond in ['conflict_hop1', 'conflict_hop2', 'conflict_hop3']:
                if cond in m:
                    conflict_accs.append(m[cond]['accuracy'])
                    cfrs.append(m[cond].get('context_following_rate', 0))
                    pors.append(m[cond].get('parametric_override_rate', 0))
            avg_conflict = np.mean(conflict_accs) if conflict_accs else 0
            max_drop = (baseline - min(conflict_accs)) * 100 if conflict_accs else 0
            avg_cfr = np.mean(cfrs) if cfrs else 0
            avg_por = np.mean(pors) if pors else 0
            table += f"| {label} | MuSiQue | 3 | {baseline:.1%} | {avg_conflict:.1%} | {max_drop:.1f} | {avg_cfr:.1%} | {avg_por:.1%} |\n"

    # Cross-hop statistical comparison per model
    table += "\n## Cross-Hop Comparison (Baseline Accuracy: 2-hop vs 3-hop)\n\n"
    table += "| Model | 2-Hop Baseline | 3-Hop Baseline | Drop (pp) | Chi2 | p-value | Significant |\n"
    table += "|-------|---------------|---------------|-----------|------|---------|-------------|\n"

    for model in MODELS:
        mid = model['id']
        if mid in data_2hop and mid in data_3hop:
            m2 = data_2hop[mid]['metrics'].get('no_conflict', {})
            m3 = data_3hop[mid]['metrics'].get('no_conflict', {})
            a2, n2 = m2.get('accuracy', 0), m2.get('n', 0)
            a3, n3 = m3.get('accuracy', 0), m3.get('n', 0)
            drop = (a2 - a3) * 100
            ct = [[int(a2 * n2), n2 - int(a2 * n2)],
                  [int(a3 * n3), n3 - int(a3 * n3)]]
            try:
                chi2, p_val, _, _ = stats.chi2_contingency(ct)
            except ValueError:
                chi2, p_val = 0, 1.0
            sig = "Yes" if p_val < 0.05 else "No"
            table += f"| {model['label']} | {a2:.1%} | {a3:.1%} | {drop:+.1f} | {chi2:.2f} | {p_val:.4f} | {sig} |\n"

    # Error propagation summary for 3-hop
    table += "\n## Error Propagation (3-Hop): Accuracy Drop by Conflict Position\n\n"
    table += "| Model | Baseline | Drop@Hop1 | Drop@Hop2 | Drop@Hop3 | Most Vulnerable |\n"
    table += "|-------|----------|-----------|-----------|-----------|----------------|\n"

    for model in MODELS:
        mid = model['id']
        if mid in data_3hop:
            m = data_3hop[mid]['metrics']
            baseline = m.get('no_conflict', {}).get('accuracy', 0) * 100
            drops = {}
            for hop in [1, 2, 3]:
                cond = f'conflict_hop{hop}'
                acc = m.get(cond, {}).get('accuracy', 0) * 100
                drops[hop] = baseline - acc
            worst_hop = max(drops, key=drops.get)
            table += (f"| {model['label']} | {baseline:.1f}% | "
                      f"{drops[1]:+.1f}pp | {drops[2]:+.1f}pp | {drops[3]:+.1f}pp | "
                      f"Hop {worst_hop} ({drops[worst_hop]:+.1f}pp) |\n")

    return table


if __name__ == "__main__":
    print("Loading experiment results...")

    data_2hop = {}
    data_3hop = {}

    for model in MODELS:
        mid = model['id']
        label = model['label']

        metrics_2hop = load_experiment(mid, "experiment")
        if metrics_2hop:
            n = metrics_2hop.get('no_conflict', {}).get('n', 0)
            data_2hop[mid] = {'label': label, 'metrics': metrics_2hop}
            print(f"  2-hop: {label} ({n} examples)")

        metrics_3hop = load_experiment(mid, "musique_experiment")
        if metrics_3hop:
            n = metrics_3hop.get('no_conflict', {}).get('n', 0)
            data_3hop[mid] = {'label': label, 'metrics': metrics_3hop}
            print(f"  3-hop: {label} ({n} examples)")

    if not data_2hop and not data_3hop:
        print("No results found. Run experiments first.")
        sys.exit(1)

    # Generate figures
    print("\nGenerating figures...")
    if data_2hop and data_3hop:
        plot_accuracy_panels(data_2hop, data_3hop)
    if data_3hop:
        plot_error_propagation(data_3hop)

    # Generate summary table
    print("\nGenerating summary table...")
    table = generate_summary_table(data_2hop, data_3hop)

    os.makedirs(RESULTS_BASE, exist_ok=True)
    path = os.path.join(RESULTS_BASE, 'hop_comparison.md')
    with open(path, 'w') as f:
        f.write(table)
    print(f"  Saved: {path}")
    print(f"\n{table}")
