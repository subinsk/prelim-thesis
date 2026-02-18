import json
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def load_results(filepath: str) -> dict:
    """Load experiment results."""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_accuracy_comparison(metrics: dict, save_path: str = 'outputs/figures/accuracy_comparison.png'):
    """Create bar chart comparing accuracy across conditions."""

    conditions = ['no_conflict', 'conflict_hop1', 'conflict_hop2']
    labels = ['No Conflict\n(Baseline)', 'Conflict at\nHop 1', 'Conflict at\nHop 2']
    colors = ['#2ecc71', '#e74c3c', '#f39c12']

    accuracies = [metrics[c]['accuracy'] * 100 for c in conditions]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(labels, accuracies, color=colors, edgecolor='black', linewidth=1.5)

    drop_hop1 = accuracies[0] - accuracies[1]
    drop_hop2 = accuracies[0] - accuracies[2]

    for idx, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        # Accuracy label on top of bar
        ax.annotate(f'{acc:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 18),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=14, fontweight='bold')
        # Drop annotation just above bar for conflict conditions
        if idx == 1:
            ax.annotate(f'\u2193 {drop_hop1:.1f}pp',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        color='red', fontsize=11, fontweight='bold')
        elif idx == 2:
            ax.annotate(f'\u2193 {drop_hop2:.1f}pp',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        color='red', fontsize=11, fontweight='bold')

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Impact of Knowledge Conflicts on Multi-Hop Reasoning', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {save_path}")


def plot_behavior_breakdown(metrics: dict, save_path: str = 'outputs/figures/behavior_breakdown.png'):
    """Create stacked bar chart showing CFR vs POR."""

    conditions = ['conflict_hop1', 'conflict_hop2']
    labels = ['Conflict at Hop 1', 'Conflict at Hop 2']

    cfr = [metrics[c]['context_following_rate'] * 100 for c in conditions]
    por = [metrics[c]['parametric_override_rate'] * 100 for c in conditions]
    other = [100 - cfr[i] - por[i] for i in range(len(conditions))]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(labels))
    width = 0.6

    ax.bar(x, cfr, width, label='Followed Context (Wrong)', color='#e74c3c')
    ax.bar(x, por, width, bottom=cfr, label='Used Parametric (Correct)', color='#2ecc71')
    ax.bar(x, other, width, bottom=[cfr[i] + por[i] for i in range(len(conditions))],
           label='Other/Hallucination', color='#95a5a6')

    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Model Behavior Under Knowledge Conflict', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {save_path}")


def compute_significance(results: dict) -> dict:
    """Compute statistical significance between conditions."""

    baseline = [1 if r['correct'] else 0 for r in results['no_conflict']]
    hop1 = [1 if r['correct'] else 0 for r in results['conflict_hop1']]
    hop2 = [1 if r['correct'] else 0 for r in results['conflict_hop2']]

    sig_results = {}

    # Baseline vs Hop1
    chi2, p_val = stats.chisquare([sum(baseline), sum(hop1)])
    sig_results['baseline_vs_hop1'] = {'chi2': float(chi2), 'p_value': float(p_val)}

    # Baseline vs Hop2
    chi2, p_val = stats.chisquare([sum(baseline), sum(hop2)])
    sig_results['baseline_vs_hop2'] = {'chi2': float(chi2), 'p_value': float(p_val)}

    # Hop1 vs Hop2
    chi2, p_val = stats.chisquare([sum(hop1), sum(hop2)])
    sig_results['hop1_vs_hop2'] = {'chi2': float(chi2), 'p_value': float(p_val)}

    return sig_results


def create_results_table(metrics: dict, sig: dict) -> str:
    """Create markdown table for presentation."""

    table = """
| Condition | N | Accuracy | CFR | POR | vs Baseline p |
|-----------|---|----------|-----|-----|---------------|
"""

    for cond, label in [('no_conflict', 'No Conflict'),
                        ('conflict_hop1', 'Conflict @ Hop 1'),
                        ('conflict_hop2', 'Conflict @ Hop 2')]:
        m = metrics[cond]

        if cond == 'no_conflict':
            p_str = "-"
        else:
            p_key = f"baseline_vs_{cond.replace('conflict_', '')}"
            p_val = sig.get(p_key, {}).get('p_value', 1.0)
            p_str = f"{p_val:.3f}" if p_val >= 0.001 else "<0.001"

        cfr = f"{m['context_following_rate']:.1%}" if m['context_following_rate'] > 0 else "-"
        por = f"{m['parametric_override_rate']:.1%}" if m['parametric_override_rate'] > 0 else "-"

        table += f"| {label} | {m['n']} | {m['accuracy']:.1%} | {cfr} | {por} | {p_str} |\n"

    return table


def generate_all_figures(results_path: str):
    """Generate all figures from results file."""

    data = load_results(results_path)
    metrics = data['metrics']
    results = data['raw_results']

    os.makedirs('outputs/figures', exist_ok=True)

    plot_accuracy_comparison(metrics)
    plot_behavior_breakdown(metrics)

    sig = compute_significance(results)

    table = create_results_table(metrics, sig)

    print("\n=== RESULTS TABLE (for presentation) ===")
    print(table)

    with open('outputs/figures/results_table.md', 'w') as f:
        f.write(table)

    print("All figures generated!")

    return metrics, sig


if __name__ == "__main__":
    results_files = glob.glob('outputs/results/experiment_*.json')

    if results_files:
        latest = sorted(results_files)[-1]
        print(f"Loading: {latest}")
        generate_all_figures(latest)
    else:
        print("No results files found. Run experiment first.")
