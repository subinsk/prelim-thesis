"""
Multi-model conflict experiment runner.

For each model in MODELS list:
  - If outputs/results/<model>/experiment.json exists → skip (already done)
  - If not → run the experiment and save results there

Then generates comparison outputs from all available model results.
"""

import json
import os
import sys
import numpy as np
from tqdm import tqdm
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.hotpotqa_loader import HotpotQALoader
from src.data.conflict_injector import ConflictInjector
from src.inference.groq_client import GroqClient
from src.inference.prompt_templates import create_cot_prompt, extract_answer
from src.evaluation.metrics import check_answer

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# ---- CONFIG: Add models here to include them ----
MODELS = [
    {"id": "llama-3.3-70b-versatile", "label": "Llama-3.3-70B"},
    {"id": "llama-3.1-8b-instant",    "label": "Llama-3.1-8B"},
]

N_EXAMPLES = 100
RESULTS_BASE = "outputs/results"


def result_path(model_id):
    return os.path.join(RESULTS_BASE, model_id, "experiment.json")


def checkpoint_path(model_id):
    return os.path.join(RESULTS_BASE, model_id, "checkpoint.json")


def has_results(model_id):
    path = result_path(model_id)
    if not os.path.exists(path):
        return False
    with open(path, 'r') as f:
        data = json.load(f)
    n = data.get('metrics', {}).get('no_conflict', {}).get('n', 0)
    return n >= N_EXAMPLES


def save_checkpoint(model_id, results, completed_idx):
    os.makedirs(os.path.join(RESULTS_BASE, model_id), exist_ok=True)
    with open(checkpoint_path(model_id), 'w') as f:
        json.dump({'completed_idx': completed_idx, 'raw_results': results}, f, indent=2)
    print(f"  Checkpoint saved ({completed_idx + 1} examples done)")


def load_checkpoint(model_id):
    cp = checkpoint_path(model_id)
    if os.path.exists(cp):
        with open(cp, 'r') as f:
            data = json.load(f)
        print(f"  Resuming from checkpoint ({data['completed_idx'] + 1} already done)")
        return data['raw_results'], data['completed_idx']
    return None, -1


def compute_metrics(results):
    metrics = {}
    for cond in ['no_conflict', 'conflict_hop1', 'conflict_hop2']:
        n = len(results[cond])
        if n == 0:
            continue
        accuracy = sum(r['correct'] for r in results[cond]) / n
        if 'conflict' in cond:
            cfr = sum(r['followed_context'] for r in results[cond]) / n
            por = sum(r['used_parametric'] for r in results[cond]) / n
        else:
            cfr = por = 0
        metrics[cond] = {
            'n': n,
            'accuracy': accuracy,
            'context_following_rate': cfr,
            'parametric_override_rate': por
        }
    return metrics


def run_experiment_for_model(model_id, loader, examples):
    """Run the 3-condition experiment for a single model."""

    print(f"\n{'='*60}")
    print(f"RUNNING: {model_id}")
    print(f"{'='*60}")

    injector = ConflictInjector()
    client = GroqClient(model=model_id)

    results, start_after = load_checkpoint(model_id)
    if results is None:
        results = {'no_conflict': [], 'conflict_hop1': [], 'conflict_hop2': []}

    try:
        for i, example in enumerate(tqdm(examples, desc=model_id)):
            if i <= start_after:
                continue

            question, doc1, doc2, answer = loader.extract_supporting_facts(example)
            if not doc1 or not doc2:
                continue

            # Condition 1: No Conflict
            prompt = create_cot_prompt(question, doc1, doc2)
            response = client.generate(prompt)
            pred = extract_answer(response)
            result = check_answer(pred, answer)
            result['condition'] = 'no_conflict'
            result['question'] = question
            results['no_conflict'].append(result)

            # Condition 2: Conflict at Hop 1
            mod_doc1, mod_doc2, fake = injector.inject_conflict(
                question, doc1, doc2, answer, conflict_hop=1
            )
            prompt = create_cot_prompt(question, mod_doc1, mod_doc2)
            response = client.generate(prompt)
            pred = extract_answer(response)
            result = check_answer(pred, answer, fake)
            result['condition'] = 'conflict_hop1'
            result['question'] = question
            results['conflict_hop1'].append(result)

            # Condition 3: Conflict at Hop 2
            mod_doc1, mod_doc2, fake = injector.inject_conflict(
                question, doc1, doc2, answer, conflict_hop=2
            )
            prompt = create_cot_prompt(question, mod_doc1, mod_doc2)
            response = client.generate(prompt)
            pred = extract_answer(response)
            result = check_answer(pred, answer, fake)
            result['condition'] = 'conflict_hop2'
            result['question'] = question
            results['conflict_hop2'].append(result)

            if (i + 1) % 10 == 0:
                save_checkpoint(model_id, results, i)

            if (i + 1) % 20 == 0:
                print(f"\n--- Progress: {i+1}/{len(examples)} ---")
                for cond in ['no_conflict', 'conflict_hop1', 'conflict_hop2']:
                    if results[cond]:
                        acc = sum(r['correct'] for r in results[cond]) / len(results[cond])
                        print(f"  {cond}: {acc:.1%}")

    except (RuntimeError, KeyboardInterrupt) as e:
        print(f"\n\nStopped: {e}")
        save_checkpoint(model_id, results, i - 1)
        return None

    # Save final results
    metrics = compute_metrics(results)
    out = result_path(model_id)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump({'metrics': metrics, 'raw_results': results}, f, indent=2)
    print(f"Results saved to {out}")

    # Clean up checkpoint
    cp = checkpoint_path(model_id)
    if os.path.exists(cp):
        os.remove(cp)

    return metrics


def generate_comparison():
    """Load all model results and generate comparison outputs."""

    print(f"\n{'='*60}")
    print("GENERATING COMPARISON")
    print(f"{'='*60}")

    all_data = {}
    for model in MODELS:
        path = result_path(model['id'])
        if os.path.exists(path):
            with open(path, 'r') as f:
                all_data[model['id']] = {
                    'label': model['label'],
                    'data': json.load(f)
                }
            print(f"  Loaded: {model['label']} ({path})")
        else:
            print(f"  MISSING: {model['label']} — skipping")

    if len(all_data) < 2:
        print("Need at least 2 models for comparison. Run missing experiments first.")
        return

    conditions = ['no_conflict', 'conflict_hop1', 'conflict_hop2']
    cond_labels = {
        'no_conflict': 'No Conflict',
        'conflict_hop1': 'Conflict@Hop1',
        'conflict_hop2': 'Conflict@Hop2'
    }

    # ---- Markdown Table ----
    table = "# Model Comparison Results\n\n"
    table += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    table += "| Model | Condition | N | Accuracy | CFR | POR |\n"
    table += "|-------|-----------|---|----------|-----|-----|\n"

    for model_id, entry in all_data.items():
        metrics = entry['data']['metrics']
        for cond in conditions:
            if cond not in metrics:
                continue
            m = metrics[cond]
            cfr = f"{m['context_following_rate']:.1%}" if m['context_following_rate'] > 0 else "-"
            por = f"{m['parametric_override_rate']:.1%}" if m['parametric_override_rate'] > 0 else "-"
            table += f"| {entry['label']} | {cond_labels[cond]} | {m['n']} | {m['accuracy']:.1%} | {cfr} | {por} |\n"

    # Statistical comparison between models
    model_ids = list(all_data.keys())
    if len(model_ids) >= 2:
        m1, m2 = model_ids[0], model_ids[1]
        l1, l2 = all_data[m1]['label'], all_data[m2]['label']
        met1, met2 = all_data[m1]['data']['metrics'], all_data[m2]['data']['metrics']

        table += f"\n## Statistical Comparison: {l1} vs {l2}\n\n"
        table += f"| Condition | {l1} | {l2} | Chi2 | p-value | Significant |\n"
        table += "|-----------|---------|--------|------|---------|-------------|\n"

        for cond in conditions:
            if cond not in met1 or cond not in met2:
                continue
            a1, n1 = met1[cond]['accuracy'], met1[cond]['n']
            a2, n2 = met2[cond]['accuracy'], met2[cond]['n']
            ct = [[int(a1*n1), n1 - int(a1*n1)],
                  [int(a2*n2), n2 - int(a2*n2)]]
            chi2, p_val, _, _ = stats.chi2_contingency(ct)
            sig = "Yes" if p_val < 0.05 else "No"
            table += f"| {cond_labels[cond]} | {a1:.1%} | {a2:.1%} | {chi2:.2f} | {p_val:.4f} | {sig} |\n"

    os.makedirs(RESULTS_BASE, exist_ok=True)
    with open(os.path.join(RESULTS_BASE, 'model_comparison.md'), 'w') as f:
        f.write(table)
    print(f"\n{table}")

    # ---- Grouped Bar Chart ----
    fig, ax = plt.subplots(figsize=(12, 7))

    labels = ['No Conflict\n(Baseline)', 'Conflict at\nHop 1', 'Conflict at\nHop 2']
    x = np.arange(len(labels))
    n_models = len(all_data)
    width = 0.7 / n_models
    colors = ['#3498db', '#e67e22', '#2ecc71', '#9b59b6']

    for idx, (model_id, entry) in enumerate(all_data.items()):
        metrics = entry['data']['metrics']
        accs = [metrics[c]['accuracy'] * 100 if c in metrics else 0 for c in conditions]
        offset = (idx - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, accs, width, label=entry['label'],
                      color=colors[idx % len(colors)], edgecolor='black', linewidth=1.2)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Accuracy (%)', fontsize=13)
    ax.set_title('Model Comparison: Impact of Knowledge Conflicts', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=12, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    os.makedirs('outputs/figures', exist_ok=True)
    plt.savefig('outputs/figures/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Chart saved to outputs/figures/model_comparison.png")


if __name__ == "__main__":
    # Load data once
    loader = HotpotQALoader()
    if not os.path.exists('data/hotpotqa/dev.json'):
        loader.download()
    loader.load()
    examples = loader.get_bridge_questions(N_EXAMPLES)

    # Run only models that don't have results yet
    for model in MODELS:
        if has_results(model['id']):
            print(f"\n>> SKIP {model['label']} — results already exist")
        else:
            run_experiment_for_model(model['id'], loader, examples)

    # Generate comparison from all available results
    generate_comparison()
