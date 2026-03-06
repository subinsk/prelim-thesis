"""
Multi-model conflict experiment runner.

Uses the global ExperimentRunner framework for fail-safe checkpoint/resume.

For each model in MODELS list:
  - If outputs/results/<model>/experiment.json exists with >= N_EXAMPLES → skip
  - If checkpoint exists → resume from where it stopped
  - Otherwise → run fresh

Then generates comparison outputs from all available model results.

Usage:
    python experiments/run_model_comparison.py                  # default 500 examples
    python experiments/run_model_comparison.py --n 100          # custom count
    python experiments/run_model_comparison.py --models llama-3.3-70b-versatile gemini-2.5-flash-lite
    python experiments/run_model_comparison.py --compare-only   # skip experiments, just compare
"""

import json
import os
import sys
import argparse
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.hotpotqa_loader import HotpotQALoader
from src.data.conflict_injector import ConflictInjector
from src.inference.prompt_templates import create_cot_prompt, extract_answer
from src.evaluation.metrics import check_answer
from src.experiments.framework import ExperimentRunner

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# ---- CONFIG: Add models here to include them ----
MODELS = [
    {"id": "llama-3.3-70b-versatile", "label": "Llama-3.3-70B",   "backend": "groq"},
    {"id": "llama-3.1-8b-instant",    "label": "Llama-3.1-8B",    "backend": "groq"},
    {"id": "gemini-2.5-flash-lite",    "label": "Gemini-2.5-Flash-Lite", "backend": "gemini"},
    {"id": "qwen/qwen3-32b",             "label": "Qwen3-32B",             "backend": "groq"},
]

N_EXAMPLES = 500
RESULTS_BASE = "outputs/results"
CONDITIONS = ["no_conflict", "conflict_hop1", "conflict_hop2"]


def get_client(model_config):
    """Create the appropriate client for a model's backend."""
    backend = model_config.get("backend", "groq")
    model_id = model_config["id"]

    if backend == "groq":
        from src.inference.groq_client import GroqClient
        return GroqClient(model=model_id)
    elif backend == "gemini":
        from src.inference.gemini_client import GeminiClient
        return GeminiClient(model=model_id)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def run_experiment_for_model(model_config, loader, examples, n_target):
    """Run the 3-condition experiment for a single model using the framework."""

    model_id = model_config["id"]
    label = model_config["label"]

    runner = ExperimentRunner(
        experiment_name="experiment",
        model_id=model_id,
        conditions=CONDITIONS,
    )

    if runner.is_complete(n_target):
        print(f"\n>> SKIP {label} — already has >= {n_target} results")
        return None

    # Create client and injector
    injector = ConflictInjector()
    client = get_client(model_config)

    def process_example(idx, example):
        """Process a single example through all 3 conditions."""
        question, doc1, doc2, answer = loader.extract_supporting_facts(example)
        if not doc1 or not doc2:
            return None  # skip

        out = {}

        # Condition 1: No Conflict (baseline)
        prompt = create_cot_prompt(question, doc1, doc2)
        response = client.generate(prompt, max_tokens=1024)
        pred = extract_answer(response)
        result = check_answer(pred, answer)
        result['condition'] = 'no_conflict'
        result['question'] = question
        out['no_conflict'] = result

        # Condition 2: Conflict at Hop 1
        mod_doc1, mod_doc2, fake = injector.inject_conflict(
            question, doc1, doc2, answer, conflict_hop=1
        )
        prompt = create_cot_prompt(question, mod_doc1, mod_doc2)
        response = client.generate(prompt, max_tokens=1024)
        pred = extract_answer(response)
        result = check_answer(pred, answer, fake)
        result['condition'] = 'conflict_hop1'
        result['question'] = question
        out['conflict_hop1'] = result

        # Condition 3: Conflict at Hop 2
        mod_doc1, mod_doc2, fake = injector.inject_conflict(
            question, doc1, doc2, answer, conflict_hop=2
        )
        prompt = create_cot_prompt(question, mod_doc1, mod_doc2)
        response = client.generate(prompt, max_tokens=1024)
        pred = extract_answer(response)
        result = check_answer(pred, answer, fake)
        result['condition'] = 'conflict_hop2'
        result['question'] = question
        out['conflict_hop2'] = result

        return out

    return runner.run(examples, process_example, n_target=n_target, desc=label)


def generate_comparison():
    """Load all model results and generate comparison outputs."""

    print(f"\n{'='*60}")
    print("GENERATING COMPARISON")
    print(f"{'='*60}")

    all_data = {}
    for model in MODELS:
        path = os.path.join(RESULTS_BASE, model['id'], "experiment.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                all_data[model['id']] = {
                    'label': model['label'],
                    'data': json.load(f)
                }
            n = all_data[model['id']]['data']['metrics'].get('no_conflict', {}).get('n', 0)
            print(f"  Loaded: {model['label']} ({n} examples)")
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

    # Statistical pairwise comparisons between all model pairs
    model_ids = list(all_data.keys())
    for i in range(len(model_ids)):
        for j in range(i + 1, len(model_ids)):
            m1, m2 = model_ids[i], model_ids[j]
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
                try:
                    chi2, p_val, _, _ = stats.chi2_contingency(ct)
                except ValueError:
                    chi2, p_val = 0, 1.0
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
    colors = ['#3498db', '#e67e22', '#2ecc71', '#9b59b6', '#e74c3c']

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
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Accuracy (%)', fontsize=13)
    ax.set_title('Model Comparison: Impact of Knowledge Conflicts', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=11, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    os.makedirs('outputs/figures', exist_ok=True)
    plt.savefig('outputs/figures/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Chart saved to outputs/figures/model_comparison.png")


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-model conflict experiment runner")
    parser.add_argument('--n', type=int, default=N_EXAMPLES,
                        help=f"Number of examples per model (default: {N_EXAMPLES})")
    parser.add_argument('--models', nargs='+', default=None,
                        help="Specific model IDs to run (default: all)")
    parser.add_argument('--compare-only', action='store_true',
                        help="Skip experiments, just regenerate comparison")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    n_examples = args.n

    # Filter models if specific ones requested
    models_to_run = MODELS
    if args.models:
        models_to_run = [m for m in MODELS if m['id'] in args.models]
        if not models_to_run:
            print(f"No matching models found. Available: {[m['id'] for m in MODELS]}")
            sys.exit(1)

    if not args.compare_only:
        # Load data once
        loader = HotpotQALoader()
        if not os.path.exists('data/hotpotqa/dev.json'):
            loader.download()
        loader.load()
        examples = loader.get_bridge_questions(n_examples)
        print(f"\nUsing {len(examples)} bridge questions")

        # Run each model
        for model in models_to_run:
            run_experiment_for_model(model, loader, examples, n_examples)

    # Generate comparison from all available results
    generate_comparison()
