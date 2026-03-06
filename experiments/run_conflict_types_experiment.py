"""
Conflict type experiment: factual vs temporal vs numerical.

Classifies HotpotQA bridge examples by answer type, then runs the standard
3-condition experiment (no_conflict, conflict@hop1, conflict@hop2) separately
for each type. This reveals whether models handle different conflict types
differently.

Uses the global ExperimentRunner framework for checkpoint/resume.

Usage:
    python experiments/run_conflict_types_experiment.py
    python experiments/run_conflict_types_experiment.py --n 200
    python experiments/run_conflict_types_experiment.py --models llama-3.3-70b-versatile
    python experiments/run_conflict_types_experiment.py --compare-only
"""

import json
import os
import sys
import argparse
import numpy as np
from datetime import datetime
from collections import defaultdict

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

# ---- CONFIG ----
MODELS = [
    {"id": "llama-3.3-70b-versatile", "label": "Llama-3.3-70B", "backend": "groq"},
    {"id": "llama-3.1-8b-instant",    "label": "Llama-3.1-8B",  "backend": "groq"},
    {"id": "gemini-2.5-flash-lite",    "label": "Gemini-2.5-Flash-Lite", "backend": "gemini"},
    {"id": "qwen/qwen3-32b",          "label": "Qwen3-32B",     "backend": "groq"},
]

N_EXAMPLES_PER_TYPE = 200  # target per conflict type
CONDITIONS = ["no_conflict", "conflict_hop1", "conflict_hop2"]
CONFLICT_TYPES = [ConflictInjector.TYPE_FACTUAL, ConflictInjector.TYPE_TEMPORAL, ConflictInjector.TYPE_NUMERICAL]
RESULTS_BASE = "outputs/results"


def get_client(model_config):
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


def split_by_answer_type(loader, examples):
    """Split examples by answer type and return dict of {type: [examples]}."""
    typed = defaultdict(list)
    for ex in examples:
        _, _, _, answer = loader.extract_supporting_facts(ex)
        atype = ConflictInjector.classify_answer_type(answer)
        typed[atype].append(ex)
    return dict(typed)


def run_typed_experiment(model_config, loader, typed_examples, n_per_type):
    """Run the 3-condition experiment for each conflict type."""

    model_id = model_config["id"]
    label = model_config["label"]
    injector = ConflictInjector()
    client = None  # lazy init

    for ctype in CONFLICT_TYPES:
        examples = typed_examples.get(ctype, [])
        n_available = len(examples)
        n_target = min(n_per_type, n_available)

        if n_target == 0:
            print(f"\n>> SKIP {label} / {ctype} — no examples available")
            continue

        experiment_name = f"conflict_type_{ctype}"
        runner = ExperimentRunner(
            experiment_name=experiment_name,
            model_id=model_id,
            conditions=CONDITIONS,
        )

        if runner.is_complete(n_target):
            print(f"\n>> SKIP {label} / {ctype} — already has >= {n_target} results")
            continue

        # Lazy init client (only when needed)
        if client is None:
            client = get_client(model_config)

        def make_process_fn(cl, inj, ld):
            def process_example(idx, example):
                question, doc1, doc2, answer = ld.extract_supporting_facts(example)
                if not doc1 or not doc2:
                    return None

                out = {}

                # No Conflict
                prompt = create_cot_prompt(question, doc1, doc2)
                response = cl.generate(prompt, max_tokens=1024)
                pred = extract_answer(response)
                result = check_answer(pred, answer)
                result['condition'] = 'no_conflict'
                result['question'] = question
                result['answer_type'] = ConflictInjector.classify_answer_type(answer)
                out['no_conflict'] = result

                # Conflict@Hop1
                mod_doc1, mod_doc2, fake = inj.inject_conflict(question, doc1, doc2, answer, conflict_hop=1)
                prompt = create_cot_prompt(question, mod_doc1, mod_doc2)
                response = cl.generate(prompt, max_tokens=1024)
                pred = extract_answer(response)
                result = check_answer(pred, answer, fake)
                result['condition'] = 'conflict_hop1'
                result['question'] = question
                result['answer_type'] = ConflictInjector.classify_answer_type(answer)
                out['conflict_hop1'] = result

                # Conflict@Hop2
                mod_doc1, mod_doc2, fake = inj.inject_conflict(question, doc1, doc2, answer, conflict_hop=2)
                prompt = create_cot_prompt(question, mod_doc1, mod_doc2)
                response = cl.generate(prompt, max_tokens=1024)
                pred = extract_answer(response)
                result = check_answer(pred, answer, fake)
                result['condition'] = 'conflict_hop2'
                result['question'] = question
                result['answer_type'] = ConflictInjector.classify_answer_type(answer)
                out['conflict_hop2'] = result

                return out
            return process_example

        process_fn = make_process_fn(client, injector, loader)
        desc = f"{label} [{ctype}]"
        print(f"\n>> Running {label} / {ctype}: {n_target} examples (of {n_available} available)")
        runner.run(examples[:n_target], process_fn, n_target=n_target, desc=desc)


def generate_comparison():
    """Load all conflict type results and generate comparison outputs."""

    print(f"\n{'='*60}")
    print("GENERATING CONFLICT TYPE COMPARISON")
    print(f"{'='*60}")

    # Collect all results: {model_id: {ctype: metrics}}
    all_data = {}
    for model in MODELS:
        model_id = model['id']
        model_data = {}
        for ctype in CONFLICT_TYPES:
            path = os.path.join(RESULTS_BASE, model_id, f"conflict_type_{ctype}.json")
            if os.path.exists(path):
                with open(path, 'r') as f:
                    data = json.load(f)
                model_data[ctype] = data['metrics']
                n = data['metrics'].get('no_conflict', {}).get('n', 0)
                print(f"  Loaded: {model['label']} / {ctype} ({n} examples)")
        if model_data:
            all_data[model_id] = {'label': model['label'], 'types': model_data}

    if not all_data:
        print("No conflict type results found. Run experiments first.")
        return

    # ---- Markdown Table ----
    cond_labels = {
        'no_conflict': 'No Conflict',
        'conflict_hop1': 'Conflict@Hop1',
        'conflict_hop2': 'Conflict@Hop2'
    }

    table = "# Conflict Type Comparison Results\n\n"
    table += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    table += "## Accuracy by Conflict Type and Condition\n\n"
    table += "| Model | Conflict Type | Condition | N | Accuracy | CFR | POR |\n"
    table += "|-------|---------------|-----------|---|----------|-----|-----|\n"

    for model_id, entry in all_data.items():
        for ctype in CONFLICT_TYPES:
            if ctype not in entry['types']:
                continue
            metrics = entry['types'][ctype]
            for cond in CONDITIONS:
                if cond not in metrics:
                    continue
                m = metrics[cond]
                cfr = f"{m['context_following_rate']:.1%}" if m['context_following_rate'] > 0 else "-"
                por = f"{m['parametric_override_rate']:.1%}" if m['parametric_override_rate'] > 0 else "-"
                table += f"| {entry['label']} | {ctype.capitalize()} | {cond_labels[cond]} | {m['n']} | {m['accuracy']:.1%} | {cfr} | {por} |\n"

    # Summary: accuracy drop by type
    table += "\n## Summary: Accuracy Drop by Conflict Type\n\n"
    table += "| Model | Conflict Type | Baseline | Avg Conflict Acc | Drop (pp) |\n"
    table += "|-------|---------------|----------|------------------|-----------|\n"

    for model_id, entry in all_data.items():
        for ctype in CONFLICT_TYPES:
            if ctype not in entry['types']:
                continue
            metrics = entry['types'][ctype]
            baseline = metrics.get('no_conflict', {}).get('accuracy', 0)
            conflict_accs = []
            for cond in ['conflict_hop1', 'conflict_hop2']:
                if cond in metrics:
                    conflict_accs.append(metrics[cond]['accuracy'])
            if conflict_accs:
                avg_conflict = np.mean(conflict_accs)
                drop = (baseline - avg_conflict) * 100
                table += f"| {entry['label']} | {ctype.capitalize()} | {baseline:.1%} | {avg_conflict:.1%} | {drop:.1f} |\n"

    os.makedirs(RESULTS_BASE, exist_ok=True)
    with open(os.path.join(RESULTS_BASE, 'conflict_types_comparison.md'), 'w') as f:
        f.write(table)
    print(f"\n{table}")

    # ---- Grouped Bar Chart: Accuracy by Conflict Type ----
    fig, axes = plt.subplots(1, len(all_data), figsize=(7 * len(all_data), 7), squeeze=False)

    colors = {'factual': '#3498db', 'temporal': '#e67e22', 'numerical': '#2ecc71'}

    for ax_idx, (model_id, entry) in enumerate(all_data.items()):
        ax = axes[0][ax_idx]
        x = np.arange(len(CONDITIONS))
        width = 0.25
        type_count = 0

        for ctype in CONFLICT_TYPES:
            if ctype not in entry['types']:
                continue
            metrics = entry['types'][ctype]
            accs = [metrics.get(c, {}).get('accuracy', 0) * 100 for c in CONDITIONS]
            n = metrics.get('no_conflict', {}).get('n', 0)
            offset = (type_count - 1) * width
            bars = ax.bar(x + offset, accs, width, label=f"{ctype.capitalize()} (n={n})",
                          color=colors.get(ctype, '#999'), edgecolor='black', linewidth=1)
            for bar in bars:
                h = bar.get_height()
                ax.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, fontweight='bold')
            type_count += 1

        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title(entry['label'], fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['No Conflict', 'Conflict\n@Hop1', 'Conflict\n@Hop2'], fontsize=10)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Impact of Knowledge Conflicts by Conflict Type', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    os.makedirs('outputs/figures', exist_ok=True)
    plt.savefig('outputs/figures/conflict_types_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Chart saved to outputs/figures/conflict_types_comparison.png")


def parse_args():
    parser = argparse.ArgumentParser(description="Conflict type experiment runner")
    parser.add_argument('--n', type=int, default=N_EXAMPLES_PER_TYPE,
                        help=f"Target examples per conflict type (default: {N_EXAMPLES_PER_TYPE})")
    parser.add_argument('--models', nargs='+', default=None,
                        help="Specific model IDs to run (default: all)")
    parser.add_argument('--compare-only', action='store_true',
                        help="Skip experiments, just regenerate comparison")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    n_per_type = args.n

    models_to_run = MODELS
    if args.models:
        models_to_run = [m for m in MODELS if m['id'] in args.models]
        if not models_to_run:
            print(f"No matching models found. Available: {[m['id'] for m in MODELS]}")
            sys.exit(1)

    if not args.compare_only:
        # Load data
        loader = HotpotQALoader()
        if not os.path.exists('data/hotpotqa/dev.json'):
            loader.download()
        loader.load()

        # Get enough bridge examples to fill all types
        # Use a large pool since numerical is rare (~3%)
        all_bridge = loader.get_bridge_questions(5000)
        typed_examples = split_by_answer_type(loader, all_bridge)

        print(f"\nAnswer type distribution:")
        for ctype in CONFLICT_TYPES:
            n = len(typed_examples.get(ctype, []))
            target = min(n_per_type, n)
            print(f"  {ctype}: {n} available, will use {target}")

        # Run each model
        for model in models_to_run:
            run_typed_experiment(model, loader, typed_examples, n_per_type)

    # Generate comparison
    generate_comparison()
