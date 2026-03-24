"""
Conflict type experiment: factual vs temporal vs numerical.

Classifies HotpotQA bridge examples by answer type, then runs the standard
3-condition experiment (no_conflict, conflict@hop1, conflict@hop2) separately
for each type. This reveals whether models handle different conflict types
differently.

Key fix: uses proper doc ordering (bridge doc = hop 1, answer doc = hop 2)
and correct entity substitution (bridge entity for hop 1, final answer for hop 2).

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
        question, bridge_doc, answer_doc, answer, bridge_entity = loader.extract_supporting_facts(ex)
        atype = ConflictInjector.classify_answer_type(answer)
        typed[atype].append(ex)
    return dict(typed)


def run_typed_experiment(model_config, loader, typed_examples, n_per_type):
    """Run the 3-condition experiment for each conflict type."""

    model_id = model_config["id"]
    label = model_config["label"]
    injector = ConflictInjector(seed=42)
    client = None  # lazy init

    for ctype in CONFLICT_TYPES:
        examples = typed_examples.get(ctype, [])
        n_available = len(examples)
        n_target = min(n_per_type, n_available)

        if n_target == 0:
            print(f"\n>> SKIP {label} / {ctype} -- no examples available")
            continue

        experiment_name = f"conflict_type_{ctype}"
        runner = ExperimentRunner(
            experiment_name=experiment_name,
            model_id=model_id,
            conditions=CONDITIONS,
        )

        if runner.is_complete(n_target):
            print(f"\n>> SKIP {label} / {ctype} -- already has >= {n_target} results")
            continue

        # Lazy init client (only when needed)
        if client is None:
            client = get_client(model_config)

        def make_process_fn(cl, inj, ld):
            def process_example(idx, example):
                question, bridge_doc, answer_doc, answer, bridge_entity = ld.extract_supporting_facts(example)
                if not bridge_doc or not answer_doc:
                    return None

                out = {}

                # No Conflict
                prompt = create_cot_prompt(question, bridge_doc, answer_doc)
                response = cl.generate(prompt, max_tokens=1024)
                pred = extract_answer(response)
                result = check_answer(pred, answer)
                result['condition'] = 'no_conflict'
                result['question'] = question
                result['response'] = response
                result['answer_type'] = ConflictInjector.classify_answer_type(answer)
                out['no_conflict'] = result

                # Conflict@Hop1 (bridge entity in bridge doc)
                if bridge_entity:
                    mod_bridge, fake1, ok1 = inj.inject_conflict(
                        doc=bridge_doc, target_entity=bridge_entity,
                        question=question, hop=1,
                    )
                    if ok1:
                        prompt = create_cot_prompt(question, mod_bridge, answer_doc)
                        response = cl.generate(prompt, max_tokens=1024)
                        pred = extract_answer(response)
                        result = check_answer(pred, answer, fake1)
                        result['condition'] = 'conflict_hop1'
                        result['question'] = question
                        result['response'] = response
                        result['answer_type'] = ConflictInjector.classify_answer_type(answer)
                        result['target_entity'] = bridge_entity
                        result['injection_succeeded'] = True
                        out['conflict_hop1'] = result

                # Conflict@Hop2 (final answer in answer doc)
                mod_answer, fake2, ok2 = inj.inject_conflict(
                    doc=answer_doc, target_entity=answer,
                    question=question, hop=2,
                )
                if ok2:
                    prompt = create_cot_prompt(question, bridge_doc, mod_answer)
                    response = cl.generate(prompt, max_tokens=1024)
                    pred = extract_answer(response)
                    result = check_answer(pred, answer, fake2)
                    result['condition'] = 'conflict_hop2'
                    result['question'] = question
                    result['response'] = response
                    result['answer_type'] = ConflictInjector.classify_answer_type(answer)
                    result['target_entity'] = answer
                    result['injection_succeeded'] = True
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
        loader = HotpotQALoader()
        if not os.path.exists('data/hotpotqa/dev.json'):
            loader.download()
        loader.load()

        all_bridge = loader.get_bridge_questions(5000)
        typed_examples = split_by_answer_type(loader, all_bridge)

        print(f"\nAnswer type distribution:")
        for ctype in CONFLICT_TYPES:
            n = len(typed_examples.get(ctype, []))
            target = min(n_per_type, n)
            print(f"  {ctype}: {n} available, will use {target}")

        for model in models_to_run:
            run_typed_experiment(model, loader, typed_examples, n_per_type)

    generate_comparison()
