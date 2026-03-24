"""
Single-model conflict injection experiment (legacy runner).

This is the original single-model runner for 2-hop HotpotQA experiments.
For multi-model runs, use run_model_comparison.py instead.

Uses the global ExperimentRunner framework for checkpoint/resume.

Usage:
    python experiments/run_conflict_experiment.py
    python experiments/run_conflict_experiment.py --n 200
"""

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.hotpotqa_loader import HotpotQALoader
from src.data.conflict_injector import ConflictInjector
from src.inference.groq_client import GroqClient
from src.inference.prompt_templates import create_cot_prompt, extract_answer
from src.evaluation.metrics import check_answer
from src.experiments.framework import ExperimentRunner

CONDITIONS = ["no_conflict", "conflict_hop1", "conflict_hop2"]


def run_experiment(n_examples: int = 500):
    loader = HotpotQALoader()
    if not os.path.exists('data/hotpotqa/dev.json'):
        loader.download()
    loader.load()

    injector = ConflictInjector(seed=42)
    client = GroqClient()

    examples = loader.get_bridge_questions(n_examples)
    print(f"\nTesting {len(examples)} bridge questions")

    runner = ExperimentRunner(
        experiment_name="experiment",
        model_id="llama-3.3-70b-versatile",
        conditions=CONDITIONS,
    )

    if runner.is_complete(n_examples):
        print(f"Already have >= {n_examples} results. Skipping.")
        return

    def process_example(idx, example):
        question, bridge_doc, answer_doc, answer, bridge_entity = loader.extract_supporting_facts(example)
        if not bridge_doc or not answer_doc:
            return None

        out = {}

        # No Conflict (baseline)
        prompt = create_cot_prompt(question, bridge_doc, answer_doc)
        response = client.generate(prompt, max_tokens=1024)
        pred = extract_answer(response)
        result = check_answer(pred, answer)
        result['condition'] = 'no_conflict'
        result['question'] = question
        result['response'] = response
        out['no_conflict'] = result

        # Conflict@Hop1 (bridge entity in bridge doc)
        if bridge_entity:
            mod_bridge, fake1, ok1 = injector.inject_conflict(
                doc=bridge_doc, target_entity=bridge_entity,
                question=question, hop=1,
            )
            if ok1:
                prompt = create_cot_prompt(question, mod_bridge, answer_doc)
                response = client.generate(prompt, max_tokens=1024)
                pred = extract_answer(response)
                result = check_answer(pred, answer, fake1)
                result['condition'] = 'conflict_hop1'
                result['question'] = question
                result['response'] = response
                result['target_entity'] = bridge_entity
                result['injection_succeeded'] = True
                out['conflict_hop1'] = result

        # Conflict@Hop2 (final answer in answer doc)
        mod_answer, fake2, ok2 = injector.inject_conflict(
            doc=answer_doc, target_entity=answer,
            question=question, hop=2,
        )
        if ok2:
            prompt = create_cot_prompt(question, bridge_doc, mod_answer)
            response = client.generate(prompt, max_tokens=1024)
            pred = extract_answer(response)
            result = check_answer(pred, answer, fake2)
            result['condition'] = 'conflict_hop2'
            result['question'] = question
            result['response'] = response
            result['target_entity'] = answer
            result['injection_succeeded'] = True
            out['conflict_hop2'] = result

        return out

    runner.run(examples, process_example, n_target=n_examples, desc="Llama-3.3-70B [single]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-model conflict experiment")
    parser.add_argument('--n', type=int, default=500, help="Number of examples (default: 500)")
    args = parser.parse_args()

    run_experiment(n_examples=args.n)
