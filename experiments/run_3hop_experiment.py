"""
3-hop conflict experiment runner using MuSiQue dataset.

Uses the global ExperimentRunner framework for fail-safe checkpoint/resume.
Tests conflict injection at each hop position (1, 2, 3) in 3-hop reasoning chains.

Usage:
    python experiments/run_3hop_experiment.py                    # default settings
    python experiments/run_3hop_experiment.py --n 200            # custom count
    python experiments/run_3hop_experiment.py --models llama-3.3-70b-versatile
"""

import json
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.musique_loader import MuSiQueLoader
from src.data.conflict_injector import ConflictInjector
from src.inference.prompt_templates import extract_answer
from src.evaluation.metrics import check_answer
from src.experiments.framework import ExperimentRunner

MODELS = [
    {"id": "llama-3.3-70b-versatile", "label": "Llama-3.3-70B",   "backend": "groq"},
    {"id": "llama-3.1-8b-instant",    "label": "Llama-3.1-8B",    "backend": "groq"},
    {"id": "gemini-2.5-flash-lite",    "label": "Gemini-2.5-Flash-Lite", "backend": "gemini"},
    {"id": "qwen/qwen3-32b",             "label": "Qwen3-32B",             "backend": "groq"},
]

N_EXAMPLES = 200
CONDITIONS = ["no_conflict", "conflict_hop1", "conflict_hop2", "conflict_hop3"]


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


def create_3hop_cot_prompt(question: str, doc1: str, doc2: str, doc3: str) -> str:
    """Create Chain-of-Thought prompt for 3-hop multi-hop QA."""
    return f"""Answer the following question using the provided documents. Think step by step.

Document 1:
{doc1}

Document 2:
{doc2}

Document 3:
{doc3}

Question: {question}

Let's solve this step by step:
1. First, I'll identify relevant information from Document 1.
2. Then, I'll connect that with information from Document 2.
3. Finally, I'll use Documents 2 and 3 to find the answer.

Step-by-step reasoning:"""


def run_3hop_for_model(model_config, loader, examples, n_target):
    """Run the 4-condition experiment for 3-hop questions using the framework."""

    model_id = model_config["id"]
    label = model_config["label"]

    runner = ExperimentRunner(
        experiment_name="musique_experiment",
        model_id=model_id,
        conditions=CONDITIONS,
    )

    if runner.is_complete(n_target):
        print(f"\n>> SKIP {label} — already has >= {n_target} 3-hop results")
        return None

    injector = ConflictInjector()
    client = get_client(model_config)

    def process_example(idx, example):
        """Process a single 3-hop example through all 4 conditions."""
        question, docs, answer = loader.extract_supporting_docs(example)

        if len(docs) < 3:
            return None  # skip — need 3 docs

        doc1, doc2, doc3 = docs[0], docs[1], docs[2]
        out = {}

        # Condition 1: No Conflict (baseline)
        prompt = create_3hop_cot_prompt(question, doc1, doc2, doc3)
        response = client.generate(prompt, max_tokens=1024)
        pred = extract_answer(response)
        result = check_answer(pred, answer)
        result['condition'] = 'no_conflict'
        result['question'] = question
        out['no_conflict'] = result

        # Conditions 2-4: Conflict at each hop
        for hop in [1, 2, 3]:
            fake_answer = injector._generate_fake_answer(answer)
            modified_docs = [doc1, doc2, doc3]
            modified_docs[hop - 1] = injector._substitute_entity(
                modified_docs[hop - 1], answer, fake_answer
            )
            prompt = create_3hop_cot_prompt(question, modified_docs[0], modified_docs[1], modified_docs[2])
            response = client.generate(prompt, max_tokens=1024)
            pred = extract_answer(response)
            result = check_answer(pred, answer, fake_answer)
            result['condition'] = f'conflict_hop{hop}'
            result['question'] = question
            out[f'conflict_hop{hop}'] = result

        return out

    return runner.run(examples, process_example, n_target=n_target, desc=f"{label} [3-hop]")


def parse_args():
    parser = argparse.ArgumentParser(description="3-hop conflict experiment (MuSiQue)")
    parser.add_argument('--n', type=int, default=N_EXAMPLES,
                        help=f"Number of examples (default: {N_EXAMPLES})")
    parser.add_argument('--models', nargs='+', default=None,
                        help="Specific model IDs to run (default: all)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Filter models if specified
    models_to_run = MODELS
    if args.models:
        models_to_run = [m for m in MODELS if m['id'] in args.models]
        if not models_to_run:
            print(f"No matching models. Available: {[m['id'] for m in MODELS]}")
            sys.exit(1)

    # Load MuSiQue
    loader = MuSiQueLoader()
    data_path = 'data/musique/validation.jsonl'
    if not os.path.exists(data_path):
        print("Downloading MuSiQue dataset...")
        loader.download()
    loader.load(data_path)

    # Get 3-hop questions
    examples = loader.get_questions_by_hops(3, n=args.n)

    if not examples:
        print("No 3-hop questions found. Check dataset.")
        sys.exit(1)

    # Run for each model
    for model in models_to_run:
        run_3hop_for_model(model, loader, examples, args.n)
