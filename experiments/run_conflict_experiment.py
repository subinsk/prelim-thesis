import json
import os
import sys
from tqdm import tqdm
from datetime import datetime

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.hotpotqa_loader import HotpotQALoader
from src.data.conflict_injector import ConflictInjector
from src.inference.groq_client import GroqClient
from src.inference.prompt_templates import create_cot_prompt, extract_answer
from src.evaluation.metrics import normalize_answer, check_answer

CHECKPOINT_PATH = 'outputs/results/checkpoint.json'


def save_checkpoint(results, completed_idx):
    """Save partial results to checkpoint file."""
    os.makedirs('outputs/results', exist_ok=True)
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump({
            'completed_idx': completed_idx,
            'raw_results': results
        }, f, indent=2)
    print(f"  Checkpoint saved ({completed_idx + 1} examples done)")


def load_checkpoint():
    """Load checkpoint if exists."""
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, 'r') as f:
            data = json.load(f)
        print(f"  Resuming from checkpoint ({data['completed_idx'] + 1} examples already done)")
        return data['raw_results'], data['completed_idx']
    return None, -1


def compute_metrics(results):
    """Compute final metrics from results."""
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


def run_experiment(n_examples: int = 100, save_results: bool = True):
    """
    Run conflict injection experiment with checkpoint/resume support.

    Tests three conditions:
    1. No conflict (baseline)
    2. Conflict at hop 1
    3. Conflict at hop 2
    """

    print("=" * 60)
    print("KNOWLEDGE CONFLICT EXPERIMENT")
    print("=" * 60)

    # Initialize
    loader = HotpotQALoader()
    if not os.path.exists('data/hotpotqa/dev.json'):
        loader.download()
    loader.load()

    injector = ConflictInjector()
    client = GroqClient()

    # Get bridge questions
    examples = loader.get_bridge_questions(n_examples)
    print(f"\nTesting {len(examples)} bridge questions")

    # Check for checkpoint
    results, start_after = load_checkpoint()
    if results is None:
        results = {
            'no_conflict': [],
            'conflict_hop1': [],
            'conflict_hop2': []
        }

    try:
        for i, example in enumerate(tqdm(examples, desc="Running experiments")):
            # Skip already-completed examples
            if i <= start_after:
                continue

            question, doc1, doc2, answer = loader.extract_supporting_facts(example)

            if not doc1 or not doc2:
                continue

            # === Condition 1: No Conflict (Baseline) ===
            prompt = create_cot_prompt(question, doc1, doc2)
            response = client.generate(prompt)
            pred = extract_answer(response)
            result = check_answer(pred, answer)
            result['condition'] = 'no_conflict'
            result['question'] = question
            results['no_conflict'].append(result)

            # === Condition 2: Conflict at Hop 1 ===
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

            # === Condition 3: Conflict at Hop 2 ===
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

            # Auto-save checkpoint every 10 examples
            if (i + 1) % 10 == 0:
                save_checkpoint(results, i)

            # Progress update
            if (i + 1) % 20 == 0:
                print(f"\n--- Progress: {i+1}/{len(examples)} ---")
                for cond in ['no_conflict', 'conflict_hop1', 'conflict_hop2']:
                    if results[cond]:
                        acc = sum(r['correct'] for r in results[cond]) / len(results[cond])
                        print(f"  {cond}: {acc:.1%}")

    except (RuntimeError, KeyboardInterrupt) as e:
        print(f"\n\nStopped: {e}")
        print("Saving partial results...")
        save_checkpoint(results, i - 1)

    # === Compute Final Metrics ===
    metrics = compute_metrics(results)

    print("\n" + "=" * 60)
    print(f"RESULTS ({len(results['no_conflict'])} examples completed)")
    print("=" * 60)

    for cond in ['no_conflict', 'conflict_hop1', 'conflict_hop2']:
        if cond in metrics:
            m = metrics[cond]
            print(f"\n{cond}:")
            print(f"  Accuracy: {m['accuracy']:.1%} ({int(m['accuracy']*m['n'])}/{m['n']})")
            if 'conflict' in cond:
                print(f"  Context-Following Rate: {m['context_following_rate']:.1%}")
                print(f"  Parametric-Override Rate: {m['parametric_override_rate']:.1%}")

    # Save final results
    if save_results and results['no_conflict']:
        os.makedirs('outputs/results', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outpath = f'outputs/results/experiment_{timestamp}.json'

        with open(outpath, 'w') as f:
            json.dump({
                'metrics': metrics,
                'raw_results': results
            }, f, indent=2)

        print(f"\nResults saved to {outpath}")

        # Clean up checkpoint if we finished all examples
        if len(results['no_conflict']) >= n_examples and os.path.exists(CHECKPOINT_PATH):
            os.remove(CHECKPOINT_PATH)
            print("Checkpoint cleaned up (experiment complete)")

    return metrics, results


if __name__ == "__main__":
    metrics, results = run_experiment(n_examples=100)
