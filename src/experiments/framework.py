"""
Global Experiment Framework — Fail-Safe Checkpoint/Resume System

All experiment runners MUST use this framework. It guarantees:
  1. Auto-checkpoint every N examples (default: 10)
  2. Resume from last checkpoint on restart
  3. Graceful stop on rate-limit / daily-limit / Ctrl+C
  4. Final results saved atomically (write temp → rename)
  5. Checkpoints are NEVER deleted (kept for audit trail)
  6. Per-example error handling (skip bad examples, don't crash)
  7. Run history logged to run_history.json (status, timestamps, counts)

Usage:
    from src.experiments.framework import ExperimentRunner

    runner = ExperimentRunner(
        experiment_name="experiment",
        model_id="llama-3.3-70b-versatile",
        conditions=["no_conflict", "conflict_hop1", "conflict_hop2"],
    )

    if runner.is_complete(min_examples=500):
        print("Already done")
    else:
        runner.run(examples, process_fn, n_target=500)

    # process_fn(example, client, injector) -> dict of {condition: result_dict}
"""

import json
import os
import sys
import time
from datetime import datetime
from tqdm import tqdm


class ExperimentRunner:
    """Fail-safe experiment runner with checkpoint/resume."""

    RESULTS_BASE = "outputs/results"

    def __init__(
        self,
        experiment_name: str,
        model_id: str,
        conditions: list,
        checkpoint_every: int = 10,
        progress_every: int = 50,
    ):
        self.experiment_name = experiment_name
        self.model_id = model_id
        self.conditions = conditions
        self.checkpoint_every = checkpoint_every
        self.progress_every = progress_every

    # ---- Paths ----

    @property
    def result_dir(self):
        return os.path.join(self.RESULTS_BASE, self.model_id)

    @property
    def result_file(self):
        return os.path.join(self.result_dir, f"{self.experiment_name}.json")

    @property
    def checkpoint_file(self):
        return os.path.join(self.result_dir, f"{self.experiment_name}_checkpoint.json")

    # ---- Check state ----

    def is_complete(self, min_examples: int) -> bool:
        """Check if we already have enough results."""
        if not os.path.exists(self.result_file):
            return False
        try:
            with open(self.result_file, 'r') as f:
                data = json.load(f)
            first_cond = self.conditions[0]
            n = data.get('metrics', {}).get(first_cond, {}).get('n', 0)
            return n >= min_examples
        except (json.JSONDecodeError, KeyError):
            return False

    def get_existing_count(self) -> int:
        """Get number of completed examples in existing results."""
        if not os.path.exists(self.result_file):
            return 0
        try:
            with open(self.result_file, 'r') as f:
                data = json.load(f)
            first_cond = self.conditions[0]
            return data.get('metrics', {}).get(first_cond, {}).get('n', 0)
        except (json.JSONDecodeError, KeyError):
            return 0

    # ---- Checkpoint ----

    def save_checkpoint(self, results: dict, completed_idx: int):
        """Save partial results to checkpoint file."""
        os.makedirs(self.result_dir, exist_ok=True)
        tmp = self.checkpoint_file + ".tmp"
        with open(tmp, 'w') as f:
            json.dump({
                'completed_idx': completed_idx,
                'raw_results': results,
                'timestamp': datetime.now().isoformat(),
                'model_id': self.model_id,
                'experiment': self.experiment_name,
            }, f, indent=2)
        # Atomic rename
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
        os.rename(tmp, self.checkpoint_file)

        n = len(results.get(self.conditions[0], []))
        print(f"  [Checkpoint] {n} examples saved (index {completed_idx})")

    def load_checkpoint(self):
        """Load checkpoint if exists. Returns (results_dict, last_completed_idx)."""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                n = len(data['raw_results'].get(self.conditions[0], []))
                print(f"  [Resume] Found checkpoint with {n} examples (index {data['completed_idx']})")
                return data['raw_results'], data['completed_idx']
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  [Warning] Corrupt checkpoint, starting fresh: {e}")
        return None, -1

    # ---- Save final results ----

    def save_results(self, results: dict, metrics: dict):
        """Save final results atomically."""
        os.makedirs(self.result_dir, exist_ok=True)
        payload = {
            'metrics': metrics,
            'raw_results': results,
            'metadata': {
                'model_id': self.model_id,
                'experiment': self.experiment_name,
                'timestamp': datetime.now().isoformat(),
                'conditions': self.conditions,
            }
        }
        tmp = self.result_file + ".tmp"
        with open(tmp, 'w') as f:
            json.dump(payload, f, indent=2)
        if os.path.exists(self.result_file):
            os.remove(self.result_file)
        os.rename(tmp, self.result_file)
        print(f"  [Saved] {self.result_file}")

    @property
    def history_file(self):
        return os.path.join(self.result_dir, f"{self.experiment_name}_run_history.json")

    def log_run(self, status: str, n_examples: int, start_time: str, end_time: str, details: str = ""):
        """Append a run entry to run_history.json.

        status: 'completed', 'partial', 'aborted' (Ctrl+C), 'error'
        """
        os.makedirs(self.result_dir, exist_ok=True)
        history = []
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    history = json.load(f)
            except (json.JSONDecodeError, KeyError):
                history = []

        history.append({
            'run_number': len(history) + 1,
            'status': status,
            'n_examples': n_examples,
            'start_time': start_time,
            'end_time': end_time,
            'duration_seconds': round((datetime.fromisoformat(end_time) - datetime.fromisoformat(start_time)).total_seconds(), 1),
            'details': details,
            'model_id': self.model_id,
            'experiment': self.experiment_name,
        })

        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"  [History] Run #{len(history)} logged as '{status}'")

    # ---- Compute metrics ----

    def compute_metrics(self, results: dict) -> dict:
        """Compute accuracy, CFR, POR for each condition."""
        metrics = {}
        for cond in self.conditions:
            entries = results.get(cond, [])
            n = len(entries)
            if n == 0:
                continue
            accuracy = sum(r['correct'] for r in entries) / n
            if 'conflict' in cond:
                cfr = sum(r.get('followed_context', False) for r in entries) / n
                por = sum(r.get('used_parametric', False) for r in entries) / n
            else:
                cfr = por = 0
            metrics[cond] = {
                'n': n,
                'accuracy': accuracy,
                'context_following_rate': cfr,
                'parametric_override_rate': por,
            }
        return metrics

    # ---- Main run loop ----

    def run(self, examples: list, process_fn, n_target: int = None, desc: str = None):
        """
        Run the experiment with fail-safe checkpoint/resume.

        Args:
            examples: list of raw examples to process
            process_fn: callable(example_idx, example) -> dict {condition: result_dict}
                        Must return a dict mapping condition names to result dicts.
                        Each result dict must have at minimum: 'correct', 'predicted', 'gold'.
                        For conflict conditions: 'followed_context', 'used_parametric', 'fake'.
                        If process_fn returns None, the example is skipped.
            n_target: target number of examples (for progress display)
            desc: tqdm description override
        """
        if desc is None:
            desc = f"{self.model_id} [{self.experiment_name}]"

        run_start_time = datetime.now().isoformat()

        # Load checkpoint or start fresh
        results, start_after = self.load_checkpoint()
        if results is None:
            results = {cond: [] for cond in self.conditions}

        completed = len(results.get(self.conditions[0], []))
        target = n_target or len(examples)

        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {self.experiment_name}")
        print(f"MODEL:      {self.model_id}")
        print(f"PROGRESS:   {completed}/{target} examples done")
        print(f"CHECKPOINT: every {self.checkpoint_every} examples")
        print(f"{'='*60}")

        if completed >= target:
            print(f"  Already have {completed} examples >= target {target}. Skipping.")
            metrics = self.compute_metrics(results)
            return metrics, results

        stopped_early = False
        stop_reason = "completed"
        stop_detail = ""

        try:
            for i, example in enumerate(tqdm(examples, desc=desc, initial=min(start_after+1, len(examples)))):
                if i <= start_after:
                    continue

                # Process single example with error handling
                try:
                    condition_results = process_fn(i, example)
                except Exception as e:
                    error_str = str(e)
                    # Daily limit / hard stop — save and exit
                    if "daily" in error_str.lower() or "Daily" in error_str:
                        print(f"\n  [STOP] Daily limit reached: {e}")
                        self.save_checkpoint(results, i - 1)
                        stopped_early = True
                        stop_reason = "partial"
                        stop_detail = f"Daily limit: {e}"
                        break
                    # Transient error — skip this example
                    print(f"\n  [Skip] Example {i} failed: {e}")
                    continue

                if condition_results is None:
                    continue  # process_fn signals skip

                # Append results per condition
                for cond in self.conditions:
                    if cond in condition_results:
                        results[cond].append(condition_results[cond])

                # Auto-checkpoint
                if (i + 1) % self.checkpoint_every == 0:
                    self.save_checkpoint(results, i)

                # Progress print
                if (i + 1) % self.progress_every == 0:
                    self._print_progress(results, i + 1, len(examples))

        except KeyboardInterrupt:
            print(f"\n\n  [INTERRUPTED] Saving checkpoint...")
            self.save_checkpoint(results, i - 1)
            stopped_early = True
            stop_reason = "aborted"
            stop_detail = "User pressed Ctrl+C"

        except RuntimeError as e:
            print(f"\n\n  [ERROR] {e}")
            self.save_checkpoint(results, i - 1)
            stopped_early = True
            stop_reason = "error"
            stop_detail = str(e)

        # Compute metrics
        metrics = self.compute_metrics(results)

        # Print summary
        completed = len(results.get(self.conditions[0], []))
        print(f"\n{'='*60}")
        print(f"{'PARTIAL' if stopped_early else 'COMPLETE'}: {completed} examples")
        print(f"{'='*60}")
        for cond in self.conditions:
            if cond in metrics:
                m = metrics[cond]
                parts = [f"Acc={m['accuracy']:.1%}"]
                if m['context_following_rate'] > 0:
                    parts.append(f"CFR={m['context_following_rate']:.1%}")
                if m['parametric_override_rate'] > 0:
                    parts.append(f"POR={m['parametric_override_rate']:.1%}")
                print(f"  {cond}: {' | '.join(parts)}  (n={m['n']})")

        run_end_time = datetime.now().isoformat()

        # Save final results + checkpoint (checkpoint is NEVER deleted)
        # Allow 1% tolerance for skipped examples (e.g., 499/500 counts as complete)
        close_enough = completed >= target * 0.99
        if not stopped_early and close_enough:
            self.save_results(results, metrics)
            self.save_checkpoint(results, i if 'i' in dir() else start_after)
            self.log_run("completed", completed, run_start_time, run_end_time,
                         f"Target {target} reached")
        else:
            self.save_checkpoint(results, i if 'i' in dir() else start_after)
            self.log_run(stop_reason, completed, run_start_time, run_end_time,
                         stop_detail)

        return metrics, results

    def _print_progress(self, results, current, total):
        """Print intermediate progress."""
        print(f"\n--- Progress: {current}/{total} ---")
        for cond in self.conditions:
            entries = results.get(cond, [])
            if entries:
                acc = sum(r['correct'] for r in entries) / len(entries)
                print(f"  {cond}: {acc:.1%} (n={len(entries)})")
