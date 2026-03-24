"""
Statistical analysis utilities for experiment results.

Provides:
- 95% Confidence Intervals (binomial)
- Effect sizes (Cohen's h for proportions)
- McNemar's test (paired comparisons)
- Comprehensive results tables with all statistics
"""

import json
import math
import os
import numpy as np
from scipy import stats


def binomial_ci(p: float, n: int, confidence: float = 0.95) -> tuple:
    """Compute binomial confidence interval using Wilson score method.

    More accurate than the normal approximation for small n or extreme p.
    """
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (max(0, center - margin), min(1, center + margin))


def cohens_h(p1: float, p2: float) -> float:
    """Compute Cohen's h effect size for two proportions.

    |h| interpretation: 0.2 = small, 0.5 = medium, 0.8 = large.
    """
    return 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))


def effect_size_label(h: float) -> str:
    """Interpret Cohen's h magnitude."""
    h_abs = abs(h)
    if h_abs < 0.2:
        return "negligible"
    if h_abs < 0.5:
        return "small"
    if h_abs < 0.8:
        return "medium"
    return "large"


def mcnemar_test(results_cond1: list, results_cond2: list) -> dict:
    """McNemar's test for paired binary outcomes.

    Tests whether the proportion of correct answers differs between
    two conditions applied to the SAME examples.

    Pairs results by 'question' field to ensure alignment when conditions
    have different skip rates. Falls back to index-based pairing if no
    'question' field is present.
    """
    # Build lookup by question for proper alignment
    has_questions = (results_cond1 and 'question' in results_cond1[0]
                     and results_cond2 and 'question' in results_cond2[0])

    if has_questions:
        lookup2 = {}
        for r in results_cond2:
            lookup2[r['question']] = r

        b = 0  # correct in cond1, wrong in cond2
        c = 0  # wrong in cond1, correct in cond2
        n = 0
        for r1 in results_cond1:
            r2 = lookup2.get(r1['question'])
            if r2 is None:
                continue  # no matching pair — skip
            n += 1
            if r1['correct'] and not r2['correct']:
                b += 1
            elif not r1['correct'] and r2['correct']:
                c += 1
    else:
        # Fallback: index-based pairing (legacy)
        n = min(len(results_cond1), len(results_cond2))
        b = 0
        c = 0
        for i in range(n):
            if results_cond1[i]['correct'] and not results_cond2[i]['correct']:
                b += 1
            elif not results_cond1[i]['correct'] and results_cond2[i]['correct']:
                c += 1

    # McNemar's chi-square (with continuity correction)
    if b + c == 0:
        return {'chi2': 0.0, 'p_value': 1.0, 'b': b, 'c': c, 'n': n}

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_val = 1 - stats.chi2.cdf(chi2, df=1)
    return {'chi2': float(chi2), 'p_value': float(p_val), 'b': b, 'c': c, 'n': n}


def compute_full_statistics(experiment_path: str) -> dict:
    """Compute comprehensive statistics for an experiment.

    Returns dict with per-condition stats including CIs, effect sizes,
    and paired tests.
    """
    with open(experiment_path, 'r') as f:
        data = json.load(f)

    metrics = data['metrics']
    raw = data['raw_results']
    conditions = list(metrics.keys())

    result = {}

    # Per-condition stats with CIs
    for cond in conditions:
        m = metrics[cond]
        n = m['n']
        acc = m['accuracy']
        acc_lo, acc_hi = binomial_ci(acc, n)

        entry = {
            'n': n,
            'accuracy': acc,
            'accuracy_ci': [round(acc_lo, 4), round(acc_hi, 4)],
        }

        if 'conflict' in cond:
            cfr = m['context_following_rate']
            por = m['parametric_override_rate']
            cfr_lo, cfr_hi = binomial_ci(cfr, n)
            por_lo, por_hi = binomial_ci(por, n)
            entry['cfr'] = cfr
            entry['cfr_ci'] = [round(cfr_lo, 4), round(cfr_hi, 4)]
            entry['por'] = por
            entry['por_ci'] = [round(por_lo, 4), round(por_hi, 4)]

        result[cond] = entry

    # Pairwise comparisons: baseline vs each conflict condition
    baseline_cond = 'no_conflict'
    if baseline_cond in metrics:
        baseline_acc = metrics[baseline_cond]['accuracy']
        baseline_n = metrics[baseline_cond]['n']

        for cond in conditions:
            if cond == baseline_cond:
                continue

            cond_acc = metrics[cond]['accuracy']
            cond_n = metrics[cond]['n']

            # Chi-square test — use raw counts for exact values (no rounding)
            baseline_correct = sum(r['correct'] for r in raw[baseline_cond])
            cond_correct = sum(r['correct'] for r in raw[cond])
            ct = [[baseline_correct, baseline_n - baseline_correct],
                  [cond_correct, cond_n - cond_correct]]
            try:
                chi2, p_val, _, _ = stats.chi2_contingency(ct)
            except ValueError:
                chi2, p_val = 0, 1.0

            # Effect size
            h = cohens_h(baseline_acc, cond_acc)

            # McNemar's (paired)
            mcnemar = mcnemar_test(raw[baseline_cond], raw[cond])

            result[cond]['vs_baseline'] = {
                'chi2': round(chi2, 4),
                'p_value': float(f"{p_val:.6f}"),
                'cohens_h': round(h, 4),
                'effect_size': effect_size_label(h),
                'mcnemar_chi2': round(mcnemar['chi2'], 4),
                'mcnemar_p': float(f"{mcnemar['p_value']:.6f}"),
            }

    return result


def generate_statistics_table(experiment_path: str, model_label: str = "") -> str:
    """Generate a markdown table with full statistics."""
    stats_data = compute_full_statistics(experiment_path)

    table = f"### {model_label}\n\n" if model_label else ""
    table += "| Condition | N | Accuracy | 95% CI | CFR | POR | vs Baseline p | Effect Size |\n"
    table += "|-----------|---|----------|--------|-----|-----|---------------|-------------|\n"

    for cond in ['no_conflict', 'conflict_hop1', 'conflict_hop2',
                 'conflict_hop3']:  # 3-hop support
        if cond not in stats_data:
            continue
        s = stats_data[cond]
        ci = f"[{s['accuracy_ci'][0]:.1%}, {s['accuracy_ci'][1]:.1%}]"
        cfr = f"{s.get('cfr', 0):.1%}" if s.get('cfr', 0) > 0 else "-"
        por = f"{s.get('por', 0):.1%}" if s.get('por', 0) > 0 else "-"

        if 'vs_baseline' in s:
            vb = s['vs_baseline']
            p_str = f"{vb['p_value']:.4f}" if vb['p_value'] >= 0.0001 else "<0.0001"
            es = f"h={vb['cohens_h']:.2f} ({vb['effect_size']})"
        else:
            p_str = "-"
            es = "-"

        cond_label = cond.replace('_', ' ').replace('conflict ', 'Conflict@').title()
        if cond == 'no_conflict':
            cond_label = 'No Conflict'

        table += f"| {cond_label} | {s['n']} | {s['accuracy']:.1%} | {ci} | {cfr} | {por} | {p_str} | {es} |\n"

    return table


if __name__ == "__main__":
    import sys
    import glob

    # Generate stats for all models
    paths = glob.glob("outputs/results/*/experiment.json")
    for path in sorted(paths):
        model_id = path.split(os.sep)[-2]
        print(f"\n{'='*60}")
        print(f"Statistics for: {model_id}")
        print(f"{'='*60}")
        table = generate_statistics_table(path, model_id)
        print(table)
