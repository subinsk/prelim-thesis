"""
Entity Popularity Scoring via Wikipedia Page Views.

Uses the Wikimedia REST API to get monthly page views for answer entities.
Results are cached locally to avoid redundant API calls.

Key hypothesis: popular entities (strong parametric memory) lead to higher POR,
while obscure entities lead to higher CFR under knowledge conflict.
"""

import json
import os
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict

CACHE_PATH = "outputs/results/entity_popularity_cache.json"
WIKI_API = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
# Use last 12 months of data for stable popularity estimates
START_DATE = "20250301"
END_DATE = "20260301"


def _load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, 'r') as f:
            return json.load(f)
    return {}


def _save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, 'w') as f:
        json.dump(cache, f, indent=2)


def get_wikipedia_pageviews(title: str, cache: dict = None) -> int:
    """Get average monthly page views for a Wikipedia article.

    Returns average monthly views, or -1 if article not found.
    """
    if cache is None:
        cache = {}

    if title in cache:
        return cache[title]

    encoded = urllib.parse.quote(title.replace(" ", "_"), safe="")
    url = f"{WIKI_API}/en.wikipedia/all-access/user/{encoded}/monthly/{START_DATE}/{END_DATE}"

    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "PrelimThesisResearch/1.0 (academic research)"
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        views = [item["views"] for item in data.get("items", [])]
        avg = int(sum(views) / len(views)) if views else -1
    except Exception:
        avg = -1

    cache[title] = avg
    return avg


def score_entities(answers: list, batch_delay: float = 0.05) -> dict:
    """Score a list of answer entities by Wikipedia popularity.

    Args:
        answers: list of answer strings
        batch_delay: seconds between API calls (be polite)

    Returns:
        dict mapping answer -> avg monthly page views (-1 if not found)
    """
    cache = _load_cache()
    unique = list(set(answers))
    new_count = sum(1 for a in unique if a not in cache)

    if new_count > 0:
        print(f"Fetching Wikipedia page views for {new_count} new entities...")

    for i, answer in enumerate(unique):
        if answer in cache:
            continue
        get_wikipedia_pageviews(answer, cache)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(unique)} fetched")
            _save_cache(cache)  # periodic save
        time.sleep(batch_delay)

    _save_cache(cache)
    return {a: cache.get(a, -1) for a in answers}


def classify_popularity(views: int) -> str:
    """Classify into High/Medium/Low/Unknown popularity bins.

    Thresholds based on typical Wikipedia page view distributions:
    - High:    > 10,000 avg monthly views (well-known)
    - Medium:  1,000 - 10,000 (moderately known)
    - Low:     < 1,000 (obscure but has a Wikipedia page)
    - Unknown: no Wikipedia page resolves (exclude from analysis)

    Note: previously -1 was silently coerced to "low"; this contaminated the
    low bin with API-failure placeholders. See thesis/AUDIT.md A1.
    """
    if views < 0:
        return "unknown"  # no page found — excluded from popularity analysis
    if views >= 10000:
        return "high"
    if views >= 1000:
        return "medium"
    return "low"


def analyze_popularity_effect(results_path: str, n_examples: int = None):
    """Analyze how entity popularity affects conflict resolution.

    Loads existing experiment results, scores answer entities by popularity,
    and compares CFR/POR across popularity bins.

    Args:
        results_path: path to experiment.json (e.g. outputs/results/llama-3.3-70b-versatile/experiment.json)
        n_examples: limit number of examples (None = all)

    Returns:
        dict with per-bin metrics
    """
    with open(results_path, 'r') as f:
        data = json.load(f)

    raw = data['raw_results']

    # Only use factual answers (skip temporal/numerical)
    from src.data.conflict_injector import ConflictInjector

    # Collect answer entities from no_conflict results
    answers = []
    indices = []
    for i, r in enumerate(raw['no_conflict']):
        answer = r['gold']
        atype = ConflictInjector.classify_answer_type(answer)
        if atype == ConflictInjector.TYPE_FACTUAL:
            answers.append(answer)
            indices.append(i)

    if n_examples:
        answers = answers[:n_examples]
        indices = indices[:n_examples]

    print(f"Scoring {len(answers)} factual answer entities...")
    scores = score_entities(answers)

    # Bin by popularity
    bins = defaultdict(lambda: {"no_conflict": [], "conflict_hop1": [], "conflict_hop2": []})

    for answer, idx in zip(answers, indices):
        views = scores[answer]
        pop_bin = classify_popularity(views)

        for cond in ["no_conflict", "conflict_hop1", "conflict_hop2"]:
            if idx < len(raw[cond]):
                entry = raw[cond][idx]
                entry['popularity'] = pop_bin
                entry['page_views'] = views
                bins[pop_bin][cond].append(entry)

    # Compute metrics per bin
    results = {}
    for pop_bin in ["high", "medium", "low"]:
        bin_data = bins[pop_bin]
        metrics = {}
        for cond in ["no_conflict", "conflict_hop1", "conflict_hop2"]:
            entries = bin_data[cond]
            n = len(entries)
            if n == 0:
                continue
            accuracy = sum(r['correct'] for r in entries) / n
            cfr = sum(r.get('followed_context', False) for r in entries) / n if 'conflict' in cond else 0
            por = sum(r.get('used_parametric', False) for r in entries) / n if 'conflict' in cond else 0
            metrics[cond] = {
                'n': n,
                'accuracy': accuracy,
                'context_following_rate': cfr,
                'parametric_override_rate': por,
            }
        results[pop_bin] = metrics

    # Print summary
    print(f"\n{'='*60}")
    print("ENTITY POPULARITY EFFECT")
    print(f"{'='*60}")
    for pop_bin in ["high", "medium", "low"]:
        m = results.get(pop_bin, {})
        nc = m.get('no_conflict', {})
        h1 = m.get('conflict_hop1', {})
        h2 = m.get('conflict_hop2', {})
        print(f"\n  {pop_bin.upper()} popularity (n={nc.get('n', 0)}):")
        print(f"    Baseline: {nc.get('accuracy', 0):.1%}")
        if h1:
            print(f"    Conflict@Hop1: acc={h1['accuracy']:.1%} CFR={h1['context_following_rate']:.1%} POR={h1['parametric_override_rate']:.1%}")
        if h2:
            print(f"    Conflict@Hop2: acc={h2['accuracy']:.1%} CFR={h2['context_following_rate']:.1%} POR={h2['parametric_override_rate']:.1%}")

    return results, scores


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "outputs/results/llama-3.3-70b-versatile/experiment.json"
    results, scores = analyze_popularity_effect(path)

    # Save analysis
    out_path = "outputs/results/entity_popularity_analysis.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
