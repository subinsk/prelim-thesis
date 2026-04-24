"""
Refetch Wikipedia pageviews for entities cached as -1 (failed lookup).

The original popularity cache has 803 of 864 entities coded as -1 (API failure),
many of which are well-known entities (Orson Welles, Continental Army, T. R. M.
Howard, ...) whose direct lookups succeed on retry. The original code silently
classified every -1 as "low popularity," contaminating the popularity analysis.

This script mirrors the fail-safe patterns used elsewhere in the project
(``src/experiments/framework.py``):

  * Atomic writes  -- write to ``*.tmp`` then ``os.rename`` so a crash mid-write
    cannot corrupt the cache.
  * Auto-checkpoints  -- flush cache to disk every ``BATCH`` items and keep a
    separate checkpoint pointer so a restart resumes exactly where it stopped.
  * Resume-safe  -- entities already recovered in a prior run are skipped.
  * Run history  -- every invocation appends a row to
    ``entity_popularity_refetch_history.json``.
  * Per-item error isolation  -- an individual API failure is logged and the
    loop moves on.

Run:
    python -m src.analysis.refetch_popularity
    python -m src.analysis.refetch_popularity --batch 25
    python -m src.analysis.refetch_popularity --limit 100          # quick test
"""

import argparse
import json
import os
import signal
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime

CACHE_PATH = "outputs/results/entity_popularity_cache.json"
BACKUP_PATH = "outputs/results/entity_popularity_cache.stale_before_refetch.json"
CHECKPOINT_PATH = "outputs/results/entity_popularity_refetch_checkpoint.json"
HISTORY_PATH = "outputs/results/entity_popularity_refetch_history.json"

PAGEVIEWS_API = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
OPENSEARCH_API = "https://en.wikipedia.org/w/api.php"
MEDIAWIKI_API = "https://en.wikipedia.org/w/api.php"
START_DATE = "20250301"
END_DATE = "20260301"
USER_AGENT = "PrelimThesisResearch/1.0 (academic audit refetch)"

DEFAULT_BATCH = 25        # flush every N entities
DEFAULT_DELAY = 0.08      # seconds between API calls (be polite)


# ---------- Atomic IO helpers ----------

def _atomic_write(path, payload):
    """Write JSON to ``path`` via temp + rename. Safe across crashes."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    if os.path.exists(path):
        os.remove(path)
    os.rename(tmp, path)


def _load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


# ---------- Wikipedia API ----------

def _pageviews(title):
    """Average monthly views for ``title``. Returns -1 if no article / API fails."""
    encoded = urllib.parse.quote(title.replace(" ", "_"), safe="")
    url = f"{PAGEVIEWS_API}/en.wikipedia/all-access/user/{encoded}/monthly/{START_DATE}/{END_DATE}"
    for attempt in range(2):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
            views = [item["views"] for item in data.get("items", [])]
            if views:
                return int(sum(views) / len(views))
            return -1
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return -1  # genuine: no article at this exact title
            time.sleep(1.0 + attempt)
        except Exception:
            time.sleep(1.0 + attempt)
    return -1


def _opensearch_best_title(query):
    """Canonical Wikipedia title for ``query`` via MediaWiki opensearch, or None."""
    params = {
        "action": "opensearch",
        "search": query,
        "limit": "1",
        "namespace": "0",
        "format": "json",
    }
    url = f"{OPENSEARCH_API}?" + urllib.parse.urlencode(params)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        if len(data) >= 2 and data[1]:
            return data[1][0]
    except Exception:
        pass
    return None


def _resolve_redirect(title):
    """Follow a Wikipedia redirect to its canonical article title, or return
    the input if it isn't a redirect. Returns None on API failure.

    Wikipedia's pageviews API is case-sensitive and does NOT follow
    redirects, so passing "lower_Manhattan" (a redirect stub) yields near-zero
    views. This function asks the MediaWiki query API to do the redirect
    resolution server-side, so we can then query pageviews on the canonical
    title.
    """
    params = {
        "action": "query",
        "titles": title,
        "redirects": "1",
        "format": "json",
    }
    url = f"{MEDIAWIKI_API}?" + urllib.parse.urlencode(params)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        pages = data.get("query", {}).get("pages", {})
        for _, page in pages.items():
            if "missing" in page:
                return None
            if "title" in page:
                return page["title"]
    except Exception:
        return None
    return title


def resolve_one(entity):
    """Resolve ``entity`` to a view count and a resolution method.

    The Wikipedia pageviews API is case-sensitive and does NOT follow
    redirects. Hitting a redirect-stub URL (e.g. ``lower_Manhattan``)
    returns views for the stub itself (near zero), not for the canonical
    article (``Lower_Manhattan``). So we always resolve via the MediaWiki
    ``query?redirects=1`` endpoint first, which returns the canonical
    title server-side, then ask pageviews for that.

    Returns ``(views, method)`` where method is one of:
      ``'canonical'``    -- redirect-resolved title, non-zero views
      ``'opensearch'``   -- opensearch fallback found a matching title
      ``'direct_zero'``  -- canonical article exists but has 0 views
      ``'missing'``      -- no Wikipedia article at all
    """
    # Step 1: ask the query API to resolve redirects and return the canonical
    # title. This handles both redirects and case-folding in one call.
    canonical = _resolve_redirect(entity)
    if canonical:
        views = _pageviews(canonical)
        if views > 0:
            return views, "canonical"
        if views == 0:
            return 0, "direct_zero"

    # Step 2: opensearch fallback. Sometimes the exact title isn't found
    # but a search finds a reasonable match (e.g. misspellings).
    best = _opensearch_best_title(entity)
    if best:
        # Resolve that opensearch hit through redirects too.
        canonical2 = _resolve_redirect(best) or best
        views2 = _pageviews(canonical2)
        if views2 > 0:
            return views2, "opensearch"
        if views2 == 0:
            return 0, "direct_zero"

    return -1, "missing"


# ---------- Run history ----------

def _log_run(status, n_queried, recovered_direct, recovered_search, missing,
             start, end, detail=""):
    history = _load_json(HISTORY_PATH, [])
    history.append({
        "run_number": len(history) + 1,
        "status": status,
        "n_queried": n_queried,
        "recovered_direct": recovered_direct,
        "recovered_opensearch": recovered_search,
        "still_missing": missing,
        "start_time": start,
        "end_time": end,
        "duration_seconds": round(
            (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds(), 1
        ),
        "detail": detail,
    })
    _atomic_write(HISTORY_PATH, history)


# ---------- Main loop ----------

def refetch(batch=DEFAULT_BATCH, delay=DEFAULT_DELAY, limit=None):
    cache = _load_json(CACHE_PATH, {})
    checkpoint = _load_json(CHECKPOINT_PATH, {"processed": [], "next_index": 0})

    # One-time backup of the pre-refetch cache.
    if not os.path.exists(BACKUP_PATH):
        _atomic_write(BACKUP_PATH, cache)
        print(f"[backup] Saved pre-refetch cache -> {BACKUP_PATH}")

    # Build the work list: entities currently cached as -1,
    # minus those already reprocessed in prior runs.
    processed = set(checkpoint.get("processed", []))
    todo = [k for k, v in cache.items() if v == -1 and k not in processed]
    if limit is not None:
        todo = todo[:limit]

    print(f"[plan] total={len(cache)}  to_retry={len(todo)}  "
          f"already_reprocessed={len(processed)}  batch={batch}")

    if not todo:
        print("[plan] nothing to do; cache already refetched.")
        return

    start_ts = datetime.now().isoformat()
    recovered_direct = 0
    recovered_search = 0
    missing = 0
    stop_reason = "completed"
    stop_detail = ""

    def _flush(index_in_todo):
        """Atomically persist cache + checkpoint pointer."""
        _atomic_write(CACHE_PATH, cache)
        checkpoint["processed"] = sorted(processed)
        checkpoint["next_index"] = index_in_todo
        checkpoint["last_flush"] = datetime.now().isoformat()
        _atomic_write(CHECKPOINT_PATH, checkpoint)

    # Handle Ctrl+C cleanly on both POSIX and Windows.
    interrupted = {"flag": False}
    def _sigint(sig, frame):
        interrupted["flag"] = True
        print("\n[interrupt] received; flushing and exiting after current item...")
    try:
        signal.signal(signal.SIGINT, _sigint)
    except (ValueError, AttributeError):
        pass  # not in main thread on some platforms

    try:
        for i, entity in enumerate(todo, start=1):
            try:
                views, method = resolve_one(entity)
            except Exception as e:
                print(f"  [skip] {entity!r}: {e}")
                processed.add(entity)  # don't retry the same broken entity forever
                continue

            cache[entity] = views
            processed.add(entity)

            if method == "direct":
                recovered_direct += 1
            elif method == "opensearch":
                recovered_search += 1
            else:
                missing += 1

            if i % batch == 0:
                _flush(i)
                print(f"  [batch {i//batch:>3}] processed={i}/{len(todo)}  "
                      f"direct={recovered_direct} opensearch={recovered_search} "
                      f"missing={missing}")

            if interrupted["flag"]:
                stop_reason = "aborted"
                stop_detail = "SIGINT"
                break

            time.sleep(delay)

    except Exception as e:
        stop_reason = "error"
        stop_detail = f"{type(e).__name__}: {e}"
        print(f"[error] {stop_detail}")

    # Final flush.
    _flush(len(processed))

    end_ts = datetime.now().isoformat()
    _log_run(
        stop_reason, len(todo),
        recovered_direct, recovered_search, missing,
        start_ts, end_ts, stop_detail,
    )

    # Summary.
    all_vals = list(cache.values())
    neg = sum(1 for v in all_vals if v == -1)
    high = sum(1 for v in all_vals if isinstance(v, (int, float)) and v > 10000)
    med = sum(1 for v in all_vals if isinstance(v, (int, float)) and 1000 <= v <= 10000)
    low = sum(1 for v in all_vals if isinstance(v, (int, float)) and 0 < v < 1000)
    zero = sum(1 for v in all_vals if v == 0)

    print("\n=== Refetch summary ===")
    print(f"  status:             {stop_reason}")
    print(f"  queried this run:   {len(todo)}")
    print(f"  recovered direct:   {recovered_direct}")
    print(f"  recovered search:   {recovered_search}")
    print(f"  still missing:      {missing}")
    print("\n=== Post-refetch cache distribution ===")
    print(f"  total entries:      {len(all_vals)}")
    print(f"  high  (>10K):       {high}")
    print(f"  med   (1K-10K):     {med}")
    print(f"  low   (0<v<1K):     {low}")
    print(f"  zero:               {zero}")
    print(f"  missing (-1):       {neg}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                   help=f"Flush to disk every N entities (default: {DEFAULT_BATCH})")
    p.add_argument("--delay", type=float, default=DEFAULT_DELAY,
                   help=f"Seconds between API calls (default: {DEFAULT_DELAY})")
    p.add_argument("--limit", type=int, default=None,
                   help="Only process the first N pending entities (for testing)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    refetch(batch=args.batch, delay=args.delay, limit=args.limit)
