#!/usr/bin/env python3
"""CE cross-domain robustness on OUR corpus, rerank-on vs rerank-off (#104).

The literal sprint5 experiment (``scripts/ce_cross_domain_probe.py``) plugs an
out-of-domain *code-trained* cross-encoder into the rerank slot over a chat run
and asks whether the CE inherits the bi-encoder's domain-mismatch curve. That
needs the ``codecrossenc-v2`` checkpoint, which is not on this host — so this
companion probe tests the equivalent question from the asset we DO have: our
*shipped* chat-domain CE (``ms-marco-TinyBERT-L-2-v2``, wired on by the daemon
env). Does it lift the conversational domain more than the code domain?

Method (READ-ONLY — one ``/search/age-fused`` call per query, no writes):
the age-fused response returns, per hit, both the pre-rerank fusion score
``rrf_score`` and the post-rerank cross-encoder score ``rerank_score`` over the
*same* candidate set. We reconstruct the A/B offline:

  - rerank-OFF ordering = candidates sorted by ``rrf_score`` (desc)
  - rerank-ON  ordering = candidates sorted by ``rerank_score`` (desc)

then score R@5 and MRR under each ordering, per domain. The candidate pool is
identical; only the ordering differs, so the delta isolates the CE's effect.
If the conversational domain gains more under rerank than the code domain, our
shipped default under-serves code-heavy queries — the domain-mismatch asymmetry
mempalace#306 / adaptmem sprint5 hypothesizes.

Relevance: substring match against ``relevant.content_any`` (+ optional
``source_glob`` on the source-file basename), per the existing
``rerank_eval_queries.json`` convention — survives drawer-id churn.

Pure scoring/aggregation is unit-tested in
``tests/test_ce_cross_domain_corpus_probe.py``; only ``collect`` touches the net.

Usage:
    venv/bin/python scripts/ce_cross_domain_corpus_probe.py \\
        --queries scripts/evals/ce_cross_domain_queries.json \\
        --limit 25 \\
        --out baselines/ce_cross_domain_corpus_2026-05-29.json
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import os
import time
import urllib.request
from pathlib import Path

log = logging.getLogger("ce_cross_domain_corpus")

DEFAULT_ENV_FILE = "~/.config/palace-daemon/env"


# --------------------------------------------------------------------------- #
# Relevance + ranking (pure, unit-tested)
# --------------------------------------------------------------------------- #
def hit_relevant(hit: dict, rel: dict) -> bool:
    """A hit is relevant iff a content_any substring matches (and source_glob).

    Case-insensitive substring match against the hit text; optional glob on
    the source-file basename. Empty/missing content_any -> never relevant
    (guards against a mislabeled probe silently passing everything).
    """
    text = (hit.get("text") or "").lower()
    content_any = [s.lower() for s in (rel.get("content_any") or [])]
    if not content_any:
        return False
    if not any(s in text for s in content_any):
        return False
    glob = rel.get("source_glob")
    if glob:
        src = hit.get("source_file") or (hit.get("metadata") or {}).get("source_file") or ""
        if not fnmatch.fnmatch(os.path.basename(str(src)), glob):
            return False
    return True


def order_by(hits: list[dict], field: str) -> list[dict]:
    """Return hits sorted by ``field`` descending; missing -> -inf (sinks)."""
    def key(h):
        v = h.get(field)
        return float(v) if v is not None else float("-inf")
    return sorted(hits, key=key, reverse=True)


def first_relevant_rank(ordered: list[dict], rel: dict) -> int | None:
    """1-based rank of the first relevant hit in an ordering, or None."""
    for i, h in enumerate(ordered):
        if hit_relevant(h, rel):
            return i + 1
    return None


def rank_to_metrics(rank: int | None, k: int = 5) -> tuple[int, float]:
    """(hit@k, reciprocal_rank) from a 1-based rank (None -> miss)."""
    if rank is None:
        return 0, 0.0
    return int(rank <= k), 1.0 / rank


def score_query(hits: list[dict], rel: dict, *, k: int = 5) -> dict:
    """Score one query under both orderings over the same candidate set."""
    off = order_by(hits, "rrf_score")
    on = order_by(hits, "rerank_score")
    rank_off = first_relevant_rank(off, rel)
    rank_on = first_relevant_rank(on, rel)
    h_off, rr_off = rank_to_metrics(rank_off, k)
    h_on, rr_on = rank_to_metrics(rank_on, k)
    return {
        "n_candidates": len(hits),
        "n_relevant_in_pool": sum(1 for h in hits if hit_relevant(h, rel)),
        "rank_off": rank_off, "rank_on": rank_on,
        "hit_off": h_off, "hit_on": h_on,
        "rr_off": rr_off, "rr_on": rr_on,
    }


def aggregate(per_query: list[dict]) -> dict:
    """Aggregate per-query rows into per-domain R@k / MRR under both orderings."""
    by_domain: dict[str, list[dict]] = {}
    for row in per_query:
        by_domain.setdefault(row["domain"], []).append(row)

    def agg(rows: list[dict]) -> dict:
        n = len(rows)
        if n == 0:
            return {"n": 0}
        # Only count queries whose relevant doc is actually in the candidate
        # pool — a query with 0 relevant candidates can't distinguish the two
        # orderings and would dilute the rerank signal toward 0.
        pooled = [r for r in rows if r["score"]["n_relevant_in_pool"] > 0]
        m = len(pooled)
        if m == 0:
            return {"n": n, "n_with_relevant_in_pool": 0,
                    "note": "no relevant docs landed in any candidate pool"}
        r5_off = sum(r["score"]["hit_off"] for r in pooled) / m
        r5_on = sum(r["score"]["hit_on"] for r in pooled) / m
        mrr_off = sum(r["score"]["rr_off"] for r in pooled) / m
        mrr_on = sum(r["score"]["rr_on"] for r in pooled) / m
        return {
            "n": n,
            "n_with_relevant_in_pool": m,
            "r_at_5": {"off": r5_off, "on": r5_on, "delta": r5_on - r5_off},
            "mrr": {"off": mrr_off, "on": mrr_on, "delta": mrr_on - mrr_off},
        }

    domains = {d: agg(rows) for d, rows in sorted(by_domain.items())}
    # Asymmetry: does conversational gain more from rerank than code?
    asymmetry = None
    if "code" in domains and "conversational" in domains:
        cd = domains["code"].get("mrr", {}).get("delta")
        cv = domains["conversational"].get("mrr", {}).get("delta")
        if cd is not None and cv is not None:
            asymmetry = {
                "code_mrr_delta": cd,
                "conversational_mrr_delta": cv,
                "conversational_minus_code": cv - cd,
            }
    return {"by_domain": domains, "asymmetry": asymmetry}


def _is_degenerate(agg: dict) -> str | None:
    """Detect a measurement that can't actually distinguish the orderings.

    Two failure modes make the rerank delta uninformative regardless of CE
    behavior, and we must not dress them up as an H3 "robust" verdict:
      - a domain has too few queries with the gold doc in the candidate pool
        (small/empty effective n), or
      - the relevant docs sit at rank 1 under *both* orderings for (almost)
        every scored query, i.e. the labels are so loose that first-relevant
        is trivially top-1 and the ordering swap can't move it.
    """
    dom = agg.get("by_domain", {})
    for d, dd in dom.items():
        m = dd.get("n_with_relevant_in_pool", 0)
        if m < 3:
            return (f"domain {d!r} has only {m} queries with a relevant doc in "
                    "the candidate pool — too few to measure a rerank delta")
    # both deltas ~0 AND both orderings already at ceiling -> degenerate
    a = agg.get("asymmetry")
    if a and abs(a["code_mrr_delta"]) < 1e-9 and abs(a["conversational_mrr_delta"]) < 1e-9:
        off_on_ceiling = all(
            dd.get("mrr", {}).get("off", 0) >= 0.999 and dd.get("mrr", {}).get("on", 0) >= 0.999
            for dd in dom.values() if "mrr" in dd
        )
        if off_on_ceiling:
            return ("first-relevant is rank 1 under BOTH orderings for every "
                    "scored query — relevance labels are too loose to expose any "
                    "rerank effect (the ordering swap cannot move a top-1 hit)")
    return None


def interpret(agg: dict) -> dict:
    """Map the measured asymmetry onto the sprint5 hypothesis space."""
    degenerate = _is_degenerate(agg)
    if degenerate:
        return {
            "verdict": "INCONCLUSIVE — measurement degenerate, not a robustness finding",
            "why": degenerate,
            "recommendation": (
                "Do NOT read this as H3/robust. A valid run needs (a) the "
                "out-of-domain code-trained CE checkpoint in the rerank slot "
                "and (b) a corpus with a genuine code domain. The live palace "
                "here is conversational-only and the code-CE checkpoint is not "
                "on this host — see the report's blocked_on field."
            ),
        }
    a = agg.get("asymmetry")
    if not a:
        return {"verdict": "inconclusive",
                "why": "need both code and conversational domains with relevant docs in-pool"}
    diff = a["conversational_minus_code"]
    cd, cv = a["code_mrr_delta"], a["conversational_mrr_delta"]
    if cd < -1e-6 and cv >= cd:
        verdict = "H1/H2-flavored: shipped CE HURTS the code domain (rerank ΔMRR<0 there)"
    elif diff > 0.02:
        verdict = "asymmetry present: conversational gains more from rerank than code"
    elif abs(diff) <= 0.02:
        verdict = "H3-flavored: rerank effect is roughly domain-robust on our corpus"
    else:
        verdict = "code gains MORE than conversational (unexpected direction)"
    return {
        "verdict": verdict,
        "conversational_minus_code_mrr_delta": diff,
        "code_mrr_delta": cd,
        "conversational_mrr_delta": cv,
        "recommendation": (
            "Keep rerank default-ON: it does not measurably hurt the code "
            "domain on our corpus."
            if cd >= -1e-6 else
            "Consider a domain-aware gate: rerank measurably hurt code-domain "
            "ordering here; bias code-identifier queries toward no-rerank or a "
            "code-trained CE."
        ),
    }


# --------------------------------------------------------------------------- #
# Daemon retrieval (READ-ONLY)
# --------------------------------------------------------------------------- #
def _parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    except OSError as e:
        log.warning("env file %s unreadable (%s)", path, e)
    return out


def _resolve_daemon(env_file: str | None) -> tuple[str, str]:
    env = _parse_env_file(Path(os.path.expanduser(env_file or DEFAULT_ENV_FILE)))
    url = (os.environ.get("PALACE_DAEMON_URL") or env.get("PALACE_DAEMON_URL")
           or "http://familiar:8085")
    key = os.environ.get("PALACE_API_KEY") or env.get("PALACE_API_KEY") or ""
    return url.rstrip("/"), key


def collect(queries: list[dict], *, daemon_url: str, api_key: str,
            limit: int = 25, sleep_s: float = 0.0) -> list[dict]:
    """Run each query through /search/age-fused (READ-ONLY); score both orderings."""
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    rows: list[dict] = []
    rerank_model = None
    for q in queries:
        payload = json.dumps({"query": q["query"], "limit": limit}).encode()
        req = urllib.request.Request(
            f"{daemon_url}/search/age-fused", data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=40) as resp:
                body = json.loads(resp.read().decode())
        except Exception as e:  # noqa: BLE001
            log.warning("query %r failed: %s", q["id"], e)
            continue
        rerank_model = (body.get("rerank") or {}).get("model") or rerank_model
        hits = body.get("results") or []
        rows.append({
            "id": q["id"], "domain": q["domain"], "query": q["query"],
            "score": score_query(hits, q["relevant"]),
        })
        if sleep_s:
            time.sleep(sleep_s)
    if rows:
        rows[0]["_rerank_model"] = rerank_model
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--queries", type=Path, required=True)
    p.add_argument("--limit", type=int, default=25,
                   help="Candidate pool size requested per query (default 25).")
    p.add_argument("--env-file", default=None)
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    spec = json.loads(args.queries.read_text())
    queries = spec["queries"]
    daemon_url, api_key = _resolve_daemon(args.env_file)
    if not api_key:
        log.error("no PALACE_API_KEY resolved; cannot reach daemon.")
        return 2
    log.info("running %d queries against %s/search/age-fused (READ-ONLY)",
             len(queries), daemon_url)
    rows = collect(queries, daemon_url=daemon_url, api_key=api_key,
                   limit=args.limit, sleep_s=args.sleep)
    if not rows:
        log.error("no rows collected; aborting.")
        return 3
    rerank_model = rows[0].pop("_rerank_model", None)

    agg = aggregate(rows)
    interp = interpret(agg)
    report = {
        "experiment": "CE cross-domain robustness on our corpus, rerank-on vs off (#104 / mempalace#306)",
        "posture": "controlled-condition delta over a fixed candidate pool; rerank reconstructed offline from rrf_score vs rerank_score",
        "rerank_model": rerank_model,
        "limit": args.limit,
        "n_queries": len(rows),
        "blocked_on": (
            [
                "Out-of-domain code-trained CE checkpoint (codecrossenc-v2-20260516) "
                "is not on this host — it was trained on macmini/Colab and never "
                "synced. That checkpoint IS the sprint5 treatment; without it the "
                "cross-domain hypothesis (H1/H2/H3) cannot be tested directly.",
                "The live palace (familiar:8085) is conversational-only — source "
                "files are .jsonl/.txt/.md conversation logs, not indexed .py "
                "source. There is no genuine code domain in the corpus to contrast "
                "against, so a code-vs-conversational split is not measurable here.",
            ]
            if interp["verdict"].startswith("INCONCLUSIVE") else None
        ),
        "per_query": rows,
        **agg,
        "interpretation": interp,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    log.info("rerank model: %s", rerank_model)
    for d, dd in agg["by_domain"].items():
        if "mrr" in dd:
            log.info("  %-15s MRR %.3f->%.3f (Δ%+.3f)  R@5 %.3f->%.3f (Δ%+.3f)  [n=%d, in-pool=%d]",
                     d, dd["mrr"]["off"], dd["mrr"]["on"], dd["mrr"]["delta"],
                     dd["r_at_5"]["off"], dd["r_at_5"]["on"], dd["r_at_5"]["delta"],
                     dd["n"], dd["n_with_relevant_in_pool"])
        else:
            log.info("  %-15s %s", d, dd)
    log.info("verdict: %s", report["interpretation"]["verdict"])
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
