"""LoCoMo end-to-end on the mempalace DAEMON (real pgvector + Apache AGE).

Issue techempower-org/multipass-structural-memory-eval#176 — the "daemon next"
half of JP's flat-now/daemon-next decision. Companion to
``scripts/run_longmemeval_mempalace.py`` (the LongMemEval per-question daemon
runner): this one runs the LoCoMo corpus through the same wing-scoped daemon
retrieval path, with LoCoMo's per-sample ingest topology.

Design — identical to the flat LoCoMo run (``sme-eval``'s LoCoMo path / the
stratified flat baseline) EXCEPT the retrieval substrate:

* **Substrate.** Wing-scoped ``/search/age-fused`` (or ``/search``) against a
  postgres-backed daemon — pgvector vectors + an Apache AGE knowledge graph —
  instead of an ephemeral per-sample Chroma index.
* **Ingest topology.** LoCoMo shares one conversation across a sample's
  questions. Each sample's sessions are POSTed to the daemon under a per-sample
  wing ``locomo_<sample_id>`` (room ``sessions``), mirroring the per-question
  wing scoping in ``run_longmemeval_mempalace.py``. The adapter scopes its
  search to that wing, so cross-sample contamination is impossible even though
  the daemon owns a single palace.

Everything else — stratified subset selection, the canonical LongMemEval
type-specific judge (``temporal`` → off-by-one, ``adversarial`` → abstention),
the per-LoCoMo-type + overall aggregation — is reused verbatim so daemon
numbers are a like-for-like comparison against the flat baseline.

ISOLATION CONTRACT. LoCoMo ingest writes thousands of turns; it must NEVER
touch a production palace. ``isolation_guard`` refuses to run unless the daemon
URL is a localhost instance AND the palace is empty (drawer count 0). Stand up
a throwaway scratch daemon (scratch Postgres+pgvector+AGE) and point
``--api-url`` / ``PALACE_DAEMON_URL`` at it. See
``docs/benchmarks/2026-05-29-locomo-daemon-results.md`` for the provisioning
recipe.

Usage::

    PALACE_DAEMON_URL=http://localhost:8086 PALACE_API_KEY=... \\
    AZURE_API_KEY=... AZURE_API_BASE=... \\
    python scripts/run_locomo_mempalace_daemon.py \\
      --per-type 50 --seed 1729 --search-endpoint /search/age-fused \\
      --out baselines/locomo_daemon_age_fused_<date>.json \\
      --status /tmp/locomo_daemon.STATUS
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import random
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cross_validate_longmemeval as harness  # noqa: E402
import run_longmemeval_mempalace as lme_daemon  # noqa: E402

from sme.adapters.base import QueryResult, SMEAdapter  # noqa: E402
from sme.corpora.locomo import load_questions  # noqa: E402
from sme.corpora.locomo.loader import (  # noqa: E402
    ADVERSARIAL_INCLUDED,
    SUBSET,
    SUBSET_QA_COUNT,
)

WING_PREFIX = "locomo_"
ROOM = "sessions"  # one of the 7 spec-default canonical rooms (fresh scratch DB)
DEFAULT_PER_TYPE = 50
DEFAULT_SEED = 1729
READER_MODEL = JUDGE_MODEL = "gpt-5.3-chat"


def sample_wing(sample_id: str) -> str:
    """Per-sample wing slug. Hyphens → underscores to match the daemon's
    wing-slug normalisation (``conv-26`` → ``locomo_conv_26``)."""
    return f"{WING_PREFIX}{sample_id}".replace("-", "_")


def render_session_md(session: Any) -> str:
    """sme-rich rendering of one LoCoMo session — matches the on-disk drawer
    shape ``materialize_sme_corpus`` writes for the flat run, so daemon-vs-flat
    compares retrieval substrate, not text rendering."""
    parts = [f"# Session {session.session_id}", f"_Date: {session.date}_", ""]
    for turn in session.turns:
        parts.append(f"## {turn.speaker}\n\n{turn.text}")
    return "\n".join(parts)


def ingest_sample(ingest_client, sample_qs: list, wing: str) -> dict:
    """POST each of a sample's sessions to the daemon under ``wing``.

    Sessions are shared across the sample's questions, so the first question
    carries the full conversation. Returns
    ``{posted, errors, session_to_drawer}``; the map (session_id → drawer_id)
    powers drawer-based R@K (#58), with the chunk suffix stripped at compare
    time (#98).
    """
    q0 = sample_qs[0]
    posted, errors = 0, []
    session_to_drawer: dict[str, str] = {}
    for s in q0.sessions:
        status, body = ingest_client.post_memory(
            content=render_session_md(s), wing=wing, room=ROOM,
        )
        if status not in (200, 201):
            errors.append(
                f"{s.session_id}: HTTP {status} {body.get('_raw') or body!r}"
            )
            continue
        posted += 1
        did = body.get("drawer_id") if isinstance(body, dict) else None
        if did is not None:
            session_to_drawer[s.session_id] = str(did)
    return {"posted": posted, "errors": errors, "session_to_drawer": session_to_drawer}


def stratified_sample(questions: list, per_type: int, seed: int) -> tuple[list, dict]:
    """Deterministic per-question_type sample (mirrors the flat stratified run).
    Stable sort by question_id, fixed-seed shuffle, take ``per_type``; then
    re-group sample-contiguous so each sample's vault is ingested once."""
    by_type = collections.defaultdict(list)
    for q in questions:
        by_type[q.question_type].append(q)
    rng = random.Random(seed)
    picked, counts = [], {}
    for t in sorted(by_type):
        pool = sorted(by_type[t], key=lambda q: q.question_id)
        rng.shuffle(pool)
        take = pool[:per_type]
        counts[t] = len(take)
        picked.extend(take)
    picked.sort(key=lambda q: (q.sample_id, q.question_id))
    return picked, counts


# --- Isolation guard --------------------------------------------------------

def _drawer_count(api_url: str, api_key: str) -> int:
    """Total drawer count via GET /list. Returns -1 if it can't be read."""
    req = urllib.request.Request(
        f"{api_url}/list?limit=1", headers={"X-API-Key": api_key}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return int(body.get("total") or body.get("total_count") or 0)
    except Exception:  # noqa: BLE001
        return -1


def isolation_guard(api_url: str, api_key: str, *, count_fn=_drawer_count) -> None:
    """Refuse to run against anything but an empty localhost scratch daemon.

    Two independent gates: (1) the URL host must be localhost/127.0.0.1, and
    (2) the palace must hold 0 drawers. Either failing aborts. This is the
    safety contract from #176 — LoCoMo ingest must never pollute a production
    palace's live knowledge graph.
    """
    if not (("localhost" in api_url) or ("127.0.0.1" in api_url)):
        raise SystemExit(
            f"ISOLATION GUARD: refusing — api_url={api_url!r} is not a localhost "
            "scratch instance. NEVER run LoCoMo ingest against a production daemon."
        )
    n = count_fn(api_url, api_key)
    if n != 0:
        raise SystemExit(
            f"ISOLATION GUARD: refusing — daemon at {api_url} reports drawers={n} "
            "(expected an EMPTY scratch palace; -1 means could not verify)."
        )


# --- Backfill ---------------------------------------------------------------

def backfill_age(api_url: str, api_key: str, *, poll_timeout_s: int = 1800) -> dict:
    """POST /backfill-age then poll /backfill-age/status to completion.
    Builds the full Wing/Room/Drawer + entity structure age-fused leans on."""
    req = urllib.request.Request(
        f"{api_url}/backfill-age", data=b"{}",
        headers={"X-API-Key": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        start = json.loads(resp.read().decode("utf-8"))
    t0 = time.time()
    while time.time() - t0 < poll_timeout_s:
        time.sleep(5)
        sreq = urllib.request.Request(
            f"{api_url}/backfill-age/status", headers={"X-API-Key": api_key}
        )
        try:
            with urllib.request.urlopen(sreq, timeout=30) as r:
                st = json.loads(r.read().decode("utf-8"))
        except Exception:  # noqa: BLE001 — transient; keep polling
            continue
        if not st.get("in_progress", False):
            return {"start": start, **st}
    return {"start": start, "status": "timeout"}


# --- Core run (injectable for tests) ----------------------------------------

def _group_by_sample(questions: list, limit_samples: Optional[int]) -> tuple[dict, list]:
    by_sample: dict[str, list] = {}
    order: list[str] = []
    for q in questions:
        if q.sample_id not in by_sample:
            by_sample[q.sample_id] = []
            order.append(q.sample_id)
        by_sample[q.sample_id].append(q)
    if limit_samples is not None:
        order = order[:limit_samples]
    return by_sample, order


def ingest_all_samples(
    *,
    by_sample: dict,
    order: list,
    ingest_client,
    log: Callable[[str], None] = lambda _m: None,
) -> tuple[dict, dict]:
    """Ingest every sample's sessions once under its per-sample wing.
    Returns ``(sample_to_session_drawer_map, ingest_total)``."""
    ingest_total = {"posted": 0, "errors": 0}
    sample_s2d: dict[str, dict] = {}
    for i, sid in enumerate(order):
        wing = sample_wing(sid)
        rep = ingest_sample(ingest_client, by_sample[sid], wing)
        ingest_total["posted"] += rep["posted"]
        ingest_total["errors"] += len(rep["errors"])
        sample_s2d[sid] = rep["session_to_drawer"]
        log(f"INGEST {i + 1}/{len(order)} {sid} -> {wing} posted={rep['posted']} "
            f"errors={len(rep['errors'])} | running posted={ingest_total['posted']}")
    return sample_s2d, ingest_total


def query_all_samples(
    *,
    by_sample: dict,
    order: list,
    sample_s2d: dict,
    factory_fn: Callable[[str], SMEAdapter],
    reader_model: str,
    judge_model: str,
    log: Callable[[str], None] = lambda _m: None,
    reader_client: Optional[Any] = None,
    judge_client: Optional[Any] = None,
) -> list[dict]:
    """Query+judge every question wing-scoped. Call AFTER ingest (and after
    /backfill-age, so age-fusion sees the full graph). ``factory_fn(wing)``
    builds the wing-scoped adapter; clients are injectable for tests."""
    records: list[dict] = []
    n_q = sum(len(by_sample[s]) for s in order)
    done = 0
    for sid in order:
        wing = sample_wing(sid)
        s2d = sample_s2d[sid]
        adapter = factory_fn(wing)
        try:
            for q in by_sample[sid]:
                try:
                    result = adapter.query(q.question, n_results=5)
                except Exception as e:  # noqa: BLE001
                    result = QueryResult(answer="", context_string="", error=str(e))
                rec = harness._score_and_judge(
                    question=q.question,
                    question_id=q.question_id,
                    question_type=q.question_type,
                    sme_category=q.sme_category,
                    is_abstention=q.is_adversarial,
                    gold_answer=q.gold_answer,
                    expected=q.expected_sources_session_level(),
                    result=result,
                    skip_judge=False,
                    skip_reader=False,
                    reader_model=reader_model,
                    judge_model=judge_model,
                    reader_client=reader_client,
                    judge_client=judge_client,
                    capture_context=False,
                    extra_fields={
                        "sample_id": q.sample_id,
                        "locomo_category": q.category,
                        "is_adversarial": q.is_adversarial,
                        "wing": wing,
                    },
                )
                expected_drawers = {
                    s2d[s] for s in q.expected_sources_session_level() if s in s2d
                }
                parents = [
                    lme_daemon._drawer_parent_id(d)
                    for d in (rec.get("retrieved_entity_ids") or [])
                ]
                rec["expected_drawer_ids"] = sorted(expected_drawers)
                rec["retrieved_parent_ids"] = parents
                rec["drawer_hit_at_1"] = bool(
                    expected_drawers and parents and parents[0] in expected_drawers
                )
                rec["drawer_hit_at_5"] = bool(
                    expected_drawers and any(d in expected_drawers for d in parents[:5])
                )
                records.append(rec)
                done += 1
                if done % 10 == 0:
                    log(f"QUERY {done}/{n_q} last={q.question_id} ({q.question_type})")
        finally:
            try:
                adapter.close()
            except Exception:  # noqa: BLE001
                pass
    return records


def build_report(records: list[dict], ingest_total: dict, *, meta: dict) -> dict:
    """Per-LoCoMo-type + overall QA aggregation (the flat run's contract) plus
    drawer-R@5 and substring-recall retrieval summaries."""
    def is_right(label: str) -> bool:
        return harness.judge_label_to_correct(label) is True

    by_type = collections.defaultdict(list)
    for r in records:
        by_type[r["question_type"]].append(
            (r.get("judge") or {}).get("autoeval_label", "ERROR")
        )
    per_type = {}
    for t in sorted(by_type):
        labels = by_type[t]
        scored = [x for x in labels if x != "ERROR"]
        acc = (sum(1 for x in scored if is_right(x)) / len(scored)) if scored else None
        per_type[t] = {
            "n": len(labels), "n_scored": len(scored),
            "n_error": len(labels) - len(scored),
            "qa_accuracy": round(acc, 4) if acc is not None else None,
            "label_counts": dict(collections.Counter(labels)),
        }
    all_labels = [(r.get("judge") or {}).get("autoeval_label", "ERROR") for r in records]
    scored_all = [x for x in all_labels if x != "ERROR"]
    overall = (sum(1 for x in scored_all if is_right(x)) / len(scored_all)) if scored_all else None

    dh5 = [r for r in records if r.get("expected_drawer_ids")]
    drawer_r5 = (sum(1 for r in dh5 if r.get("drawer_hit_at_5")) / len(dh5)) if dh5 else None
    substring_r = (sum(r.get("sme_recall", 0.0) for r in records) / len(records)) if records else None

    summary = harness.aggregate(records)
    return {
        "run_metadata": {**meta, "ingest_posted": ingest_total["posted"],
                         "ingest_errors": ingest_total["errors"], "n_run": len(records)},
        "retrieval": {
            "drawer_recall_at_5": round(drawer_r5, 4) if drawer_r5 is not None else None,
            "substring_recall_mean": round(substring_r, 4) if substring_r is not None else None,
            "n_with_expected_drawers": len(dh5),
        },
        "qa_by_locomo_type": per_type,
        "qa_overall": {
            "n": len(all_labels), "n_scored": len(scored_all),
            "n_error": len(all_labels) - len(scored_all),
            "qa_accuracy": round(overall, 4) if overall is not None else None,
        },
        "sme_category_view": summary["per_category"],
        "judge_total_usage": summary["judge_total_usage"],
        "per_question": records,
    }


def _build_factory(api_url: str, api_key: str, search_endpoint: str) -> Callable[[str], SMEAdapter]:
    def _factory(wing: str) -> SMEAdapter:
        return lme_daemon._make_wing_scoped_daemon_adapter(
            api_url=api_url, api_key=api_key, wing=wing,
            kind="all", search_endpoint=search_endpoint,
        )
    return _factory


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="sme/corpora/locomo/data/locomo10.json")
    ap.add_argument("--api-url", default=os.environ.get("PALACE_DAEMON_URL"))
    ap.add_argument("--api-key", default=os.environ.get("PALACE_API_KEY"))
    ap.add_argument("--per-type", type=int, default=DEFAULT_PER_TYPE)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--search-endpoint", default="/search/age-fused")
    ap.add_argument("--reader-model", default=READER_MODEL)
    ap.add_argument("--judge-model", default=JUDGE_MODEL)
    ap.add_argument("--out", required=True)
    ap.add_argument("--status")
    ap.add_argument("--skip-backfill", action="store_true")
    ap.add_argument("--limit-samples", type=int, default=None)
    args = ap.parse_args(argv)

    if not args.api_url or not args.api_key:
        raise SystemExit("need --api-url/--api-key (or PALACE_DAEMON_URL/PALACE_API_KEY)")

    statusf = pathlib.Path(args.status) if args.status else None

    def log(m: str) -> None:
        if statusf is not None:
            statusf.write_text(m + "\n")
        print(m, flush=True)

    isolation_guard(args.api_url, args.api_key)
    log(f"isolation guard OK: {args.api_url} localhost + empty")

    ingest_client = lme_daemon.DaemonIngestClient(
        api_url=args.api_url, api_key=args.api_key, timeout_s=60.0
    )
    t0 = time.time()
    all_qs = list(load_questions(args.dataset))
    subset, counts = stratified_sample(all_qs, args.per_type, args.seed)
    log(f"INGESTING per_type={args.per_type} seed={args.seed} "
        f"endpoint={args.search_endpoint} | started {time.strftime('%FT%T%z')}")

    by_sample, order = _group_by_sample(subset, args.limit_samples)

    # 1. Ingest every sample's sessions once under its per-sample wing.
    sample_s2d, ingest_total = ingest_all_samples(
        by_sample=by_sample, order=order, ingest_client=ingest_client, log=log,
    )
    ingest_client.post_flush()

    # 2. Backfill the AGE graph BEFORE querying so /search/age-fused sees the
    #    full Wing/Room/Drawer + entity structure, not just the inline
    #    write-through MENTIONS edges. (Ordering matters: querying first would
    #    under-serve age-fusion.)
    backfill_info: dict = {"skipped": True}
    if not args.skip_backfill:
        log(f"BACKFILL-AGE: {ingest_total['posted']} drawers...")
        backfill_info = backfill_age(args.api_url, args.api_key)
        log(f"BACKFILL-AGE done rc={backfill_info.get('returncode')}")

    # 3. Query + judge every question wing-scoped.
    records = query_all_samples(
        by_sample=by_sample, order=order, sample_s2d=sample_s2d,
        factory_fn=_build_factory(args.api_url, args.api_key, args.search_endpoint),
        reader_model=args.reader_model, judge_model=args.judge_model, log=log,
    )

    dt = time.time() - t0
    meta = {
        "issue": "techempower-org/multipass-structural-memory-eval#176",
        "corpus": "locomo",
        "subset": SUBSET,
        "subset_qa_count_full": SUBSET_QA_COUNT,
        "adversarial_included": ADVERSARIAL_INCLUDED,
        "sampling": "stratified per question_type",
        "per_type_cap": args.per_type,
        "seed": args.seed,
        "per_type_counts": counts,
        "adapter": "mempalace-daemon",
        "substrate": "postgres (pgvector) + Apache AGE",
        "search_endpoint": args.search_endpoint,
        "ingest_topology": "per-sample wing locomo_<sample_id>, room sessions, sme-rich rendering",
        "backfill_age": (not args.skip_backfill),
        "backfill_info": {k: backfill_info.get(k) for k in
                          ("returncode", "total_drawers", "status")},
        "reader_model": args.reader_model,
        "judge_model": args.judge_model,
        "judge_prompts": "canonical LongMemEval type-specific "
                         "(temporal->off-by-one; adversarial->abstention)",
        "retrieval_n_results": 5,
        "daemon_isolation": "throwaway scratch palace-daemon (localhost) + scratch "
                            "postgres; never touched a production palace (#176)",
        "elapsed_sec": round(dt, 1),
        "timestamp_utc": time.strftime("%FT%TZ", time.gmtime()),
    }
    report = build_report(records, ingest_total, meta=meta)
    pathlib.Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    log(f"DONE in {dt:.1f}s n={len(records)} overall_qa={report['qa_overall']['qa_accuracy']} "
        f"drawer_R@5={report['retrieval']['drawer_recall_at_5']} "
        f"substring_R={report['retrieval']['substring_recall_mean']} wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
