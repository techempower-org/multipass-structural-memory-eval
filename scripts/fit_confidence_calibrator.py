#!/usr/bin/env python3
"""Fit an isotonic confidence calibrator on our corpus (#105 / mempalace#250).

mempalace PR #167 shipped the confidence-calibration *plumbing* — an
isotonic PAV calibrator with ``Calibrator`` load/save, ``_CALIBRATOR_CACHE``
keyed by ``(path, mtime)``, and ``MEMPALACE_CALIBRATION_PATH`` wiring through
``config.py``/``searcher.py``. What it never shipped was a calibrator *fitted
on real data*: ``search_memories`` only attaches a ``confidence`` field when a
calibration JSON exists on disk, and there is none in production. So today the
raw similarity is the only signal a caller sees, and it is not a probability.

This script closes that gap. It collects ``(similarity, relevant?)`` pairs by
running a labeled probe set through the live daemon's ``/search/age-fused``
endpoint (READ-ONLY — retrieval only, never ingest), fits an isotonic
calibrator with the Pool-Adjacent-Violators algorithm, and reports **ECE** and
**Brier score** before vs. after calibration. The fitted calibrator is emitted
as a JSON artifact in the shape mempalace's ``Calibrator.load`` expects
(``{"x": [...], "y": [...]}`` knot points + metadata), so the fork (mempalace#250)
can commit it and point ``MEMPALACE_CALIBRATION_PATH`` at it.

Posture (per CLAUDE.md): diagnostic delta under a controlled condition, not an
absolute score. The "before" model is the identity map similarity→confidence
(what production does today: raw similarity used as if it were a probability);
the "after" model is the fitted isotonic map. ECE/Brier deltas quantify how
miscalibrated the raw similarity is and how much the PAV fit corrects it.

The PAV / ECE / Brier functions are pure and unit-tested in
``tests/test_fit_confidence_calibrator.py`` (no daemon needed). Only ``main``
and ``collect_pairs`` touch the network.

Usage:
    venv/bin/python scripts/fit_confidence_calibrator.py \\
        --probes baselines/rrf_multi_encoder_age_2026-05-29.json \\
        --endpoint /search/age-fused --limit 10 \\
        --calibrator-out baselines/confidence_calibrator_age_2026-05-29.json \\
        --report-out baselines/confidence_calibration_report_2026-05-29.json

    # Offline: fit from a previously collected pairs file (no daemon):
    venv/bin/python scripts/fit_confidence_calibrator.py \\
        --pairs-in /path/to/pairs.json --calibrator-out ... --report-out ...
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

log = logging.getLogger("fit_calibrator")

DEFAULT_ENV_FILE = "~/.config/palace-daemon/env"


# --------------------------------------------------------------------------- #
# Pure calibration math (unit-tested, no I/O)
# --------------------------------------------------------------------------- #
def pav_fit(scores: list[float], labels: list[int]) -> tuple[list[float], list[float]]:
    """Pool-Adjacent-Violators isotonic regression.

    Given raw ``scores`` (e.g. cosine similarity in [0,1]) and binary
    ``labels`` (1 = relevant), returns ``(x_knots, y_knots)`` — a monotone
    non-decreasing step function mapping score -> calibrated probability.

    Sorts by score, then pools adjacent blocks that violate monotonicity by
    replacing them with their weighted mean. The returned knots are the
    distinct score thresholds and their pooled probabilities; ``apply_pav``
    interpolates between them. This mirrors the ``_pav`` shape mempalace
    PR #167 ships so the artifact loads cleanly via ``Calibrator.load``.
    """
    if not scores:
        return [], []
    pairs = sorted(zip(scores, labels), key=lambda p: p[0])
    # Each block: [sum_label, weight, x_left, x_right].
    # x_left/x_right are the span of raw scores pooled into this block; the
    # fitted probability is flat (sum/weight) across [x_left, x_right] and we
    # interpolate *between* blocks. Tracking both edges (not just the left)
    # is what makes apply_pav agree with sklearn's IsotonicRegression: a block
    # spanning [0.3, 0.7] holds its value out to 0.7, so the next block
    # interpolates from 0.7 rather than from the block's start.
    blocks: list[list[float]] = []
    for x, y in pairs:
        x = float(x)
        # Tied x-values are indistinguishable to a monotone score->prob map:
        # they must collapse into one block carrying their mean, *before* the
        # monotonicity pooling runs. Otherwise an order like (0.9,0),(0.9,1)
        # leaves two blocks at x=0.9 whose means (0, 1) never "violate"
        # (0 < 1), and the function reports a non-monotone duplicate knot.
        if blocks and blocks[-1][3] == x:
            blocks[-1][0] += float(y)
            blocks[-1][1] += 1.0
        else:
            blocks.append([float(y), 1.0, x, x])
        # Merge backwards while the previous block's mean exceeds this one's.
        while len(blocks) >= 2 and (blocks[-2][0] / blocks[-2][1]) > (
            blocks[-1][0] / blocks[-1][1]
        ):
            s2, w2, _xl2, xr2 = blocks.pop()
            s1, w1, xl1, _xr1 = blocks.pop()
            blocks.append([s1 + s2, w1 + w2, xl1, xr2])  # span left..right
    # Emit a knot at each block's left and right edge with the pooled value,
    # so the step function is flat within a block and linear between blocks.
    x_knots: list[float] = []
    y_knots: list[float] = []
    for s, w, xl, xr in blocks:
        prob = s / w
        x_knots.append(xl)
        y_knots.append(prob)
        if xr != xl:
            x_knots.append(xr)
            y_knots.append(prob)
    return x_knots, y_knots


def apply_pav(x_knots: list[float], y_knots: list[float], score: float) -> float:
    """Map a raw score to a calibrated probability via the PAV step function.

    Linear interpolation between knots; clamps below the first / above the
    last knot to the edge probability. Empty knots -> pass score through.
    """
    if not x_knots:
        return float(score)
    if score <= x_knots[0]:
        return float(y_knots[0])
    if score >= x_knots[-1]:
        return float(y_knots[-1])
    # find bracketing knots
    lo = 0
    hi = len(x_knots) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if x_knots[mid] <= score:
            lo = mid
        else:
            hi = mid
    x0, x1 = x_knots[lo], x_knots[hi]
    y0, y1 = y_knots[lo], y_knots[hi]
    if x1 == x0:
        return float(y0)
    frac = (score - x0) / (x1 - x0)
    return float(y0 + frac * (y1 - y0))


def brier_score(probs: list[float], labels: list[int]) -> float:
    """Mean squared error between predicted probability and outcome."""
    if not probs:
        return 0.0
    return sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(probs)


def expected_calibration_error(
    probs: list[float], labels: list[int], *, n_bins: int = 10
) -> float:
    """Expected Calibration Error with equal-width bins over [0,1].

    ECE = sum over bins of (|bin| / N) * |mean_confidence - accuracy|.
    The standard reliability-diagram summary statistic.
    """
    if not probs:
        return 0.0
    n = len(probs)
    ece = 0.0
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        # last bin is closed on the right so prob==1.0 lands somewhere
        idx = [
            i
            for i, p in enumerate(probs)
            if (p >= lo and p < hi) or (b == n_bins - 1 and p == hi)
        ]
        if not idx:
            continue
        conf = sum(probs[i] for i in idx) / len(idx)
        acc = sum(labels[i] for i in idx) / len(idx)
        ece += (len(idx) / n) * abs(conf - acc)
    return ece


# --------------------------------------------------------------------------- #
# Probe loading + relevance labeling
# --------------------------------------------------------------------------- #
def load_probes(path: Path) -> list[dict]:
    """Extract (query, expected) probes from an RRF baseline artifact.

    The 2026-05-29 RRF artifact stores its labeled probe set under
    ``baseline.per_probe`` as ``{"query", "expected"}`` records, where
    ``expected`` is the gold source-file basename. Reusing it keeps the
    calibration set identical to the retrieval-gate set (same corpus, same
    labels), so the two issues share one ground truth.
    """
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "baseline" in data:
        pp = data["baseline"].get("per_probe") or []
        return [
            {"query": r["query"], "expected": r["expected"]}
            for r in pp
            if r.get("query") and r.get("expected")
        ]
    # plain list of {query, expected}
    if isinstance(data, list):
        return [
            {"query": r["query"], "expected": r["expected"]}
            for r in data
            if r.get("query") and r.get("expected")
        ]
    raise ValueError(f"Unrecognized probe file shape: {path}")


def hit_is_relevant(hit: dict, expected: str) -> bool:
    """A hit is relevant iff its source-file basename matches ``expected``.

    Matches the rank-attribution rule the RRF artifact used (basename of
    ``source_file`` == expected filename). Tolerates both the flat
    age-fused shape (``source_file`` top-level) and the nested ``metadata``
    shape from plain ``/search``.
    """
    src = hit.get("source_file") or (hit.get("metadata") or {}).get("source_file") or ""
    return os.path.basename(str(src)) == expected


# --------------------------------------------------------------------------- #
# Daemon retrieval (READ-ONLY)
# --------------------------------------------------------------------------- #
def _parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    except OSError as e:
        log.warning("env file %s unreadable (%s)", path, e)
    return out


def _resolve_daemon(env_file: Optional[str]) -> tuple[str, str]:
    env = _parse_env_file(Path(os.path.expanduser(env_file or DEFAULT_ENV_FILE)))
    url = (
        os.environ.get("PALACE_DAEMON_URL")
        or env.get("PALACE_DAEMON_URL")
        or "http://familiar:8085"
    )
    key = os.environ.get("PALACE_API_KEY") or env.get("PALACE_API_KEY") or ""
    return url.rstrip("/"), key


def collect_pairs(
    probes: list[dict],
    *,
    daemon_url: str,
    api_key: str,
    endpoint: str = "/search/age-fused",
    limit: int = 10,
    score_field: str = "similarity",
    sleep_s: float = 0.0,
) -> list[dict]:
    """Run probes through the daemon, return per-hit (score, relevant) rows.

    READ-ONLY: issues retrieval requests only (GET ``/search`` or POST
    ``/search/age-fused``). Never writes. Every retrieved hit becomes one
    calibration row: its ``score_field`` value (default ``similarity``) and a
    binary relevance label from ``hit_is_relevant``.
    """
    rows: list[dict] = []
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    for i, probe in enumerate(probes):
        q = probe["query"]
        expected = probe["expected"]
        try:
            if endpoint == "/search":
                params = urllib.parse.urlencode({"q": q, "limit": limit})
                req = urllib.request.Request(
                    f"{daemon_url}{endpoint}?{params}", headers=headers, method="GET"
                )
            else:
                payload = json.dumps({"query": q, "limit": limit}).encode()
                req = urllib.request.Request(
                    f"{daemon_url}{endpoint}", data=payload, headers=headers,
                    method="POST",
                )
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode())
        except Exception as e:  # noqa: BLE001 — network is best-effort here
            log.warning("probe %d (%r) failed: %s", i, q[:50], e)
            continue
        results = body.get("results") or []
        for rank, hit in enumerate(results):
            score = hit.get(score_field)
            if score is None:
                continue
            rows.append(
                {
                    "query": q,
                    "expected": expected,
                    "rank": rank + 1,
                    "score": float(score),
                    "relevant": int(hit_is_relevant(hit, expected)),
                }
            )
        if sleep_s:
            time.sleep(sleep_s)
        if (i + 1) % 25 == 0:
            log.info("collected through probe %d/%d (%d rows)", i + 1, len(probes), len(rows))
    return rows


# --------------------------------------------------------------------------- #
# Cross-validated evaluation (honest out-of-sample ECE/Brier)
# --------------------------------------------------------------------------- #
def cross_validated_eval(
    rows: list[dict], *, n_folds: int = 5, n_bins: int = 10, seed: int = 42
) -> dict:
    """Out-of-sample ECE/Brier via k-fold CV (fit on train, score on test).

    In-sample ECE for isotonic regression is misleadingly optimistic — PAV can
    drive the training ECE to ~0 by construction. This pools held-out
    predictions across folds and computes ECE/Brier on those, which is the
    number that reflects how a *deployed* calibrator (fit once, applied to
    unseen scores) would actually behave. The "before" column is the identity
    map on the same held-out scores, for a like-for-like comparison.
    """
    import random as _random

    if len(rows) < n_folds:
        return {"skipped": f"n={len(rows)} < n_folds={n_folds}"}
    rng = _random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    folds: list[list[int]] = [idx[i::n_folds] for i in range(n_folds)]

    oos_raw: list[float] = []
    oos_cal: list[float] = []
    oos_lab: list[int] = []
    for f in range(n_folds):
        test_i = set(folds[f])
        train = [rows[i] for i in idx if i not in test_i]
        test = [rows[i] for i in folds[f]]
        if not train or not test:
            continue
        xk, yk = pav_fit([r["score"] for r in train], [int(r["relevant"]) for r in train])
        for r in test:
            oos_raw.append(r["score"])
            oos_cal.append(apply_pav(xk, yk, r["score"]))
            oos_lab.append(int(r["relevant"]))
    return {
        "n_folds": n_folds,
        "n_oos": len(oos_lab),
        "before_calibration": {
            "ece": expected_calibration_error(oos_raw, oos_lab, n_bins=n_bins),
            "brier": brier_score(oos_raw, oos_lab),
        },
        "after_calibration": {
            "ece": expected_calibration_error(oos_cal, oos_lab, n_bins=n_bins),
            "brier": brier_score(oos_cal, oos_lab),
        },
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def fit_and_report(rows: list[dict], *, n_bins: int = 10) -> dict:
    """Fit PAV on (score, relevant) rows; report ECE/Brier before vs after.

    "Before" = identity map (raw similarity treated as the probability, which
    is exactly what a caller does today with the uncalibrated score).
    "After" = the fitted isotonic map.
    """
    scores = [r["score"] for r in rows]
    labels = [int(r["relevant"]) for r in rows]
    x_knots, y_knots = pav_fit(scores, labels)
    calibrated = [apply_pav(x_knots, y_knots, s) for s in scores]

    before = {
        "ece": expected_calibration_error(scores, labels, n_bins=n_bins),
        "brier": brier_score(scores, labels),
    }
    after = {
        "ece": expected_calibration_error(calibrated, labels, n_bins=n_bins),
        "brier": brier_score(calibrated, labels),
    }
    cv = cross_validated_eval(rows, n_folds=5, n_bins=n_bins)
    return {
        "n_pairs": len(rows),
        "n_relevant": sum(labels),
        "base_rate": (sum(labels) / len(labels)) if labels else 0.0,
        "n_bins": n_bins,
        "in_sample": {
            "before_calibration": before,
            "after_calibration": after,
            "note": "in-sample ECE for isotonic is optimistic (PAV fits the "
                    "eval set); see cross_validated for the honest number.",
        },
        "cross_validated": cv,
        # Headline before/after = the honest cross-validated numbers when
        # available, falling back to in-sample for tiny sets.
        "before_calibration": cv.get("before_calibration", before),
        "after_calibration": cv.get("after_calibration", after),
        "delta_ece": (
            cv["after_calibration"]["ece"] - cv["before_calibration"]["ece"]
            if "after_calibration" in cv else after["ece"] - before["ece"]
        ),
        "delta_brier": (
            cv["after_calibration"]["brier"] - cv["before_calibration"]["brier"]
            if "after_calibration" in cv else after["brier"] - before["brier"]
        ),
        "calibrator": {"x": x_knots, "y": y_knots},
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--probes", type=Path, help="RRF baseline JSON with per_probe labels.")
    src.add_argument("--pairs-in", type=Path, help="Pre-collected (score,relevant) rows JSON (offline).")
    p.add_argument("--endpoint", default="/search/age-fused",
                   help="Retrieval endpoint (default /search/age-fused).")
    p.add_argument("--limit", type=int, default=10, help="Hits per probe (default 10).")
    p.add_argument("--score-field", default="similarity",
                   help="Hit field used as the raw score (default similarity).")
    p.add_argument("--n-bins", type=int, default=10, help="ECE bins (default 10).")
    p.add_argument("--env-file", default=None, help="palace-daemon env file.")
    p.add_argument("--sleep", type=float, default=0.0, help="Seconds between probes.")
    p.add_argument("--pairs-out", type=Path, default=None,
                   help="Optional: write collected pairs here (for reuse/offline).")
    p.add_argument("--calibrator-out", type=Path, required=True,
                   help="Fitted calibrator JSON (Calibrator.load shape).")
    p.add_argument("--report-out", type=Path, required=True,
                   help="ECE/Brier before-after report JSON.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.pairs_in:
        rows = json.loads(args.pairs_in.read_text())
        log.info("loaded %d pre-collected pairs from %s", len(rows), args.pairs_in)
        source_desc = str(args.pairs_in)
    else:
        probes = load_probes(args.probes)
        log.info("loaded %d labeled probes from %s", len(probes), args.probes)
        daemon_url, api_key = _resolve_daemon(args.env_file)
        if not api_key:
            log.error("no PALACE_API_KEY resolved; cannot reach daemon.")
            return 2
        log.info("collecting (score,relevant) pairs from %s%s (READ-ONLY)",
                 daemon_url, args.endpoint)
        rows = collect_pairs(
            probes, daemon_url=daemon_url, api_key=api_key,
            endpoint=args.endpoint, limit=args.limit,
            score_field=args.score_field, sleep_s=args.sleep,
        )
        source_desc = f"{daemon_url}{args.endpoint} (limit={args.limit})"

    if not rows:
        log.error("no calibration pairs collected; aborting.")
        return 3

    if args.pairs_out:
        args.pairs_out.parent.mkdir(parents=True, exist_ok=True)
        args.pairs_out.write_text(json.dumps(rows, indent=2))
        log.info("wrote %d pairs -> %s", len(rows), args.pairs_out)

    result = fit_and_report(rows, n_bins=args.n_bins)

    # Calibrator artifact in Calibrator.load shape (knots + provenance).
    calibrator_artifact = {
        "method": "isotonic_pav",
        "score_field": args.score_field,
        "fitted_on": source_desc,
        "n_pairs": result["n_pairs"],
        "base_rate": result["base_rate"],
        "x": result["calibrator"]["x"],
        "y": result["calibrator"]["y"],
    }
    args.calibrator_out.parent.mkdir(parents=True, exist_ok=True)
    args.calibrator_out.write_text(json.dumps(calibrator_artifact, indent=2))
    log.info("wrote calibrator -> %s (%d knots)",
             args.calibrator_out, len(calibrator_artifact["x"]))

    report = {
        "experiment": "isotonic confidence calibration fit (#105 / mempalace#250)",
        "posture": "controlled-condition delta; before=identity(raw similarity), after=PAV isotonic",
        "source": source_desc,
        "score_field": args.score_field,
        **{k: v for k, v in result.items() if k != "calibrator"},
        "calibrator_artifact": str(args.calibrator_out),
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2))
    log.info("wrote report -> %s", args.report_out)

    b, a = result["before_calibration"], result["after_calibration"]
    log.info(
        "ECE  %.4f -> %.4f (Δ%+.4f)   Brier %.4f -> %.4f (Δ%+.4f)   n=%d base_rate=%.3f",
        b["ece"], a["ece"], result["delta_ece"],
        b["brier"], a["brier"], result["delta_brier"],
        result["n_pairs"], result["base_rate"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
