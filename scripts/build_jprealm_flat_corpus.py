#!/usr/bin/env python3
"""Build the information-matched FLAT (Condition A) corpus for the
jp-realm-v0.1 diagnostic corpus (issue #125, Gap 2).

Methodology
-----------
Condition B (the existing reading in
``baselines/jp_realm_v0_1_daemon_age_2026-05-29.json``) is the daemon AGE
search over JP's live familiar palace. Its retrieval is *structural*: the
candidate set is AGE-graph fused, and the context string carries
``[wing/room] source`` headers.

To make Condition A (flat) a valid A/B comparison, it must hold the *same
information* with *no structure*: the same drawer text, but flat-vector
retrieved, with no wing/room labels and no graph fusion. We therefore pull
the drawer text the palace actually holds for each of the 30 questions
(read-only ``GET /search``, ``kind=content`` — the same filter Condition B
used), dedupe into a pool, sanitize for public exposure, and emit a flat
ChromaDB collection that ``FlatBaselineAdapter`` retrieves over.

The pool is the *union* of each question's top-K hits. Flat retrieval then
does its own top-K vector ranking over that pool — the honest flat baseline:
same documents available, no structural routing.

NEVER writes to the palace (familiar#92 pollution). Read-only GET only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent


# --- sanitization ----------------------------------------------------------
#
# These 60 benchmark tokens MUST survive (they are the expected_sources the
# scorer matches on). None are secrets — tool/host/concept names. We strip
# only genuinely sensitive material: private IPs, MACs, secrets, tokens.

# Private/RFC1918 + link-local IPv4. Public IPs are rare in palace text and
# would be content; we still scrub all dotted-quads to be safe.
_IP_RE = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?:/\d{1,2})?\b")
_MAC_RE = re.compile(r"\b(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}\b")
# Long hex/base64-ish blobs that look like keys/tokens (>=32 chars).
_SECRET_BLOB_RE = re.compile(r"\b[A-Za-z0-9+/_-]{32,}={0,2}\b")
# KEY=VALUE / "key": "value" secret-looking assignments.
_SECRET_ASSIGN_RE = re.compile(
    r"(?i)\b(api[_-]?key|secret|token|password|passwd|bearer|"
    r"client[_-]?secret|access[_-]?key)\b\s*[:=]\s*\S+"
)
# All emails are redacted (defense-in-depth). JP's public git-commit email is
# already in every commit, but the corpus is public palace content so we strip
# it here too rather than special-case an allowlist.
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_ALLOWED_EMAILS: set[str] = set()

# Private self-hosted service FQDNs that reveal JP's internal topology and
# are NOT needed to answer any of the 30 benchmark questions. *.realm.watch
# is public-facing (the project itself) and stays; these *.jphe.in service
# subdomains are auth-gated personal infrastructure. Redact the host part
# so the surrounding prose still reads, but the inventory is not disclosed.
_PRIVATE_FQDN_RE = re.compile(
    r"\b(?:navidrome|jellyfin|immich|portfolio|oracle|realmcoin|realmkoin|"
    r"bestiary|auth|coin|dark|nos|clock|chat|deploy|palace)\.jphe\.in\b",
    re.IGNORECASE,
)
# VM/host inventory lines that pair a private host name with an address.
_PRIVATE_HOSTS = re.compile(
    r"\b(?:realmcoin|realmkoin|status-vm|bestiary|dreamspace)\b",
    re.IGNORECASE,
)
# Exact internal endpoint host:port forms (the daemon URL). The bare `disks`
# / `familiar` host words are benchmark tokens and survive; only the precise
# private FQDN[:port] / .lan[:port] endpoint is redacted. Public URLs
# (realm.watch, github, arxiv, etc.) are kept for corpus fidelity. Run before
# the generic blob pass. Scoped to private domains only.
_ENDPOINT_RE = re.compile(
    r"\bhttps?://[A-Za-z0-9.-]*(?:jphe\.in|\.lan|\.local)(?::\d+)?(?:/\S*)?",
    re.IGNORECASE,
)
_HOSTPORT_RE = re.compile(
    r"\b[A-Za-z0-9-]+\.(?:jphe\.in|lan|local)(?::\d+)?\b", re.IGNORECASE
)

# Tokens that must NOT be redacted even if they appear secret-shaped.
# (None of the 32+ char blob matches should hit these, but guard anyway.)
_PRESERVE = set()


def sanitize(text: str) -> str:
    """Strip sensitive material while preserving benchmark-relevant tokens.

    Order matters: assignment patterns first (they carry the secret value),
    then standalone blobs, then network identifiers.
    """
    if not text:
        return text
    out = _SECRET_ASSIGN_RE.sub(
        lambda m: f"{m.group(1)}=<redacted>", text
    )
    out = _MAC_RE.sub("<mac>", out)
    out = _IP_RE.sub("<ip>", out)
    # Private service topology — redact before generic patterns so the host
    # names don't survive as bare words in inventory lists.
    out = _PRIVATE_FQDN_RE.sub("<private-service>.jphe.in", out)
    out = _PRIVATE_HOSTS.sub("<private-host>", out)

    # Internal endpoints. Redact http(s) URLs and bare FQDN[:port] for private
    # domains (jphe.in / .lan / .local) to a placeholder — keeps the bare host
    # word (a benchmark token) but hides the exact endpoint. Public URLs
    # (*.realm.watch, github, arxiv, ...) are not matched and stay intact.
    out = _ENDPOINT_RE.sub("<internal-endpoint>", out)
    out = _HOSTPORT_RE.sub("<internal-host>", out)

    def _email_sub(m: re.Match) -> str:
        e = m.group(0)
        return e if e.lower() in _ALLOWED_EMAILS else "<email>"

    out = _EMAIL_RE.sub(_email_sub, out)

    # Redact long opaque blobs that survived, but keep ones that are clearly
    # benchmark tokens (e.g. hyphenated project names are < 32 chars so safe).
    def _blob_sub(m: re.Match) -> str:
        blob = m.group(0)
        if blob in _PRESERVE:
            return blob
        # Keep if it contains a dot or looks like a path/word — secrets are
        # typically continuous hex/base64 with no separators.
        if "." in blob or "/" in blob:
            return blob
        return "<redacted>"

    out = _SECRET_BLOB_RE.sub(_blob_sub, out)
    return out


# Drop drawers that are SME benchmark run-logs leaked back into the palace
# (self-referential pollution), not jp-realm knowledge. Heuristic markers.
_POLLUTION_MARKERS = (
    "recall=",
    "expected: [",
    "min_hops",
    "tokens=",
    "--questions sme/corpora",
    "elapsed_ms",
)


def is_pollution(text: str) -> bool:
    hits = sum(1 for m in _POLLUTION_MARKERS if m in text)
    return hits >= 2


# --- familiar read-only fetch ---------------------------------------------


def search(api_url: str, api_key: str, q: str, limit: int, kind: str) -> list[dict]:
    params = urllib.parse.urlencode({"q": q, "limit": limit, "kind": kind})
    url = f"{api_url}/search?{params}"
    req = urllib.request.Request(url, method="GET", headers={"X-API-Key": api_key})
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode())
    return body.get("results") or []


def load_env() -> tuple[str, str]:
    env = Path("~/.config/palace-daemon/env").expanduser()
    url, key = None, None
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line.startswith("PALACE_DAEMON_URL="):
                url = line.split("=", 1)[1].strip().strip('"')
            elif line.startswith("PALACE_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"')
    url = os.environ.get("PALACE_DAEMON_URL", url)
    key = os.environ.get("PALACE_API_KEY", key)
    if not url or not key:
        sys.exit("missing PALACE_DAEMON_URL / PALACE_API_KEY")
    return url, key


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--questions",
        default=str(REPO / "sme/corpora/jp_realm_v0_1/questions.yaml"),
    )
    ap.add_argument("--limit", type=int, default=10, help="hits per question")
    ap.add_argument("--kind", default="content")
    ap.add_argument(
        "--out",
        default=str(REPO / "sme/corpora/jp_realm_v0_1/flat_source.jsonl"),
        help="sanitized flat corpus (JSONL: one drawer per line)",
    )
    args = ap.parse_args()

    api_url, api_key = load_env()
    qs = yaml.safe_load(open(args.questions))["questions"]

    pool: dict[str, dict] = {}  # drawer_id -> record
    dropped_pollution = 0
    for q in qs:
        results = search(api_url, api_key, q["text"], args.limit, args.kind)
        for r in results:
            did = r.get("drawer_id") or r.get("id")
            if not did or did in pool:
                continue
            raw = r.get("text", "") or ""
            if is_pollution(raw):
                dropped_pollution += 1
                continue
            clean = sanitize(raw)
            pool[did] = {
                "id": did,
                # Keep ONLY the sanitized text. Deliberately drop wing/room
                # so the flat corpus carries no structural metadata at all.
                "text": clean,
            }

    out_path = Path(args.out)
    with open(out_path, "w") as f:
        for rec in pool.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"questions:           {len(qs)}")
    print(f"unique drawers:      {len(pool)}")
    print(f"dropped (pollution): {dropped_pollution}")
    print(f"wrote:               {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
