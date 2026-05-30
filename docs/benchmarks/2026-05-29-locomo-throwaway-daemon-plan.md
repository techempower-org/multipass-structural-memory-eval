# LoCoMo on mempalace — throwaway-daemon provisioning plan (#176)

**Date:** 2026-05-29
**Author:** Cassia (SME dream-team)
**Status:** PLAN ONLY — not executed. Scoping deliverable for
`M0nkeyFl0wer/multipass-structural-memory-eval#176`.

## Why this is a plan, not a run

The mempalace adapter that scores well on LongMemEval is the **palace-daemon**
HTTP adapter (`sme/adapters/mempalace_daemon.py`), and the only daemon deployed
in the homelab is **prod familiar** (`http://familiar.jphe.in:8085`,
`10.0.6.124`) — 370K drawers, a backfilled AGE knowledge graph (~274K entities,
~11K triples), serving JP's real memory. Running mempalace-on-LoCoMo means
*writing* ~300 LoCoMo session drawers into a palace and letting the daemon's
KG-extract worker mint entities/relationships from them. Doing that against
familiar would:

1. **Pollute the live KG** with LoCoMo personas (Caroline, Melanie, …) and
   their fabricated facts, which then surface in JP's real recall and in every
   future Cat-8 ontology reading.
2. **Contaminate retrieval** — 300 dialogue drawers in the `references` room
   shift BM25/vector neighbourhoods for unrelated queries.
3. Be **near-impossible to fully reverse** — deleting the drawers leaves
   orphaned AGE vertices/edges and embeddings unless backfill is re-run.

The daemon adapter is also **diagnostic-only by contract**:
`MemPalaceDaemonAdapter.ingest_corpus()` raises `NotImplementedError`
("To seed a test palace, use the daemon's /memory POST endpoint or the
mempalace CLI directly"). So seeding is an explicit, out-of-adapter step
regardless — which is exactly the seam where we point it at a *throwaway*
stack instead of familiar.

**Constitutional check (CLAUDE.md):** SME "stays lightweight and locally
runnable (no server hosting required)." A throwaway daemon is opt-in
infrastructure for ONE diagnostic comparison (mempalace-vs-flat on LoCoMo),
not a standing dependency. The plan keeps it ephemeral and host-local.

## Topology to reproduce (from the palace-daemon repo)

The prod stack is two Docker services (`~/Projects/palace-daemon/`):

- **`mempalace-db`** (`mempalace-db/docker-compose.yml`) — Postgres 16 +
  `pgvector` + Apache AGE. Prod binds host `:5433`, data dir
  `/var/lib/mempalace-db`, DB `mempalace_2026_05_13`, user `palace`,
  `mem_limit: 6g`, `shared_preload_libraries=age`. `init.sql` just does
  `CREATE EXTENSION vector; CREATE EXTENSION age;`.
- **`palace-daemon`** (`docker-compose.yml`) — FastAPI on `:8085`, bind-mounts
  a palace dir at `/palace`, env `PALACE_API_KEY`, `PALACE_MAX_CONCURRENCY`.

The daemon resolves its DB via `MEMPALACE_POSTGRES_DSN`
(`postgres.py::postgres_dsn`, env override first), and its palace files via
`PALACE_PATH`. **Both are fully env-overridable** — which is the entire
isolation mechanism: a throwaway stack just needs a distinct DSN (distinct DB
name + port + data dir) and a distinct palace path, and it physically cannot
touch familiar's DB or files.

### ⚠️ AGE graph-name isolation (verify before running)

AGE namespaces its graph by a **graph name**, not by Postgres database, and
MemPalace historically uses a fixed graph name. A throwaway stack on its **own
Postgres instance / own data dir** is isolated regardless (the graph lives
inside that instance's catalog). Do **not** shortcut to "same Postgres server,
different `POSTGRES_DB`" unless you have confirmed the AGE graph name is also
namespaced per-DB — otherwise two DBs could collide on one `ag_catalog` graph.
**The plan below uses a fully separate DB container + data dir to sidestep this
entirely.** This is the single most important isolation invariant.

## Provisioning steps (host-local, throwaway)

Run on the host that has Docker + the LoCoMo corpus (katana for dev, or
familiar if you want production-like embeddings — but a *second* stack, never
the live one). Pick non-colliding ports/paths/DB-name up front:

| knob | prod | throwaway |
|---|---|---|
| DB host port | `5433` | `5434` |
| DB data dir | `/var/lib/mempalace-db` | `/var/lib/mempalace-db-locomo` (or a tmp dir) |
| `POSTGRES_DB` | `mempalace_2026_05_13` | `mempalace_locomo_scratch` |
| daemon port | `8085` | `8086` |
| `PALACE_PATH` | familiar's palace | `~/.mempalace/locomo-scratch` |
| container names | `mempalace-db` / `palace-daemon` | `mempalace-db-locomo` / `palace-daemon-locomo` |

1. **DB container.** Copy `mempalace-db/` compose to a scratch override (or use
   `-p locomo-scratch` + env overrides) with the throwaway port/data-dir/DB-name
   above. `MEMPALACE_DB_PASSWORD` from a fresh `openssl rand -hex 16` (do NOT
   reuse the prod secret). `docker compose -p locomo-scratch up -d`. Wait for
   `pg_isready`. The `init.sql` creates the `vector` + `age` extensions; AGE's
   graph is created lazily by MemPalace on first write.
2. **Daemon container.** Bring up `palace-daemon` with
   `MEMPALACE_POSTGRES_DSN=postgresql://palace:<pw>@127.0.0.1:5434/mempalace_locomo_scratch`,
   `MEMPALACE_BACKEND=postgres`, `PALACE_PATH=~/.mempalace/locomo-scratch`,
   `PALACE_PORT=8086`, a fresh `PALACE_API_KEY` (`openssl rand -hex 32`).
   Verify `GET :8086/health` (no auth) and a `GET :8086/status/fast`
   (X-API-Key) returns **0 drawers** — proof the stack is empty and isolated.
3. **Seed LoCoMo.** For each of the 10 conversations, materialize the per-sample
   vault (the loader already does this:
   `sme.corpora.locomo.materialize_sme_corpus`, one `D<N>.md` per session) and
   `POST /memory {content, wing, room}` one drawer per session into a dedicated
   wing, e.g. `wing=locomo_<sample_id>`, `room=references`. ~10 samples ×
   ~20–35 sessions ≈ **~280 drawers** total. Wing-per-sample keeps each
   conversation's haystack scoped so a query for conv-26 can be wing-filtered to
   conv-26 (mirrors the per-sample ingest topology the flat run already uses).
   After seeding, `POST /flush` then `POST /backfill-age` and poll
   `GET /backfill-age/status` until `in_progress: false` so the KG is built
   before querying.
4. **Run the harness.** Point the daemon adapter at the throwaway URL/key
   (`PALACE_DAEMON_URL=http://127.0.0.1:8086`, `PALACE_API_KEY=<scratch key>`,
   or an explicit `--api-url`/`--api-key`), then run the **same** LoCoMo path
   used for the flat number:

   ```
   ./venv/bin/python scripts/cross_validate_longmemeval.py \
     --dataset sme/corpora/locomo/data/locomo10.json \
     --corpus locomo --adapter mempalace \
     --reader-model gpt-5.3-chat --judge-model gpt-5.3-chat \
     --out baselines/locomo10_mempalace_e2e_<date>.json
   ```

   Per-sample wing filtering: the daemon adapter accepts a `wing=` query param,
   so the harness must pass `wing=locomo_<sample_id>` per question (a small
   adapter-factory wrapper, or seed all sessions into one wing and rely on
   semantic ranking — decide and DISCLOSE which, since cross-sample leakage
   changes the number). **Recommended: wing-per-sample + wing-filtered query**,
   to match the flat adapter's per-sample-vault isolation exactly so the
   mempalace-vs-flat delta is attributable to the substrate, not the scoping.
5. **Tear down.** `docker compose -p locomo-scratch down -v` (the `-v` drops the
   throwaway volume), `rm -rf /var/lib/mempalace-db-locomo ~/.mempalace/locomo-scratch`,
   and revoke the scratch API key. Nothing touched familiar.

## Comparability contract (must be disclosed with any number)

- Same pinned subset as the flat row: **LoCoMo-10, n=1986, adversarial
  included** (`SUBSET=locomo10`, `SUBSET_QA_COUNT=1986`,
  `ADVERSARIAL_INCLUDED=True`).
- Same reader + judge as the flat E2E row (gpt-5.3-chat + canonical
  type-specific prompts; `temporal` → off-by-one template; adversarial →
  abstention) so the **only** changed variable is flat-ChromaDB vs
  mempalace-daemon retrieval.
- State the seeding topology (wing-per-sample, wing-filtered query) and the
  backfill-completion gate, because both materially affect retrieval.
- State the embedding model the daemon uses (it embeds server-side via
  `/embed`); if it differs from the flat adapter's ChromaDB encoder, that is a
  **confound to disclose**, not silently absorb — same lesson as the
  LongMemEval substrate-parity work.

## Risks / gotchas

- **Never run a second daemon or the mempalace CLI against familiar's palace
  path** while the live daemon is up (compose comment warns of this — single
  writer). The throwaway `PALACE_PATH` must be a NEW directory.
- **`.bench-active.lock` does not gate an in-flight mine** (known: SME memory
  `bench_active_lock_doesnt_gate_inflight_mine`); the throwaway stack avoids
  this entirely by not sharing the daemon, but don't assume the lock protects
  you.
- Host RAM: prod `mempalace-db` is capped at 6g and familiar has 15g total with
  llama-server etc. resident. A throwaway DB on familiar competes for that RAM —
  prefer katana, or set a smaller `mem_limit` (e.g. 2g) for the scratch DB since
  a 280-drawer corpus needs nothing like the prod working set.
- `GET /stats` and `mempalace_status`/`mempalace_kg_stats` via `/mcp` are flaky
  on the daemon (hang signatures, palace-daemon#49); use `GET /status/fast` for
  drawer counts and `mempalace_graph_stats` (reliable) for KG edge counts when
  verifying the seed.
