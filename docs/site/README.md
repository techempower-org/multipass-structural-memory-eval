# docs/site — SME benchmark results landing page

Single self-contained `index.html` (no build step, no external assets except Google Fonts). Carries the published benchmark readings from this fork: Tau2 prediction (2026-05-15), the first LongMemEval-S 500Q E2E QA result against palace-daemon (2026-05-28, techempower-org/multipass-structural-memory-eval#44), and placeholders for the in-flight #45 (age-fused) and #46 (Familiar) legs.

Aesthetic mirrors the MemPalace landing page (`MemPalace/mempalace`/`landing/index.html`) — same typography stack, same prism-blue palette, light-mode added via `prefers-color-scheme`.

## Serve locally

```bash
python3 -m http.server 8000 --directory docs/site
# open http://127.0.0.1:8000/
```

## Live on GitHub Pages

Source: `main` branch, `/docs` folder. Served at:

```
https://techempower-org.github.io/multipass-structural-memory-eval/site/
```

## Updating the bench cards

When `baselines/longmemeval_age_fused_2026-05-28.json` or `baselines/longmemeval_familiar_2026-05-28.json` land, replace the `<span class="bench-pill running">` / `<span class="bench-pill queued">` chips with `<span class="bench-pill live">` and fill the matching `bench-numbers` block following the #44 card's structure.
