#!/bin/bash
# Step 3 chain — FORCED + GROUNDED only (vanilla skipped per JP sign-off
# 2026-05-17 11:18 PDT; partial gemma4 vanilla n=5 preserved at
# git_probes_v2_rlm_2026-05-17_gemma4_e4b_vanilla_n5_partial_6q.json
# with recall=0/6 — confirms gemma4 100% zero-call at this corpus shape).
#
# 8 conditions × ~3-6h = ~24-32h sequential. Each sub-run writes its
# own JSON snapshot; chain skips existing outputs so kill+restart is safe.

set +e

cd ~/Projects/multipass-structural-memory-eval
source venv/bin/activate
source ~/.config/palace-daemon/env
export RLM_API_KEY="ollama"
export RLM_BASE_URL="http://localhost:11434/v1"

PROBES_G=sme/corpora/mempalace_git_probes_v2/questions.yaml
OUT=baselines

mark() { echo; echo "=== $(date +%H:%M:%S) $1 ==="; }

# Forced + grounded only.
# Order: qwen3.5 first (faster + better discipline), then gemma4 (slower).
# Within each model: forced before grounded.
for model in qwen3.5:4b gemma4:e4b; do
  m_safe="${model//[:.]/_}"
  for mode in forced grounded; do
    for n in 5 20; do
      out_path="$OUT/git_probes_v2_rlm_2026-05-17_${m_safe}_${mode}_n${n}.json"
      if [[ -f "$out_path" ]]; then
        mark "SKIP existing $out_path"
        continue
      fi
      cap=180m
      [[ "$model" == "gemma4:e4b" && "$n" == "20" ]] && cap=240m
      mark "STEP3 ${mode} ${model} n=${n} (cap=${cap})"
      RLM_MODEL="$model" timeout $cap sme-eval retrieve --adapter rlm --api-url "$PALACE_DAEMON_URL" \
        --questions "$PROBES_G" --n-results $n --invocation-mode "$mode" \
        --json "$out_path"
    done
  done
done

mark "STEP3 FORCED+GROUNDED CHAIN COMPLETE"
