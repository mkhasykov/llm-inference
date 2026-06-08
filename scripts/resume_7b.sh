#!/usr/bin/env bash
# Resume the 7B run after an interruption: skip the cells already on disk,
# run the rest + the reduced no-cache, then aggregate + plot both models.
cd "$(dirname "$0")/.."
export HF_HUB_DISABLE_XET=1  # Xet stalled the GPTQ-Int4-7B download; use plain HTTPS
PY=.venv/bin/python
M7=Qwen/Qwen2.5-7B-Instruct
D7=results/qwen7b
# cells already completed (present in results/qwen7b)
DONE="baseline_nocache baseline_cache manual_kv quant_int8 quant_nf4 quant_fp4 quant_awq"

echo "===== 7B remaining: speed+batch+quality (skipping done) ====="
$PY scripts/run_matrix.py --model "$M7" --limit 40 --repeats 3 --max-new-tokens 256 \
    --out-dir "$D7" --skip-cells $DONE || true

echo "===== 7B no-cache (reduced: 10 prompts x1) ====="
$PY scripts/run_matrix.py --model "$M7" --only baseline_nocache --limit 10 --repeats 1 \
    --max-new-tokens 256 --out-dir "$D7" || true

echo "===== aggregate + plots (both models) ====="
$PY scripts/aggregate.py --results-dir results/qwen1.5b --out results/qwen1.5b/summary_table || true
$PY scripts/plots.py     --results-dir results/qwen1.5b --out-dir results/qwen1.5b/plots || true
$PY scripts/aggregate.py --results-dir "$D7" --out "$D7/summary_table" || true
$PY scripts/plots.py     --results-dir "$D7" --out-dir "$D7/plots" || true
echo "===== DONE ====="
