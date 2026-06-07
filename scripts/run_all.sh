#!/usr/bin/env bash
# Full experiment run: 1.5B (speed+batch) then 7B (everything), then aggregate+plots.
# Quality is measured on 7B only (it is a property of the weights). no-cache on
# 7B is run at reduced scope (quadratic baseline would otherwise take hours).
# Each step uses `|| true` so a failed cell/step never aborts the whole run.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
M15=Qwen/Qwen2.5-1.5B-Instruct
M7=Qwen/Qwen2.5-7B-Instruct
D15=results/qwen1.5b
D7=results/qwen7b

echo "===== [1/5] 1.5B speed+batch (no quality) ====="
$PY scripts/run_matrix.py --model "$M15" --limit 40 --repeats 3 --max-new-tokens 256 \
    --skip-quality --out-dir "$D15" || true

echo "===== [2/5] 7B speed+batch+quality (no-cache excluded) ====="
$PY scripts/run_matrix.py --model "$M7" --limit 40 --repeats 3 --max-new-tokens 256 \
    --skip-cells baseline_nocache --out-dir "$D7" || true

echo "===== [3/5] 7B no-cache (reduced: 10 prompts x1) ====="
$PY scripts/run_matrix.py --model "$M7" --only baseline_nocache --limit 10 --repeats 1 \
    --max-new-tokens 256 --out-dir "$D7" || true

echo "===== [4/5] aggregate ====="
$PY scripts/aggregate.py --results-dir "$D15" --out "$D15/summary_table" || true
$PY scripts/aggregate.py --results-dir "$D7"  --out "$D7/summary_table"  || true

echo "===== [5/5] plots ====="
$PY scripts/plots.py --results-dir "$D15" --out-dir "$D15/plots" || true
$PY scripts/plots.py --results-dir "$D7"  --out-dir "$D7/plots"  || true

echo "===== DONE ====="
