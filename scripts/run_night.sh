#!/usr/bin/env bash
# Overnight bonus experiments (cell-timeout guards every cell against hangs):
#   B1 generation-length sweep (cache vs no-cache at 128/256/512/1024)
#   B2 vanilla speculative decoding on Qwen2.5-3B (vocab-matched 0.5B draft)
#   B3 prompt-lookup (n-gram) decoding on 7B
# Each step uses `|| true`; one failure never aborts the run.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
TIMEOUT=5400   # 90 min per cell safety net

echo "===== B1: generation-length sweep (1.5B) ====="
for L in 128 256 512 1024; do
  $PY scripts/run_matrix.py --model Qwen/Qwen2.5-1.5B-Instruct \
      --only baseline_cache baseline_nocache --max-new-tokens "$L" --limit 10 --repeats 1 \
      --out-dir results/length_sweep/qwen1.5b --cell-timeout $TIMEOUT || true
done

echo "===== B1: generation-length sweep (7B) ====="
for L in 128 256 512 1024; do
  $PY scripts/run_matrix.py --model Qwen/Qwen2.5-7B-Instruct \
      --only baseline_cache baseline_nocache --max-new-tokens "$L" --limit 8 --repeats 1 \
      --out-dir results/length_sweep/qwen7b --cell-timeout $TIMEOUT || true
done

echo "===== B2: vanilla spec on Qwen2.5-3B (vocab-matched draft) ====="
$PY scripts/run_matrix.py --model Qwen/Qwen2.5-3B-Instruct \
    --only baseline_cache manual_kv spec --limit 40 --repeats 3 --max-new-tokens 256 \
    --out-dir results/qwen3b --cell-timeout $TIMEOUT || true
$PY scripts/aggregate.py --results-dir results/qwen3b --out results/qwen3b/summary_table || true
$PY scripts/plots.py     --results-dir results/qwen3b --out-dir results/qwen3b/plots || true

echo "===== B3: prompt-lookup (n-gram) on 7B ====="
$PY scripts/benchmark_spec.py --model Qwen/Qwen2.5-7B-Instruct --prompt-lookup-tokens 10 \
    --limit 40 --repeats 3 --max-new-tokens 256 --fixed-length --out-dir results/qwen7b || true
$PY scripts/aggregate.py --results-dir results/qwen7b --out results/qwen7b/summary_table || true
$PY scripts/plots.py     --results-dir results/qwen7b --out-dir results/qwen7b/plots || true

echo "===== NIGHT RUN DONE ====="
