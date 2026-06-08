#!/usr/bin/env bash
# Densify the batch sweep: add sizes {2,8,32} for bf16/nf4/awq on both models
# (existing runs already cover {1,4,16}), then re-aggregate + re-plot.
# bf16 at batch=32 on 7B may OOM -> `|| true` keeps going; a missing point just
# marks where that format hits the VRAM ceiling.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
for MT in "Qwen/Qwen2.5-1.5B-Instruct:qwen1.5b" "Qwen/Qwen2.5-7B-Instruct:qwen7b"; do
  model="${MT%%:*}"; tag="${MT##*:}"
  for B in 2 8 32; do
    for Q in none nf4 awq; do
      echo "===== $tag  batch=$B  quant=$Q ====="
      $PY scripts/benchmark_batch.py --model "$model" --batch-size "$B" --quant "$Q" \
          --limit 40 --repeats 3 --max-new-tokens 256 --out-dir "results/$tag" || true
    done
  done
done
echo "===== re-aggregate + re-plot ====="
for tag in qwen1.5b qwen7b; do
  $PY scripts/aggregate.py --results-dir "results/$tag" --out "results/$tag/summary_table" || true
  $PY scripts/plots.py     --results-dir "results/$tag" --out-dir "results/$tag/plots" || true
done
echo "===== DONE ====="
