#!/usr/bin/env bash
# Re-measure 4-bit AWQ/GPTQ with the optimized Marlin kernels (needs the CUDA
# toolkit installed at /usr/local/cuda-12.8). New *_marlin configs sit next to
# the Triton ones; existing results are untouched.
cd "$(dirname "$0")/.."
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
PY=.venv/bin/python
for MT in "Qwen/Qwen2.5-1.5B-Instruct:qwen1.5b" "Qwen/Qwen2.5-7B-Instruct:qwen7b"; do
  model="${MT%%:*}"; tag="${MT##*:}"
  for Q in awq gptq-int4; do
    echo "===== $tag  $Q + Marlin ====="
    $PY scripts/benchmark_quant.py --model "$model" --quant "$Q" --marlin \
        --limit 40 --repeats 3 --max-new-tokens 256 --fixed-length --out-dir "results/$tag" || true
  done
done
echo "===== re-aggregate + re-plot ====="
for tag in qwen1.5b qwen7b; do
  $PY scripts/aggregate.py     --results-dir "results/$tag" --out "results/$tag/summary_table" || true
  $PY scripts/plots.py         --results-dir "results/$tag" --out-dir "results/$tag/plots" || true
  $PY scripts/plots_methods.py --results-dir "results/$tag" --length-dir "results/length_sweep/$tag" --out-dir "results/$tag/plots" || true
done
echo "===== DONE ====="
