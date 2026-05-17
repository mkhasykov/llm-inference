# llm-inference

Personal benchmarking harness for local LLM inference on a single RTX 3090 (24GB), WSL2 Ubuntu.

The plan is to start from vanilla HuggingFace generation and incrementally add optimizations — explicit KV-cache, quantization, speculative decoding — measuring each against the same set of MT-Bench prompts.

See [`CLAUDE.md`](CLAUDE.md) for the project goal and constraints.

## Setup

```bash
uv sync
source .venv/bin/activate
```

Requires CUDA-capable GPU. The torch wheel is pinned to the `pytorch-cu128` index (see `pyproject.toml`).

## Run the baseline

```bash
python scripts/benchmark_baseline.py --limit 5
```

Default model is `Qwen/Qwen2.5-1.5B-Instruct` (auto-downloaded on first run). MT-Bench prompts are in `data/mt_bench/question.jsonl`. Results go to `results/baseline_<UTC-timestamp>.jsonl`.

Key flags:

- `--model <hf-id>` — alternate model
- `--limit <N>` — number of prompts (default 5; full set is 80)
- `--max-new-tokens <N>` — generation length (default 256)

The script does one warmup pass before measurement to absorb CUDA kernel autotune overhead, then records per-token CUDA events for TTFT, decode throughput, and tail latency.

## Layout

```
data/mt_bench/question.jsonl    # 80 MT-Bench prompts
scripts/benchmark_baseline.py   # vanilla HF baseline
results/                        # JSONL outputs (most are gitignored)
```

A canonical baseline run is committed under `results/` for reference; everyday runs are gitignored.
