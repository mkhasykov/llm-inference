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

## Run a benchmark

```bash
# vanilla HF generation (canonical baseline uses --no-cache)
python scripts/benchmark_baseline.py --limit 80 --no-cache

# explicit prefill+decode loop with HF's DynamicCache
python scripts/manual_kv_loop.py --limit 80
```

Default model is `Qwen/Qwen2.5-1.5B-Instruct` (auto-downloaded on first run). MT-Bench prompts are in `data/mt_bench/question.jsonl`.

Key flags (both scripts):

- `--model <hf-id>` — alternate model
- `--limit <N>` — number of prompts (default 5; full set is 80)
- `--max-new-tokens <N>` — generation length (default 256)

Each run writes two files into `results/`:

- `<run_id>.jsonl` — per-prompt metrics (TTFT, decode latency, peak VRAM, …). Local debug only, **not committed**.
- `<run_id>.json` — per-run summary (aggregates, model, gen settings, git commit). **Committed.**

To regenerate a summary from a JSONL (e.g. after a schema change):

```bash
python scripts/jsonl_to_summary.py results/<run_id>.jsonl
```

## Results

Qwen2.5-1.5B-Instruct, RTX 3090, bf16, 80 MT-Bench prompts, max_new_tokens=256, greedy.

| run                                | tokens/sec mean | TTFT mean (ms) | peak VRAM | KV cache max |
| ---------------------------------- | --------------: | -------------: | --------: | -----------: |
| baseline, `--no-cache`             |           47.10 |          20.71 |   3.14 GB |            — |
| manual KV loop (`DynamicCache`)    |           61.82 |          19.57 |   3.23 GB |      17.6 MB |

Full summary JSON per run: [`results/`](results/).

## Layout

```
data/mt_bench/question.jsonl    # 80 MT-Bench prompts
scripts/benchmark_baseline.py   # vanilla HF baseline
scripts/manual_kv_loop.py       # explicit prefill+decode with DynamicCache
scripts/summary.py              # shared summary builder (imported by the above)
scripts/jsonl_to_summary.py     # regenerate summary from per-prompt JSONL
results/                        # per-run summary JSON (per-prompt JSONL gitignored)
```
