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

# same loop, our own pre-allocated cache (no torch.cat in decode)
python scripts/static_kv_loop.py --limit 80

# weight-quantized generation via bitsandbytes (KV-cache on), nf4 / fp4 / int8
python scripts/benchmark_quant.py --limit 80 --quant nf4

# run the whole reference matrix (one subprocess per config, fresh GPU each).
# --quality here runs a perplexity sweep ONCE per weight format, not per cell.
python scripts/run_matrix.py --limit 80 --repeats 3 --quality --quality-max-tokens 0

# perplexity for a single (model, format) — quality depends only on these
python scripts/eval_quality.py --quant nf4
```

Default model is `Qwen/Qwen2.5-1.5B-Instruct` (auto-downloaded on first run). MT-Bench prompts are in `data/mt_bench/question.jsonl`.

Key flags (all benchmark scripts):

- `--model <hf-id>` — alternate model
- `--limit <N>` — number of prompts (default 5; full set is 80)
- `--max-new-tokens <N>` — generation length (default 256)
- `--repeats <N>` — generations per prompt for run-to-run variance (default 1)
- `--quality` — also compute WikiText-2 perplexity for this model/config
  (`--quality-max-tokens <N>`, 0 = full test split)

`benchmark_quant.py` additionally takes `--quant {int8,nf4,fp4}` (default `nf4`). It reuses the baseline generate path, so the only variable is the quantized weights.

`--repeats` and `--quality` give the rigor needed to compare methods: timing
metrics are reported as `mean ± std` (run-to-run jitter), and perplexity
captures the quality cost of lossy methods (quantization is not free).

Quality is a property of `(model, weight format)` only — lossless methods
(KV-cache, no-cache, speculative decoding) all share the base model's
perplexity. So it is measured once per format via `eval_quality.py`
(`run_matrix.py --quality` sweeps formats automatically) rather than recomputed
for every speed run; report tooling joins it back by `(model, format)`.

Each run writes two files into `results/`:

- `<run_id>.jsonl` — per-prompt metrics (TTFT, decode latency, peak VRAM, …). Local debug only, **not committed**.
- `<run_id>.json` — per-run summary (aggregates, model, gen settings, git commit). **Committed.**

To regenerate a summary from a JSONL (e.g. after a schema change):

```bash
python scripts/jsonl_to_summary.py results/<run_id>.jsonl
```

## Roofline analysis

Single-stream decode reads the whole weight set from VRAM per token, so it is
memory-bandwidth-bound: `tokens/sec_max = bandwidth / weight_bytes`. This
offline tool (no GPU needed) reports, per run, that ceiling, the achieved
fraction (MBU), and the arithmetic-intensity regime:

```bash
python scripts/roofline.py results/*.json
```

It turns raw tokens/sec into "what fraction of the only resource that matters
did we use", and shows what each method moves: e.g. nf4 cuts `weight_bytes`
~3× so the ceiling rises ~3×, but if MBU collapses the speedup wasn't realized
(dequant overhead). GPU specs are matched by device name; override with
`--bandwidth-gb-s` / `--peak-fp16-tflops`.

## Results

Qwen2.5-1.5B-Instruct, RTX 3090, 80 MT-Bench prompts, max_new_tokens=256, greedy, **repeats=3**.
Quality = WikiText-2 (raw) test perplexity, lower is better. Perplexity is a property of the
weight format only, so the three bf16 (lossless) rows share the same number.

| config                | tokens/sec (mean ± std) | peak VRAM | perplexity | ceiling tok/s | MBU   |
| --------------------- | ----------------------: | --------: | ---------: | ------------: | ----: |
| baseline, `--no-cache`|              33.6 ± 3.8 |   3.14 GB |      9.159 |           303 | 11.1% |
| manual_kv (`Dynamic`) |              52.2 ± 4.3 |   3.23 GB |      9.159 |           303 | 17.2% |
| static_kv (`Prealloc`)|              52.8 ± 2.8 |   3.23 GB |      9.159 |           303 | 17.4% |
| quant int8            |              11.2 ± 0.5 |   1.87 GB |      9.209 |           527 |  2.1% |
| quant nf4             |              44.5 ± 2.5 |   1.22 GB |      9.884 |           834 |  5.3% |
| quant fp4             |              44.5 ± 2.2 |   1.22 GB |     10.825 |           834 |  5.3% |

Findings (greedy, batch=1, single-stream — i.e. per-request latency):

- **KV-cache is the clear win**: +55% tok/s (33.6 → 52) at zero quality cost.
- **Pre-allocated vs dynamic cache is a tie** here (52.8 vs 52.2, within run-to-run
  std) — the earlier apparent slowdown washes out at full scale. The pre-allocated
  buffer returns a **non-contiguous** view (`buffer[:, :, :len, :]`), which can push
  PyTorch SDPA onto slower kernels; its real payoff is downstream (static shapes for
  CUDA graphs, KV quantization, paged attention), not raw batch-1 speed.
- **Quantization trades VRAM for throughput, not the reverse**: 4-bit cuts VRAM ~2.6×
  but is slower than bf16+cache; int8 is near-lossless in quality yet the slowest.
  The roofline explains it — nf4 lifts the ceiling ~3× (834 vs 303) but MBU collapses
  to ~5% (dequant-on-the-fly), so the headroom goes unused.
- **NF4 beats plain FP4** at the same 4-bit budget (9.88 vs 10.83 perplexity) —
  the distribution-aware format earns its keep.
- **Everything is memory-bound** (intensity 1–4 ≪ ridge ~76 FLOP/byte) and MBU is only
  2–17%, so there is large headroom that single-stream decode cannot reach — the
  motivation for the batch-size axis.

Full summary JSON per run: [`results/`](results/). Regenerate the roofline table with
`python scripts/roofline.py results/*.json`.

## Layout

```
data/mt_bench/question.jsonl    # 80 MT-Bench prompts
src/llm_inference/              # shared harness (scripts are thin wrappers over this)
  data.py                       #   dataset loading + chat-template prompt
  modeling.py                   #   model/tokenizer loading, eos ids, dtype
  timing.py                     #   CudaEventStreamer + begin/finish_measure (metrics)
  decode.py                     #   explicit prefill+decode loop (manual-cache scripts)
  caches.py                     #   PreallocatedKVCache (Cache subclass)
  summary.py                    #   repeat aggregation + per-run summary (mean ± std)
  quality.py                    #   WikiText-2 perplexity
  cli.py                        #   shared CLI args, env checks, optional quality eval
  roofline.py                   #   bandwidth ceiling, MBU, intensity regime
  runner.py                     #   per-prompt loop: repeats → aggregate → write
scripts/benchmark_baseline.py   # vanilla HF generate (--no-cache for the floor)
scripts/manual_kv_loop.py       # explicit prefill+decode with DynamicCache
scripts/static_kv_loop.py       # same loop with our PreallocatedKVCache
scripts/benchmark_quant.py      # weight-quantized generate (bitsandbytes int8/nf4/fp4)
scripts/run_matrix.py           # run a matrix of configs, one subprocess per cell
scripts/eval_quality.py         # WikiText-2 perplexity for one (model, format)
scripts/roofline.py             # offline roofline analysis over result summaries
scripts/jsonl_to_summary.py     # regenerate summary from per-prompt JSONL
results/                        # per-run summary JSON (per-prompt JSONL gitignored)
```
