# llm-inference

## Project goal

This is a personal LLM inference benchmarking project on a home RTX 3090 24GB machine.

The goal is to build a small, reproducible benchmark harness for local LLM inference and then compare optimization techniques:

1. vanilla HuggingFace generation
2. explicit/manual KV-cache understanding and measurement
3. quantization
4. speculative decoding
5. later: EAGLE-style acceleration if practical

The project is for learning, portfolio, and engineering depth. Prefer clear experiments over large abstractions.

## Environment

Machine:

- Windows host
- WSL2 Ubuntu
- GPU: NVIDIA GeForce RTX 3090, 24GB
- CUDA works inside WSL
- PyTorch sees GPU successfully

Current Python environment:

- project path: `~/code/llm-inference`
- package manager: `uv`
- venv: `.venv`
- activate with: `source .venv/bin/activate`
- PyTorch: `2.11.0+cu128`
- CUDA available: true
- device capability: `(8, 6)`

Do not assume conda. Use `uv pip install ...`.

## Current status

Already verified:

- WSL Ubuntu works
- `nvidia-smi` works inside WSL
- PyTorch CUDA works
- `Qwen/Qwen2.5-1.5B-Instruct` downloaded and ran successfully
- observed peak VRAM for smoke test: about 2.93 GB

Llama access:

- `meta-llama/Llama-3.1-8B-Instruct` is gated
- access request has been submitted
- do not block work on Llama approval

Claude Code is installed and connected.

## Candidate models

Use two model tiers:

### Dev model

`Qwen/Qwen2.5-1.5B-Instruct`

Purpose:

- fast iteration
- debug benchmark harness
- validate metrics
- avoid long downloads

### Main model later

Prefer one of:

- `Qwen/Qwen3-4B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct` once access is approved

Do not download very large models without asking first.

## Benchmark dataset

Primary dataset:

- MT-Bench questions
- expected path: `data/mt_bench/question.jsonl`
- 80 multi-turn prompts
- use initially for latency and throughput benchmarking, not quality judging

Secondary dataset later:

- HumanEval prompts only
- do not execute generated code initially
- use as a code-generation latency profile

## First milestone

Build a minimal baseline benchmark for HuggingFace generation.

It should measure:

- prompt tokens
- generated tokens
- TTFT if practical
- total latency
- tokens/sec
- ms/token
- peak VRAM
- model name
- dataset item id
- category
- generation settings

Output results as JSONL or CSV under `results/`.

Start simple. Do not introduce vLLM, EAGLE, Triton, custom CUDA, or complex abstractions yet.

## Coding style

Prefer:

- simple Python
- explicit functions
- small files
- readable benchmark code
- reproducible CLI commands

Avoid:

- overengineering
- hidden global state
- big frameworks before baseline works
- training anything before inference benchmark is stable

## Suggested structure

```text
llm-inference/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── data/
│   └── mt_bench/
│       └── question.jsonl
├── scripts/
│   └── benchmark_baseline.py
├── results/
└── src/
    └── llm_inference/
```

This structure is flexible. Keep it minimal.

## First task for Claude

Implement `scripts/benchmark_baseline.py`.

Requirements:

- load MT-Bench from `data/mt_bench/question.jsonl`
- default model: `Qwen/Qwen2.5-1.5B-Instruct`
- run only first N prompts by default, e.g. `--limit 5`
- use chat template if tokenizer supports it
- generate with `max_new_tokens=256`
- default to greedy decoding for reproducibility
- measure CUDA timings with `torch.cuda.Event`
- record peak VRAM with `torch.cuda.max_memory_allocated`
- write JSONL results to `results/baseline_<timestamp>.jsonl`
- print aggregate summary at the end

Ask before installing new heavy dependencies.
Ask before downloading models larger than 5GB.