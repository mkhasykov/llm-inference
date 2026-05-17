"""Vanilla HuggingFace generation baseline benchmark on MT-Bench prompts.

Measures TTFT, per-token decode latency, throughput, and peak VRAM using
CUDA events recorded from a custom BaseStreamer. One pass per example.
"""

import argparse
import datetime as dt
import json
import statistics
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer


class CudaEventStreamer(BaseStreamer):
    """Record a CUDA event for every token generate() emits.

    generate() calls put() once with the prompt input_ids, then once per
    newly generated token. We skip the first call and record an event for
    each subsequent token.
    """

    def __init__(self):
        self.events: list[torch.cuda.Event] = []
        self._saw_prompt = False

    def put(self, value):
        if not self._saw_prompt:
            self._saw_prompt = True
            return
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self.events.append(ev)

    def end(self):
        pass


def load_dataset(path: Path, limit: int) -> list[dict]:
    items = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
            if len(items) >= limit:
                break
    return items


def build_prompt(tokenizer, user_text: str) -> str:
    messages = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def run_one(model, tokenizer, prompt_text: str, max_new_tokens: int) -> dict:
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_tokens = int(inputs["input_ids"].shape[1])

    torch.cuda.reset_peak_memory_stats()

    streamer = CudaEventStreamer()
    gen_start = torch.cuda.Event(enable_timing=True)
    gen_end = torch.cuda.Event(enable_timing=True)

    gen_start.record()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            streamer=streamer,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_end.record()
    torch.cuda.synchronize()

    peak_vram = int(torch.cuda.max_memory_allocated())
    total_ms = gen_start.elapsed_time(gen_end)
    generated_tokens = int(output.shape[1] - prompt_tokens)

    events = streamer.events
    token_ms = [gen_start.elapsed_time(ev) for ev in events]

    if len(token_ms) >= 1:
        ttft_ms = token_ms[0]
    else:
        ttft_ms = None

    if len(token_ms) >= 2:
        decode_intervals = [token_ms[i] - token_ms[i - 1] for i in range(1, len(token_ms))]
        decode_total_s = (token_ms[-1] - token_ms[0]) / 1000.0
        decode_tokens = len(token_ms) - 1
        tokens_per_sec = decode_tokens / decode_total_s if decode_total_s > 0 else None
        ms_per_token_mean = statistics.fmean(decode_intervals)
        ms_per_token_p50 = statistics.median(decode_intervals)
        sorted_ints = sorted(decode_intervals)
        p95_idx = max(0, int(round(0.95 * (len(sorted_ints) - 1))))
        ms_per_token_p95 = sorted_ints[p95_idx]
    else:
        decode_intervals = []
        tokens_per_sec = None
        ms_per_token_mean = None
        ms_per_token_p50 = None
        ms_per_token_p95 = None

    return {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "events_recorded": len(events),
        "total_ms": total_ms,
        "ttft_ms": ttft_ms,
        "tokens_per_sec": tokens_per_sec,
        "ms_per_token_mean": ms_per_token_mean,
        "ms_per_token_p50": ms_per_token_p50,
        "ms_per_token_p95": ms_per_token_p95,
        "peak_vram_bytes": peak_vram,
    }


def warmup(model, tokenizer, max_new_tokens: int) -> None:
    """One discarded generation to amortize CUDA kernel compile/autotune.

    Without this, the first measured example absorbs all the one-time GPU
    overhead and reports inflated TTFT.
    """
    prompt_text = build_prompt(tokenizer, "Hello.")
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--dataset", default="data/mt_bench/question.jsonl", type=Path)
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--out-dir", default="results", type=Path)
    return p.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; this benchmark requires a GPU.", file=sys.stderr)
        sys.exit(1)

    if not args.dataset.exists():
        print(f"dataset not found: {args.dataset}", file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out_dir / f"baseline_{ts}.jsonl"

    print(f"loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="cuda",
    )
    model.eval()
    dtype = str(next(model.parameters()).dtype).replace("torch.", "")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"model loaded, dtype={dtype}, gpu={gpu_name}")

    print("warmup...")
    warmup(model, tokenizer, args.max_new_tokens)

    items = load_dataset(args.dataset, args.limit)
    print(f"running {len(items)} prompts → {out_path}")

    gen_settings = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "dtype": dtype,
    }

    rows = []
    with out_path.open("w") as out_f:
        for item in items:
            user_text = item["turns"][0]
            prompt_text = build_prompt(tokenizer, user_text)
            metrics = run_one(model, tokenizer, prompt_text, args.max_new_tokens)

            row = {
                "model": args.model,
                "question_id": item["question_id"],
                "category": item["category"],
                "gen_settings": gen_settings,
                "gpu": gpu_name,
                "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                **metrics,
            }
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()
            rows.append(row)

            tps = f"{metrics['tokens_per_sec']:.1f}" if metrics["tokens_per_sec"] else "n/a"
            ttft = f"{metrics['ttft_ms']:.1f}" if metrics["ttft_ms"] else "n/a"
            print(
                f"  q{item['question_id']} [{item['category']}] "
                f"prompt={metrics['prompt_tokens']} gen={metrics['generated_tokens']} "
                f"ttft={ttft}ms tok/s={tps} "
                f"vram={metrics['peak_vram_bytes'] / 1e9:.2f}GB"
            )

    print("\n=== summary ===")
    tps_values = [r["tokens_per_sec"] for r in rows if r["tokens_per_sec"]]
    ttft_values = [r["ttft_ms"] for r in rows if r["ttft_ms"]]
    vram_values = [r["peak_vram_bytes"] for r in rows]
    if tps_values:
        print(f"tokens/sec  mean={statistics.fmean(tps_values):.2f}  "
              f"min={min(tps_values):.2f}  max={max(tps_values):.2f}")
    if ttft_values:
        print(f"ttft (ms)   mean={statistics.fmean(ttft_values):.2f}  "
              f"min={min(ttft_values):.2f}  max={max(ttft_values):.2f}")
    if vram_values:
        print(f"peak VRAM   max={max(vram_values) / 1e9:.2f}GB")
    print(f"results: {out_path}")


if __name__ == "__main__":
    main()
