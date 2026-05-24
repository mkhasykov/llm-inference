"""Manual decode loop with HF's DynamicCache. No model.generate() call.

Same metrics as benchmark_baseline.py with cache=on (TTFT, per-token decode
latency, tokens/sec, peak VRAM), but the loop is explicit: prefill is one
forward pass over the prompt; decode is N forward passes over a single new
token each, with the cache mutated in place. This is the smallest "manual
KV-cache" — we own the loop, HF owns the cache class. Later milestones can
swap DynamicCache for a custom implementation and reuse the rest.
"""

import argparse
import datetime as dt
import json
import statistics
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from summary import build_summary, print_summary


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


def get_eos_ids(model, tokenizer) -> set[int]:
    eos = getattr(model.generation_config, "eos_token_id", None)
    if eos is None:
        eos = tokenizer.eos_token_id
    if isinstance(eos, int):
        return {eos}
    return set(eos)


def manual_generate(
    model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    eos_ids: set[int],
):
    """Prefill + decode loop with DynamicCache. Records a CUDA event after
    each new token. Returns (gen_start_event, per_token_events, cache,
    n_generated). The first event is the prefill end (TTFT)."""
    cache = DynamicCache()

    gen_start = torch.cuda.Event(enable_timing=True)
    gen_start.record()

    # Prefill: full prompt → logits for last position → first new token.
    out = model(input_ids=prompt_ids, past_key_values=cache, use_cache=True)
    next_token = out.logits[:, -1:, :].argmax(-1)

    first_ev = torch.cuda.Event(enable_timing=True)
    first_ev.record()
    events = [first_ev]

    # If the very first sampled token is already EOS, stop.
    if int(next_token.item()) in eos_ids:
        return gen_start, events, cache, 1

    # Decode loop: one new token per step.
    for _ in range(max_new_tokens - 1):
        out = model(input_ids=next_token, past_key_values=cache, use_cache=True)
        next_token = out.logits[:, -1:, :].argmax(-1)

        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        events.append(ev)

        if int(next_token.item()) in eos_ids:
            break

    return gen_start, events, cache, len(events)


def cache_size_bytes(cache: DynamicCache) -> int:
    total = 0
    for layer in cache.layers:
        total += layer.keys.numel() * layer.keys.element_size()
        total += layer.values.numel() * layer.values.element_size()
    return total


def run_one(model, tokenizer, prompt_text: str, max_new_tokens: int, eos_ids: set[int]) -> dict:
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)
    prompt_tokens = int(prompt_ids.shape[1])

    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        gen_start, events, cache, n_generated = manual_generate(
            model, prompt_ids, max_new_tokens, eos_ids
        )

    gen_end = torch.cuda.Event(enable_timing=True)
    gen_end.record()
    torch.cuda.synchronize()

    peak_vram = int(torch.cuda.max_memory_allocated())
    total_ms = gen_start.elapsed_time(gen_end)
    token_ms = [gen_start.elapsed_time(ev) for ev in events]

    ttft_ms = token_ms[0]
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
        tokens_per_sec = ms_per_token_mean = ms_per_token_p50 = ms_per_token_p95 = None

    return {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": n_generated,
        "events_recorded": len(events),
        "total_ms": total_ms,
        "ttft_ms": ttft_ms,
        "tokens_per_sec": tokens_per_sec,
        "ms_per_token_mean": ms_per_token_mean,
        "ms_per_token_p50": ms_per_token_p50,
        "ms_per_token_p95": ms_per_token_p95,
        "peak_vram_bytes": peak_vram,
        "cache_kv_bytes": cache_size_bytes(cache),
        "cache_seq_len": int(cache.get_seq_length()),
    }


def warmup(model, tokenizer, max_new_tokens: int, eos_ids: set[int]) -> None:
    prompt_text = build_prompt(tokenizer, "Hello.")
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)
    with torch.inference_mode():
        manual_generate(model, prompt_ids, max_new_tokens, eos_ids)
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
    out_path = args.out_dir / f"manual_kv_{ts}.jsonl"

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
    eos_ids = get_eos_ids(model, tokenizer)
    print(f"model loaded, dtype={dtype}, gpu={gpu_name}, eos_ids={sorted(eos_ids)}")

    print("warmup...")
    warmup(model, tokenizer, args.max_new_tokens, eos_ids)

    items = load_dataset(args.dataset, args.limit)
    print(f"running {len(items)} prompts → {out_path}")

    gen_settings = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "cache_class": "DynamicCache",
        "dtype": dtype,
    }

    rows = []
    with out_path.open("w") as out_f:
        for item in items:
            user_text = item["turns"][0]
            prompt_text = build_prompt(tokenizer, user_text)
            metrics = run_one(model, tokenizer, prompt_text, args.max_new_tokens, eos_ids)

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
                f"vram={metrics['peak_vram_bytes'] / 1e9:.2f}GB "
                f"kv={metrics['cache_kv_bytes'] / 1e6:.1f}MB@{metrics['cache_seq_len']}tok"
            )

    summary = build_summary(
        rows,
        run_id=out_path.stem,
        kind="manual_kv",
        model=args.model,
        gpu=gpu_name,
        gen_settings=gen_settings,
    )
    summary_path = out_path.with_suffix(".json")
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== summary ===")
    print_summary(summary)
    print(f"results:  per-prompt={out_path}  summary={summary_path}")


if __name__ == "__main__":
    main()
