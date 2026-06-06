"""Weight-quantized generation benchmark on MT-Bench prompts.

Same measurement path as `benchmark_baseline.py` (model.generate() with a
CudaEventStreamer recording per-token CUDA events), but the model is loaded
with bitsandbytes weight quantization. Quantization is purely a load-time
change — the decode loop, metrics, and KV-cache (on) are identical to the
baseline, so the delta is attributable to the quantized weights alone.

Modes:
  int8  LLM.int8() 8-bit weights
  nf4   4-bit NormalFloat + double quant, bf16 compute (QLoRA default)
  fp4   4-bit float + double quant, bf16 compute

Headline metric for quantization is peak VRAM, not throughput: 4-bit weights
cut the model footprint ~4x, but dequant-on-the-fly can leave tokens/sec
flat or slightly slower on a small model. Greedy output is NOT identical to
fp16 (quantization changes the arithmetic) — unlike speculative decoding.
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Reuse the baseline generate-path helpers verbatim so the measurement code
# stays in lockstep — quant differs only in how the model is loaded.
from benchmark_baseline import (
    build_prompt,
    load_dataset,
    run_one,
    warmup,
)
from summary import build_summary, print_summary


def build_quant_config(mode: str) -> BitsAndBytesConfig:
    if mode == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    if mode in ("nf4", "fp4"):
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=mode,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    raise ValueError(f"unknown quant mode: {mode}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--dataset", default="data/mt_bench/question.jsonl", type=Path)
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--out-dir", default="results", type=Path)
    p.add_argument(
        "--quant",
        choices=["int8", "nf4", "fp4"],
        default="nf4",
        help="bitsandbytes weight-quantization scheme (default: nf4)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; this benchmark requires a GPU.", file=sys.stderr)
        sys.exit(1)

    if not args.dataset.exists():
        print(f"dataset not found: {args.dataset}", file=sys.stderr)
        sys.exit(1)

    use_cache = True  # quantization is always benchmarked with the KV-cache on
    compute_dtype = "bfloat16"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out_dir / f"quant_{args.quant}_{ts}.jsonl"

    print(f"loading model: {args.model}  quant={args.quant}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=build_quant_config(args.quant),
        device_map="auto",
    )
    model.eval()
    gpu_name = torch.cuda.get_device_name(0)
    print(
        f"model loaded, quant={args.quant}, compute_dtype={compute_dtype}, "
        f"gpu={gpu_name}, use_cache={use_cache}"
    )

    print("warmup...")
    warmup(model, tokenizer, args.max_new_tokens, use_cache)

    items = load_dataset(args.dataset, args.limit)
    print(f"running {len(items)} prompts → {out_path}")

    gen_settings = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "use_cache": use_cache,
        "quant": args.quant,
        "compute_dtype": compute_dtype,
    }

    rows = []
    with out_path.open("w") as out_f:
        for item in items:
            user_text = item["turns"][0]
            prompt_text = build_prompt(tokenizer, user_text)
            metrics = run_one(model, tokenizer, prompt_text, args.max_new_tokens, use_cache)

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

    summary = build_summary(
        rows,
        run_id=out_path.stem,
        kind="quant",
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
