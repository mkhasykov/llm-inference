"""Static-batching throughput benchmark on MT-Bench prompts.

Batching is the throughput lever: reusing the weights across B sequences raises
arithmetic intensity, so aggregate tokens/sec climbs with batch size until the
GPU saturates (compute) or VRAM runs out. This sweeps a single --batch-size;
the driver runs several to trace the curve.

Decoder-only batching needs LEFT padding so the generated tokens align. We
force fixed length (min=max new tokens) so every sequence does identical work
and the batch has no ragged tail. Works for fp16 and any --quant mode (the
memory headroom of quantization lets larger batches fit).

This is *static* batching (one fixed batch in flight); production continuous
batching (vLLM) is out of scope and discussed in the text only.
"""

import datetime as dt
import json

import torch

from cli import common_parser, require_cuda_and_dataset
from data import build_prompt, load_dataset
from modeling import dtype_str, load_model_and_tokenizer, model_stats
from summary import aggregate_repeats, build_summary, print_summary
from timing import CudaEventStreamer, begin_measure, finish_measure_batch
from env import gpu_state

from benchmark_quant import load_quant


def make_run_batch(model, tokenizer, max_new_tokens):
    def run_batch(prompts: list[str]) -> dict:
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        padded = int(enc["input_ids"].shape[1])
        streamer = CudaEventStreamer()
        gen_start = begin_measure()
        with torch.inference_mode():
            model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                min_new_tokens=max_new_tokens,  # fixed length: no ragged tail
                do_sample=False,
                use_cache=True,
                streamer=streamer,
                pad_token_id=tokenizer.eos_token_id,
            )
        return finish_measure_batch(gen_start, streamer.events, len(prompts), padded)

    return run_batch


def parse_args():
    p = common_parser(__doc__)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--quant",
        default="none",
        choices=["none", "int8", "nf4", "fp4", "awq", "gptq-int4", "gptq-int8"],
        help="none = native bf16; otherwise the weight-quantization mode",
    )
    return p.parse_args()


def main():
    args = parse_args()
    require_cuda_and_dataset(args)
    B = args.batch_size

    if args.quant == "none":
        model, tokenizer = load_model_and_tokenizer(args.model)
        fmt, backend = dtype_str(model), None
    else:
        model, tokenizer, _repo, backend = load_quant(args.model, args.quant)
        fmt = args.quant

    tokenizer.padding_side = "left"  # required for decoder-only batch generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    gpu_name = torch.cuda.get_device_name(0)
    print(f"model loaded, fmt={fmt}, backend={backend}, batch_size={B}, gpu={gpu_name}")

    run_batch = make_run_batch(model, tokenizer, args.max_new_tokens)
    print("warmup...")
    with torch.inference_mode():
        run_batch([build_prompt(tokenizer, "Hello.")] * B)
    torch.cuda.synchronize()

    items = load_dataset(args.dataset, args.limit)
    batches = [items[i:i + B] for i in range(0, len(items), B)]
    full = [b for b in batches if len(b) == B]
    if not full:
        print(f"note: fewer than one full batch ({len(items)} < {B}); using a single partial batch")
        full = batches
    elif len(full) < len(batches):
        print(f"note: dropped trailing partial batch ({len(items) - len(full) * B} prompts) "
              f"for clean batch-size measurement")
    batches = full

    gen_settings = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "fixed_length": True,
        "batch_size": B,
        "quant": args.quant,
        "backend": backend,
        "format": fmt,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    kind = f"batch_{fmt}_b{B}"
    out_path = args.out_dir / f"{kind}_{ts}.jsonl"
    print(f"running {len(batches)} batches of {B} × {args.repeats} repeats → {out_path}")

    state_start = gpu_state()
    rows = []
    with out_path.open("w") as f:
        for bi, batch in enumerate(batches):
            prompts = [build_prompt(tokenizer, it["turns"][0]) for it in batch]
            reps = [run_batch(prompts) for _ in range(args.repeats)]
            row = {
                "model": args.model,
                "question_id": f"batch{bi}",
                "category": f"bs{len(prompts)}",
                "gen_settings": gen_settings,
                "gpu": gpu_name,
                "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "repeats": args.repeats,
                **aggregate_repeats(reps),
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            rows.append(row)
            tps = row["tokens_per_sec"]
            tps_s = f"{tps['mean']:.0f}±{tps['std']:.0f}" if tps else "n/a"
            print(f"  batch{bi} (bs={len(prompts)}): agg tok/s={tps_s}  "
                  f"vram={row['peak_vram_bytes'] / 1e9:.2f}GB")

    summary = build_summary(
        rows,
        run_id=out_path.stem,
        kind=kind,
        model=args.model,
        gpu=gpu_name,
        gen_settings=gen_settings,
        repeats=args.repeats,
        model_info=model_stats(model),
        gpu_state={"start": state_start, "end": gpu_state()},
    )
    summary_path = out_path.with_suffix(".json")
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== summary ===")
    print_summary(summary)
    print(f"results:  per-batch={out_path}  summary={summary_path}")


if __name__ == "__main__":
    main()
