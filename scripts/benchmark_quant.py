"""Weight-quantized generation benchmark on MT-Bench prompts.

Same generate path as benchmark_baseline.py — only the model load changes
(a BitsAndBytesConfig per --quant mode). KV-cache is on, so the measured
delta is attributable to the quantized weights alone. Pair with --quality to
capture the perplexity cost of quantization (it is NOT lossless, unlike spec
decoding).

Modes: int8 (LLM.int8()), nf4 / fp4 (4-bit + double quant, bf16 compute).
"""

import sys
from pathlib import Path

import torch
from transformers import BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_inference.cli import common_parser, maybe_quality, require_cuda_and_dataset
from llm_inference.data import load_dataset
from llm_inference.modeling import load_model_and_tokenizer, model_stats
from llm_inference.runner import run_dataset

from benchmark_baseline import make_run_one, warmup


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
    p = common_parser(__doc__)
    p.add_argument(
        "--quant",
        choices=["int8", "nf4", "fp4"],
        default="nf4",
        help="bitsandbytes weight-quantization scheme (default: nf4)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    require_cuda_and_dataset(args)

    use_cache = True  # quantization is always benchmarked with the KV-cache on
    compute_dtype = "bfloat16"

    print(f"loading model: {args.model}  quant={args.quant}")
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        quantization_config=build_quant_config(args.quant),
        device_map="auto",
    )
    gpu_name = torch.cuda.get_device_name(0)
    print(f"model loaded, quant={args.quant}, compute_dtype={compute_dtype}, gpu={gpu_name}")

    run_one = make_run_one(model, tokenizer, args.max_new_tokens, use_cache)
    print("warmup...")
    warmup(run_one, tokenizer)

    quality = maybe_quality(args, model, tokenizer)
    items = load_dataset(args.dataset, args.limit)

    gen_settings = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "use_cache": use_cache,
        "quant": args.quant,
        "compute_dtype": compute_dtype,
    }

    run_dataset(
        items=items,
        repeats=args.repeats,
        run_one=run_one,
        tokenizer=tokenizer,
        kind=f"quant_{args.quant}",
        model_id=args.model,
        gpu=gpu_name,
        gen_settings=gen_settings,
        out_dir=args.out_dir,
        quality=quality,
        model_info=model_stats(model),
    )


if __name__ == "__main__":
    main()
