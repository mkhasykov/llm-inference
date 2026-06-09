"""Weight-quantized generation benchmark on MT-Bench prompts.

Two families of weight quantization, same generate path as the baseline:
  * bitsandbytes (int8 / nf4 / fp4): a BitsAndBytesConfig on the base model;
    calibration-free, memory-first (often NOT faster than bf16).
  * pre-quantized checkpoints (awq / gptq-int4 / gptq-int8): load Qwen's
    published <model>-AWQ / -GPTQ-Int{4,8} repo. These use calibration and
    kernel-optimized matmul. The default backend is Triton (gemm_triton /
    triton), a no-nvcc lower bound on 4-bit speed; pass --marlin to use the
    optimized Marlin kernels (needs a CUDA toolkit / nvcc to JIT-compile),
    which on memory-bound decode make 4-bit faster than bf16, not just smaller.

KV-cache is on, so the measured delta is attributable to the quantized weights.
Pair with --quality to capture the perplexity cost (quantization is lossy).
"""

import torch
from transformers import AwqConfig, BitsAndBytesConfig, GPTQConfig

from cli import common_parser, maybe_quality, require_cuda_and_dataset
from data import load_dataset
from modeling import load_model_and_tokenizer, model_stats
from runner import run_dataset

from benchmark_baseline import make_run_one, warmup

# base-model id -> Qwen's published pre-quantized checkpoint suffix
PREQUANT_SUFFIX = {"awq": "-AWQ", "gptq-int4": "-GPTQ-Int4", "gptq-int8": "-GPTQ-Int8"}
BNB_MODES = ("int8", "nf4", "fp4")


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
    raise ValueError(f"unknown bnb quant mode: {mode}")


def load_quant(base_model: str, quant: str, *, marlin: bool = False):
    """Load the quantized model. Returns (model, tokenizer, actual_repo, backend).

    bnb modes quantize the base model in-place; awq/gptq load Qwen's published
    pre-quantized checkpoint (derived by suffix). Matmul backend is Triton by
    default; marlin=True uses the optimized Marlin kernels (needs a CUDA toolkit
    / nvcc to JIT-compile).
    """
    if quant in BNB_MODES:
        model, tok = load_model_and_tokenizer(
            base_model, quantization_config=build_quant_config(quant), device_map="auto"
        )
        return model, tok, base_model, f"bitsandbytes/{quant}"

    if quant in PREQUANT_SUFFIX:
        repo = base_model + PREQUANT_SUFFIX[quant]
        if quant == "awq":
            be = "marlin" if marlin else "gemm_triton"
            qc = AwqConfig(bits=4, group_size=128, backend=be)
            backend = f"awq/{be}"
        else:
            bits = 8 if quant.endswith("int8") else 4
            be = "marlin" if marlin else "triton"
            qc = GPTQConfig(bits=bits, backend=be)
            backend = f"gptq{bits}/{be}"
        # Only `backend` is taken from this config; the rest is read from the
        # checkpoint's own quantization_config.
        model, tok = load_model_and_tokenizer(repo, quantization_config=qc, device_map="cuda")
        return model, tok, repo, backend

    raise ValueError(f"unknown quant mode: {quant}")


def parse_args():
    p = common_parser(__doc__)
    p.add_argument(
        "--quant",
        choices=[*BNB_MODES, *PREQUANT_SUFFIX],
        default="nf4",
        help="bitsandbytes (int8/nf4/fp4) or a pre-quantized checkpoint "
             "(awq/gptq-int4/gptq-int8)",
    )
    p.add_argument(
        "--marlin",
        action="store_true",
        help="use optimized Marlin kernels for awq/gptq (needs CUDA toolkit); "
             "default is the Triton backend",
    )
    return p.parse_args()


def main():
    args = parse_args()
    require_cuda_and_dataset(args)

    use_cache = True  # quantization is always benchmarked with the KV-cache on

    print(f"loading model: {args.model}  quant={args.quant}  marlin={args.marlin}")
    model, tokenizer, repo, backend = load_quant(args.model, args.quant, marlin=args.marlin)
    kind = f"quant_{args.quant}" + ("_marlin" if args.marlin and args.quant in PREQUANT_SUFFIX else "")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"model loaded, quant={args.quant}, repo={repo}, backend={backend}, gpu={gpu_name}")

    run_one = make_run_one(
        model, tokenizer, args.max_new_tokens, use_cache,
        fixed_length=args.fixed_length, dump_text=args.dump_text,
    )
    print("warmup...")
    warmup(run_one, tokenizer)

    quality = maybe_quality(args, model, tokenizer)
    items = load_dataset(args.dataset, args.limit)

    gen_settings = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "use_cache": use_cache,
        "fixed_length": args.fixed_length,
        "quant": args.quant,
        "quant_model": repo,
        "backend": backend,
    }

    run_dataset(
        items=items,
        repeats=args.repeats,
        run_one=run_one,
        tokenizer=tokenizer,
        kind=kind,
        model_id=args.model,
        gpu=gpu_name,
        gen_settings=gen_settings,
        out_dir=args.out_dir,
        quality=quality,
        model_info=model_stats(model),
    )


if __name__ == "__main__":
    main()
