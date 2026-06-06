"""Vanilla HuggingFace generation baseline benchmark on MT-Bench prompts.

Uses model.generate() with a CudaEventStreamer for per-token timing. The
canonical baseline runs with --no-cache (true no-optimization floor); HF's
default is cache-on. All measurement/aggregation lives in llm_inference.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_inference.cli import common_parser, maybe_quality, require_cuda_and_dataset
from llm_inference.data import build_prompt, load_dataset
from llm_inference.modeling import dtype_str, load_model_and_tokenizer
from llm_inference.runner import run_dataset
from llm_inference.timing import CudaEventStreamer, begin_measure, finish_measure


def make_run_one(model, tokenizer, max_new_tokens: int, use_cache: bool):
    def run_one(prompt_text: str) -> dict:
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        prompt_tokens = int(inputs["input_ids"].shape[1])

        streamer = CudaEventStreamer()
        gen_start = begin_measure()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=use_cache,
                streamer=streamer,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_tokens = int(output.shape[1] - prompt_tokens)
        return finish_measure(gen_start, streamer.events, prompt_tokens, generated_tokens)

    return run_one


def warmup(run_one, tokenizer) -> None:
    """One discarded generation to amortize CUDA kernel compile/autotune, so
    the first measured prompt doesn't absorb one-time GPU overhead."""
    run_one(build_prompt(tokenizer, "Hello."))
    torch.cuda.synchronize()


def parse_args():
    p = common_parser(__doc__)
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="disable KV-cache (use_cache=False); off by default, matching HF",
    )
    return p.parse_args()


def main():
    args = parse_args()
    require_cuda_and_dataset(args)

    use_cache = not args.no_cache
    cache_tag = "cache" if use_cache else "nocache"

    print(f"loading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(args.model)
    dtype = dtype_str(model)
    gpu_name = torch.cuda.get_device_name(0)
    print(f"model loaded, dtype={dtype}, gpu={gpu_name}, use_cache={use_cache}")

    run_one = make_run_one(model, tokenizer, args.max_new_tokens, use_cache)
    print("warmup...")
    warmup(run_one, tokenizer)

    quality = maybe_quality(args, model, tokenizer)
    items = load_dataset(args.dataset, args.limit)

    gen_settings = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "use_cache": use_cache,
        "dtype": dtype,
    }

    run_dataset(
        items=items,
        repeats=args.repeats,
        run_one=run_one,
        tokenizer=tokenizer,
        kind=f"baseline_{cache_tag}",
        model_id=args.model,
        gpu=gpu_name,
        gen_settings=gen_settings,
        out_dir=args.out_dir,
        quality=quality,
    )


if __name__ == "__main__":
    main()
