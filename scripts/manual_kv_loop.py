"""Manual prefill+decode loop with HF's DynamicCache. No model.generate().

The smallest "manual KV-cache": we own the loop, HF owns the cache class.
DynamicCache grows via torch.cat on every decode step. All timing/aggregation
lives in the shared modules; only the cache + loop are here. Also reports the
KV-cache size in bytes, which generate() does not expose.
"""

import torch
from transformers.cache_utils import DynamicCache

from cli import common_parser, maybe_quality, require_cuda_and_dataset
from data import build_prompt, load_dataset
from decode import manual_decode
from modeling import dtype_str, get_eos_ids, load_model_and_tokenizer, model_stats
from runner import run_dataset
from timing import begin_measure, finish_measure


def cache_size_bytes(cache: DynamicCache) -> int:
    total = 0
    for layer in cache.layers:
        total += layer.keys.numel() * layer.keys.element_size()
        total += layer.values.numel() * layer.values.element_size()
    return total


def make_run_one(model, tokenizer, max_new_tokens: int, eos_ids: set[int]):
    def run_one(prompt_text: str) -> dict:
        prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)
        prompt_tokens = int(prompt_ids.shape[1])

        cache = DynamicCache()
        gen_start = begin_measure()
        with torch.inference_mode():
            events, n_generated = manual_decode(
                model, prompt_ids, cache, max_new_tokens, eos_ids
            )
        extra = {
            "cache_kv_bytes": cache_size_bytes(cache),
            "cache_seq_len": int(cache.get_seq_length()),
        }
        return finish_measure(gen_start, events, prompt_tokens, n_generated, extra)

    return run_one


def warmup(run_one, tokenizer) -> None:
    run_one(build_prompt(tokenizer, "Hello."))
    torch.cuda.synchronize()


def main():
    args = common_parser(__doc__).parse_args()
    require_cuda_and_dataset(args)

    print(f"loading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(args.model)
    dtype = dtype_str(model)
    gpu_name = torch.cuda.get_device_name(0)
    eos_ids = get_eos_ids(model, tokenizer)
    print(f"model loaded, dtype={dtype}, gpu={gpu_name}, eos_ids={sorted(eos_ids)}")

    run_one = make_run_one(model, tokenizer, args.max_new_tokens, eos_ids)
    print("warmup...")
    warmup(run_one, tokenizer)

    quality = maybe_quality(args, model, tokenizer)
    items = load_dataset(args.dataset, args.limit)

    gen_settings = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "cache_class": "DynamicCache",
        "dtype": dtype,
    }

    run_dataset(
        items=items,
        repeats=args.repeats,
        run_one=run_one,
        tokenizer=tokenizer,
        kind="manual_kv",
        model_id=args.model,
        gpu=gpu_name,
        gen_settings=gen_settings,
        out_dir=args.out_dir,
        quality=quality,
        model_info=model_stats(model),
    )


if __name__ == "__main__":
    main()
