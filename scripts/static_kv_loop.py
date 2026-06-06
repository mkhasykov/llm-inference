"""Manual prefill+decode loop with our pre-allocated KV cache.

Same loop as manual_kv_loop.py, but DynamicCache is replaced with
PreallocatedKVCache: a per-layer buffer of (batch, n_kv_heads,
prompt+max_new_tokens, head_dim) allocated once and written in place — no
torch.cat in the decode hot path. Cost: VRAM is pinned for the full budget
upfront. See README for the (negative) throughput result and why.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_inference.caches import PreallocatedKVCache
from llm_inference.cli import common_parser, maybe_quality, require_cuda_and_dataset
from llm_inference.data import build_prompt, load_dataset
from llm_inference.decode import manual_decode
from llm_inference.modeling import dtype_str, get_eos_ids, load_model_and_tokenizer
from llm_inference.runner import run_dataset
from llm_inference.timing import begin_measure, finish_measure


def make_run_one(model, tokenizer, max_new_tokens: int, eos_ids: set[int], n_layers: int):
    def run_one(prompt_text: str) -> dict:
        prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)
        prompt_tokens = int(prompt_ids.shape[1])

        cache = PreallocatedKVCache(
            max_cache_len=prompt_tokens + max_new_tokens,
            n_layers=n_layers,
        )
        gen_start = begin_measure()
        with torch.inference_mode():
            events, n_generated = manual_decode(
                model, prompt_ids, cache, max_new_tokens, eos_ids
            )
        extra = {
            "cache_kv_bytes": cache.buffer_bytes(),
            "cache_kv_used_bytes": cache.used_bytes(),
            "cache_seq_len": int(cache.get_seq_length()),
            "cache_max_len": cache.max_cache_len,
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
    n_layers = model.config.num_hidden_layers
    print(
        f"model loaded, dtype={dtype}, gpu={gpu_name}, "
        f"n_layers={n_layers}, eos_ids={sorted(eos_ids)}"
    )

    run_one = make_run_one(model, tokenizer, args.max_new_tokens, eos_ids, n_layers)
    print("warmup...")
    warmup(run_one, tokenizer)

    quality = maybe_quality(args, model, tokenizer)
    items = load_dataset(args.dataset, args.limit)

    gen_settings = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "cache_class": "PreallocatedKVCache",
        "dtype": dtype,
    }

    run_dataset(
        items=items,
        repeats=args.repeats,
        run_one=run_one,
        tokenizer=tokenizer,
        kind="static_kv",
        model_id=args.model,
        gpu=gpu_name,
        gen_settings=gen_settings,
        out_dir=args.out_dir,
        quality=quality,
    )


if __name__ == "__main__":
    main()
