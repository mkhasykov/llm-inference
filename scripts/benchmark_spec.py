"""Speculative (assisted) decoding benchmark on MT-Bench prompts.

A small draft model proposes K tokens; the target verifies them in one forward
pass and accepts the longest correct prefix. Greedy assisted decoding is
*lossless* — the output is identical to the target's own greedy output (see
check_fidelity.py) — so there is no quality axis here, only speed.

generate() emits the accepted tokens of each step as a block, so we time with
BlockStreamer and report mean_block_size (tokens per target forward pass) and
effective tokens/sec. Compare against benchmark_baseline.py (cache on) to see
the net speedup; the draft adds VRAM, captured in peak_vram.
"""

import torch

from cli import common_parser, require_cuda_and_dataset
from data import load_dataset
from modeling import dtype_str, load_model_and_tokenizer, model_stats
from runner import run_dataset
from timing import BlockStreamer, begin_measure, finish_measure_blocks

from benchmark_baseline import warmup
from benchmark_quant import load_quant


def make_run_one(model, draft, tokenizer, max_new_tokens, *,
                 fixed_length=False, dump_text=False, assistant_tokenizer=None,
                 prompt_lookup_tokens=0):
    def run_one(prompt_text: str) -> dict:
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        prompt_tokens = int(inputs["input_ids"].shape[1])

        streamer = BlockStreamer()
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "use_cache": True,
            "streamer": streamer,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if prompt_lookup_tokens > 0:
            # n-gram prompt-lookup decoding: candidates copied from the context,
            # no draft model and no vocab-size constraint.
            gen_kwargs["prompt_lookup_num_tokens"] = prompt_lookup_tokens
        else:
            gen_kwargs["assistant_model"] = draft
            # Universal assisted decoding: needed when target/draft have different
            # config.vocab_size (e.g. Qwen2.5-7B=152064 vs 0.5B=151936; the actual
            # tokenizers are identical, but generate() requires both to be passed).
            if assistant_tokenizer is not None:
                gen_kwargs["tokenizer"] = tokenizer
                gen_kwargs["assistant_tokenizer"] = assistant_tokenizer
        if fixed_length:
            gen_kwargs["min_new_tokens"] = max_new_tokens

        gen_start = begin_measure()
        with torch.inference_mode():
            output = model.generate(**inputs, **gen_kwargs)
        generated_tokens = int(output.shape[1] - prompt_tokens)

        extra = None
        if dump_text:
            extra = {"generated_text": tokenizer.decode(output[0, prompt_tokens:])}
        return finish_measure_blocks(
            gen_start, streamer.events, streamer.block_sizes,
            prompt_tokens, generated_tokens, extra,
        )

    return run_one


def parse_args():
    p = common_parser(__doc__)
    p.add_argument(
        "--draft-model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="small assistant model proposing tokens (must share the target's tokenizer)",
    )
    p.add_argument(
        "--quant",
        default="none",
        choices=["none", "int8", "nf4", "fp4", "awq", "gptq-int4", "gptq-int8"],
        help="quantize the TARGET (draft stays bf16) — for spec×quant combinations",
    )
    p.add_argument(
        "--prompt-lookup-tokens",
        type=int, default=0,
        help="if >0, use n-gram prompt-lookup decoding (no draft model, no vocab constraint)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    require_cuda_and_dataset(args)

    if args.quant == "none":
        print(f"loading target: {args.model}")
        model, tokenizer = load_model_and_tokenizer(args.model)
        target_fmt, backend = dtype_str(model), None
    else:
        print(f"loading target: {args.model}  quant={args.quant}")
        model, tokenizer, _repo, backend = load_quant(args.model, args.quant)
        target_fmt = args.quant
    lookup = args.prompt_lookup_tokens > 0
    if lookup:
        draft, draft_tok, uad = None, None, False
        print(f"prompt-lookup decoding (n-gram), lookup_tokens={args.prompt_lookup_tokens}, "
              f"target_fmt={target_fmt}")
    else:
        print(f"loading draft:  {args.draft_model}")
        draft, draft_tok = load_model_and_tokenizer(args.draft_model)
        # Different output vocab size (embedding padding, e.g. 7B=152064 vs
        # 0.5B=151936) -> generate() needs universal assisted decoding with both
        # tokenizers passed, even though the tokenizers themselves are identical.
        uad = model.config.vocab_size != draft.config.vocab_size
        print(f"models loaded, target_fmt={target_fmt}, backend={backend}, "
              f"vocab target={model.config.vocab_size} draft={draft.config.vocab_size} "
              f"universal_assisted={uad}")
    gpu_name = torch.cuda.get_device_name(0)

    run_one = make_run_one(
        model, draft, tokenizer, args.max_new_tokens,
        fixed_length=args.fixed_length, dump_text=args.dump_text,
        assistant_tokenizer=(draft_tok if uad else None),
        prompt_lookup_tokens=args.prompt_lookup_tokens,
    )
    print("warmup...")
    warmup(run_one, tokenizer)

    items = load_dataset(args.dataset, args.limit)

    gen_settings = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "fixed_length": args.fixed_length,
        "method": "prompt_lookup" if lookup else "speculative",
        "draft_model": None if lookup else args.draft_model,
        "prompt_lookup_tokens": args.prompt_lookup_tokens,
        "quant": args.quant,
        "target_format": target_fmt,
        "backend": backend,
        "universal_assisted": uad,
    }
    if lookup:
        kind = "spec_lookup"
    else:
        kind = "spec" if args.quant == "none" else f"spec_{args.quant}"

    # No quality axis: greedy assisted decoding is lossless (= target greedy).
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
        quality=None,
        model_info=model_stats(model),
    )


if __name__ == "__main__":
    main()
