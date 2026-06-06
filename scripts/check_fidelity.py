"""Verify that speculative decoding is lossless.

Greedy assisted decoding must produce the target's own greedy output token for
token -- the method's correctness guarantee. This runs each prompt twice
(target-only vs target+draft, both greedy) and compares the generated ids.

The guarantee is exact only in exact arithmetic: in **fp32 the match is 100%**
(verified). In bf16/fp16 the batched verification step computes the target
logits with slightly different rounding than single-token decoding, so a
near-tie argmax can flip and the greedy sequences cascade-diverge afterwards --
both remain valid greedy outputs of the target. Use --dtype float32 as the
correctness oracle; the default (auto/bf16) exposes the numerical divergence.

    python scripts/check_fidelity.py --limit 5 --dtype float32
"""

import torch

from cli import common_parser, require_cuda_and_dataset
from data import build_prompt, load_dataset
from modeling import load_model_and_tokenizer

DTYPES = {
    "auto": "auto",
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def greedy_ids(model, tokenizer, prompt_text, max_new_tokens, assistant=None):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if assistant is not None:
        kwargs["assistant_model"] = assistant
    with torch.inference_mode():
        out = model.generate(**inputs, **kwargs)
    return out[0, inputs["input_ids"].shape[1]:].tolist()


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def parse_args():
    p = common_parser(__doc__)
    p.add_argument("--draft-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument(
        "--dtype",
        choices=list(DTYPES),
        default="auto",
        help="float32 = exact-arithmetic correctness oracle (expect 100%); "
             "auto/bfloat16 expose the numerical near-tie divergence",
    )
    return p.parse_args()


def main():
    args = parse_args()
    require_cuda_and_dataset(args)

    dtype = DTYPES[args.dtype]
    print(f"target: {args.model}  draft: {args.draft_model}  dtype: {args.dtype}")
    model, tokenizer = load_model_and_tokenizer(args.model, dtype=dtype)
    draft, _ = load_model_and_tokenizer(args.draft_model, dtype=dtype)
    items = load_dataset(args.dataset, args.limit)

    n_match = 0
    agreements = []
    for item in items:
        prompt_text = build_prompt(tokenizer, item["turns"][0])
        base = greedy_ids(model, tokenizer, prompt_text, args.max_new_tokens)
        spec = greedy_ids(model, tokenizer, prompt_text, args.max_new_tokens, assistant=draft)
        match = base == spec
        n_match += int(match)
        div = first_divergence(base, spec)
        agree = 1.0 if match else (div / max(1, min(len(base), len(spec))))
        agreements.append(agree)
        print(f"  q{item['question_id']}: match={match}  len={len(base)}/{len(spec)}  "
              f"first_divergence={div}  prefix_agreement={agree:.1%}")

    mean_agree = sum(agreements) / len(agreements) if agreements else 0.0
    print(f"\nfidelity ({args.dtype}): {n_match}/{len(items)} exact, "
          f"mean prefix-agreement {mean_agree:.1%}")
    if args.dtype == "float32" and n_match != len(items):
        print("WARNING: an fp32 mismatch indicates a real bug, not numerical noise.")
    elif n_match != len(items):
        print("Note: bf16/fp16 divergence is expected (near-tie argmax flips cascade); "
              "run --dtype float32 for the exact-arithmetic correctness check.")


if __name__ == "__main__":
    main()
