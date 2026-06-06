"""Quality of a (model, weight-format): WikiText-2 perplexity + MMLU + GSM8K.

Quality is a property of the weights alone — independent of cache, batch, or
speculative decoding (all lossless) — so we measure it once per format and let
report tooling join it onto the speed runs by (model, format). Only the lossy
weight formats need this: bf16 (reference), int8, nf4, fp4, awq, gptq-*.

  * perplexity — our own sliding-window WikiText-2 (intrinsic LM quality)
  * MMLU       — lm-eval, loglikelihood over A/B/C/D (knowledge/reasoning)
  * GSM8K      — lm-eval, generate + numeric match (multi-step reasoning)

We load the model ourselves (AWQ/GPTQ via the Triton backend) and hand the
instance to lm-eval's HFLM, so every format goes through one code path.

    python scripts/eval_quality.py --quant none            # bf16 reference
    python scripts/eval_quality.py --quant nf4
    python scripts/eval_quality.py --quant awq --mmlu-limit 10 --gsm8k-limit 200
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from modeling import load_model_and_tokenizer, model_stats
from quality import compute_perplexity, load_wikitext_text

from benchmark_quant import load_quant


def load_for_format(base_model: str, quant: str):
    """Return (model, tokenizer, format_label, backend)."""
    if quant == "none":
        model, tok = load_model_and_tokenizer(base_model)
        return model, tok, "bf16", None
    model, tok, _repo, backend = load_quant(base_model, quant)
    return model, tok, quant, backend


def run_lm_eval(model, tokenizer, *, mmlu_limit, gsm8k_limit, batch_size):
    """Run MMLU and GSM8K via lm-eval on an already-loaded model."""
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    out = {}

    mmlu = simple_evaluate(model=lm, tasks=["mmlu"], limit=mmlu_limit or None)
    out["mmlu_acc"] = round(float(mmlu["results"]["mmlu"]["acc,none"]), 4)

    gsm = simple_evaluate(model=lm, tasks=["gsm8k"], limit=gsm8k_limit or None)
    g = gsm["results"]["gsm8k"]
    key = "exact_match,strict-match" if "exact_match,strict-match" in g else "exact_match,flexible-extract"
    out["gsm8k_exact_match"] = round(float(g[key]), 4)
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument(
        "--quant",
        default="none",
        choices=["none", "int8", "nf4", "fp4", "awq", "gptq-int4", "gptq-int8"],
        help="weight format; 'none' = the model's native bf16",
    )
    p.add_argument("--out-dir", default="results", type=Path)
    p.add_argument("--mmlu-limit", type=int, default=10,
                   help="MMLU examples PER subtask (57 subtasks); 0 = full")
    p.add_argument("--gsm8k-limit", type=int, default=200, help="GSM8K examples; 0 = full")
    p.add_argument("--ppl-max-tokens", type=int, default=50000,
                   help="cap WikiText-2 tokens for perplexity; 0 = full test split")
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        print("CUDA not available; this requires a GPU.", file=sys.stderr)
        sys.exit(1)

    print(f"loading {args.model}  quant={args.quant}")
    model, tokenizer, fmt, backend = load_for_format(args.model, args.quant)
    gpu_name = torch.cuda.get_device_name(0)

    print("perplexity (wikitext-2-raw, test)...")
    text = load_wikitext_text()
    ppl_cap = None if args.ppl_max_tokens == 0 else args.ppl_max_tokens
    ppl = compute_perplexity(model, tokenizer, text, max_tokens=ppl_cap)
    print(f"  perplexity={ppl['perplexity']:.3f} over {ppl['n_tokens']} tokens")

    print(f"MMLU (limit {args.mmlu_limit}/subtask) + GSM8K (limit {args.gsm8k_limit})...")
    acc = run_lm_eval(
        model, tokenizer,
        mmlu_limit=args.mmlu_limit, gsm8k_limit=args.gsm8k_limit,
        batch_size=args.batch_size,
    )
    print(f"  mmlu_acc={acc['mmlu_acc']:.4f}  gsm8k_exact_match={acc['gsm8k_exact_match']:.4f}")

    result = {
        "kind": "quality",
        "model": args.model,
        "quant": args.quant,
        "format": fmt,
        "backend": backend,
        "gpu": gpu_name,
        "model_info": model_stats(model),
        "perplexity": ppl,
        "mmlu": {"acc": acc["mmlu_acc"], "limit_per_subtask": args.mmlu_limit},
        "gsm8k": {"exact_match": acc["gsm8k_exact_match"], "limit": args.gsm8k_limit},
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    safe_model = args.model.replace("/", "_")
    out_path = args.out_dir / f"quality_{safe_model}_{args.quant}.json"
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"\nquality result → {out_path}")


if __name__ == "__main__":
    main()
