"""Compute WikiText-2 perplexity for ONE (model, weight-format).

Quality is a property of (model, weight format) only — it does not depend on
the inference method, cache, batch, or sequence length: greedy decode with a
KV-cache is bit-identical to no cache, and speculative decode is lossless. So
the perplexity of bf16 weights is the same number whether measured through the
baseline, the manual KV loop, or the static cache. Measuring it once per format
here avoids recomputing that identical number for every speed run.

    python scripts/eval_quality.py --quant none   # bf16 reference
    python scripts/eval_quality.py --quant nf4

Writes results/quality_<model>_<format>.json, which report tooling joins onto
the speed summaries by (model, format).
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_inference.modeling import load_model_and_tokenizer, model_stats
from llm_inference.quality import compute_perplexity, load_wikitext_text

from benchmark_quant import build_quant_config


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument(
        "--quant",
        choices=["none", "int8", "nf4", "fp4"],
        default="none",
        help="weight format; 'none' = the model's native bf16/fp16",
    )
    p.add_argument("--quality-max-tokens", type=int, default=0, help="0 = full test split")
    p.add_argument("--out-dir", default="results", type=Path)
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        print("CUDA not available; perplexity eval requires a GPU.", file=sys.stderr)
        sys.exit(1)

    fmt = "bf16" if args.quant == "none" else args.quant
    quant_config = None if args.quant == "none" else build_quant_config(args.quant)
    device_map = "cuda" if args.quant == "none" else "auto"

    print(f"loading {args.model}  format={fmt}")
    model, tokenizer = load_model_and_tokenizer(
        args.model, quantization_config=quant_config, device_map=device_map
    )

    text = load_wikitext_text()
    max_tokens = None if args.quality_max_tokens == 0 else args.quality_max_tokens
    q = compute_perplexity(model, tokenizer, text, max_tokens=max_tokens)

    record = {
        "model": args.model,
        "format": fmt,
        "model_info": model_stats(model),
        **q,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    slug = args.model.split("/")[-1]
    out_path = args.out_dir / f"quality_{slug}_{fmt}.json"
    out_path.write_text(json.dumps(record, indent=2))
    print(f"perplexity={q['perplexity']:.4f} over {q['n_tokens']} tokens → {out_path}")


if __name__ == "__main__":
    main()
