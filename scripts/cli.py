"""Shared CLI plumbing: common args, env checks, optional quality eval."""

import argparse
import sys
from pathlib import Path

import torch

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def common_parser(description: str) -> argparse.ArgumentParser:
    """Parser preloaded with the args every benchmark script shares. Scripts
    add their own strategy-specific flags (e.g. --no-cache, --quant)."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--dataset", default="data/mt_bench/question.jsonl", type=Path)
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--out-dir", default="results", type=Path)
    p.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="generations per prompt, for run-to-run variance (default 1)",
    )
    p.add_argument(
        "--quality",
        action="store_true",
        help="also compute WikiText-2 perplexity for this model/config",
    )
    p.add_argument(
        "--quality-max-tokens",
        type=int,
        default=50000,
        help="cap corpus tokens for perplexity; 0 = full test split",
    )
    return p


def require_cuda_and_dataset(args) -> None:
    if not torch.cuda.is_available():
        print("CUDA not available; this benchmark requires a GPU.", file=sys.stderr)
        sys.exit(1)
    if not args.dataset.exists():
        print(f"dataset not found: {args.dataset}", file=sys.stderr)
        sys.exit(1)


def maybe_quality(args, model, tokenizer) -> dict | None:
    """Compute WikiText-2 perplexity if --quality was passed, else None."""
    if not args.quality:
        return None
    from quality import compute_perplexity, load_wikitext_text

    print("computing perplexity (wikitext-2-raw, test)...")
    text = load_wikitext_text()
    max_tokens = None if args.quality_max_tokens == 0 else args.quality_max_tokens
    q = compute_perplexity(model, tokenizer, text, max_tokens=max_tokens)
    print(f"  perplexity={q['perplexity']:.3f} over {q['n_tokens']} tokens")
    return q
