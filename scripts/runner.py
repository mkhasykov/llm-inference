"""The per-prompt benchmark loop, shared by every experiment.

Each script supplies a `run_one(prompt_text) -> metrics` callable (its own
generation strategy, ending in `finish_measure`). The runner handles the
parts that never change: build the prompt, repeat each prompt R times,
aggregate the repeats, stream rows to JSONL, build + write + print the summary.
"""

import datetime as dt
import json
from pathlib import Path

from data import build_prompt
from summary import aggregate_repeats, build_summary, print_summary


def _fmt_stat(stat: dict | None, decimals: int = 1) -> str:
    if not stat or stat.get("mean") is None:
        return "n/a"
    return f"{stat['mean']:.{decimals}f}±{stat['std']:.{decimals}f}"


def _print_row(row: dict) -> None:
    parts = [
        f"  q{row['question_id']} [{row['category']}]",
        f"prompt={row['prompt_tokens']} gen={row['generated_tokens']}",
        f"ttft={_fmt_stat(row.get('ttft_ms'))}ms",
        f"tok/s={_fmt_stat(row.get('tokens_per_sec'))}",
        f"vram={row['peak_vram_bytes'] / 1e9:.2f}GB",
    ]
    if isinstance(row.get("cache_kv_bytes"), int):
        parts.append(f"kv={row['cache_kv_bytes'] / 1e6:.1f}MB@{row.get('cache_seq_len')}tok")
    print(" ".join(parts))


def run_dataset(
    *,
    items: list[dict],
    repeats: int,
    run_one,
    tokenizer,
    kind: str,
    model_id: str,
    gpu: str,
    gen_settings: dict,
    out_dir: Path,
    quality: dict | None = None,
    model_info: dict | None = None,
) -> dict:
    """Run `items` through `run_one`, R repeats each, and write results.

    `run_one(prompt_text)` must return the per-generation metric dict from
    `finish_measure`. Returns the summary dict (also written to disk).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"{kind}_{ts}.jsonl"
    print(f"running {len(items)} prompts × {repeats} repeats → {out_path}")

    rows = []
    with out_path.open("w") as f:
        for item in items:
            prompt_text = build_prompt(tokenizer, item["turns"][0])
            reps = [run_one(prompt_text) for _ in range(repeats)]
            row = {
                "model": model_id,
                "question_id": item["question_id"],
                "category": item["category"],
                "gen_settings": gen_settings,
                "gpu": gpu,
                "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "repeats": repeats,
                **aggregate_repeats(reps),
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            rows.append(row)
            _print_row(row)

    summary = build_summary(
        rows,
        run_id=out_path.stem,
        kind=kind,
        model=model_id,
        gpu=gpu,
        gen_settings=gen_settings,
        repeats=repeats,
        quality=quality,
        model_info=model_info,
    )
    summary_path = out_path.with_suffix(".json")
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== summary ===")
    print_summary(summary)
    print(f"results:  per-prompt={out_path}  summary={summary_path}")
    return summary
