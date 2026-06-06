"""Aggregate per-run result JSONs into one comparison table.

Reads every results/*.json (speed summaries written by the benchmark scripts +
quality JSONs from eval_quality.py), dedupes speed runs by config (keeping the
latest), and joins each speed run to its quality numbers by (model, format) --
since quality is a property of the weights, shared by all lossless methods on
the same format. Prints a table and writes summary_table.{csv,json}.

Reads only what is present, so partial / incrementally-filled result dirs work.

    python scripts/aggregate.py
"""

import argparse
import csv
import glob
import json
from pathlib import Path


def fmt_stat(d, decimals=0):
    if not isinstance(d, dict) or d.get("mean") is None:
        return "-"
    return f"{d['mean']:.{decimals}f}±{d.get('std', 0):.{decimals}f}"


def norm_format(s):
    """Normalize a weight-format label to the quality 'quant' key space."""
    if s in (None, "bf16", "bfloat16", "float16", "auto", "none"):
        return "none"
    return s


def speed_format(summary):
    gs = summary.get("gen_settings", {})
    q = gs.get("quant")
    if q and q != "none":
        return q
    return norm_format(gs.get("format"))


def load_results(results_dir):
    speed, quality = [], {}
    for path in glob.glob(str(Path(results_dir) / "*.json")):
        with open(path) as f:
            obj = json.load(f)
        if obj.get("kind") == "quality":
            quality[(obj["model"], obj.get("quant", "none"))] = obj
        elif "tokens_per_sec" in obj:
            speed.append(obj)
    return speed, quality


def build_rows(speed, quality):
    by_kind = {}
    for s in speed:
        k = s["kind"]
        if k not in by_kind or s.get("timestamp_utc", "") > by_kind[k].get("timestamp_utc", ""):
            by_kind[k] = s

    rows = []
    for kind, s in by_kind.items():
        gs = s.get("gen_settings", {})
        fmt = speed_format(s)
        q = quality.get((s["model"], fmt))
        vram = s.get("peak_vram_gb") or {}
        rows.append({
            "config": kind,
            "format": fmt,
            "batch": gs.get("batch_size", 1),
            "tok/s": fmt_stat(s.get("tokens_per_sec"), 0),
            "ttft_ms": fmt_stat(s.get("ttft_ms"), 0),
            "VRAM_GB": vram.get("max", "-"),
            "block": fmt_stat(s.get("mean_block_size"), 2),
            "ppl": q["perplexity"]["perplexity"] if q else "-",
            "mmlu": q["mmlu"]["acc"] if q else "-",
            "gsm8k": q["gsm8k"]["exact_match"] if q else "-",
        })
    rows.sort(key=lambda r: (r["batch"], r["config"]))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="results/summary_table")
    args = ap.parse_args()

    speed, quality = load_results(args.results_dir)
    if not speed:
        print(f"no speed summaries found in {args.results_dir}/")
        return
    rows = build_rows(speed, quality)

    headers = ["config", "format", "batch", "tok/s", "ttft_ms", "VRAM_GB", "block", "ppl", "mmlu", "gsm8k"]
    try:
        from tabulate import tabulate
        print(tabulate(rows, headers={h: h for h in headers}, tablefmt="github"))
    except Exception:
        print("\t".join(headers))
        for r in rows:
            print("\t".join(str(r[h]) for h in headers))

    csv_path = Path(args.out).with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)
    json_path = Path(args.out).with_suffix(".json")
    with json_path.open("w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {csv_path} and {json_path}  ({len(rows)} configs)")


if __name__ == "__main__":
    main()
