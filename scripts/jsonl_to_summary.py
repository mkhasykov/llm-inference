"""Build a results/<run>.json summary from a results/<run>.jsonl file.

Use when:
- migrating an old JSONL run to the new summary-only format;
- the summary schema changes and you want to regenerate without
  re-running the benchmark.
"""

import argparse
import json
from pathlib import Path

from summary import build_summary


def infer_kind(stem: str) -> str:
    if stem.startswith("manual_kv"):
        return "manual_kv"
    if stem.startswith("baseline"):
        return "baseline"
    return "unknown"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jsonl", type=Path, nargs="+", help="one or more per-prompt JSONL files")
    args = p.parse_args()

    for jsonl_path in args.jsonl:
        rows = [json.loads(line) for line in jsonl_path.open() if line.strip()]
        if not rows:
            print(f"skip empty: {jsonl_path}")
            continue
        summary = build_summary(
            rows,
            run_id=jsonl_path.stem,
            kind=infer_kind(jsonl_path.stem),
            model=rows[0]["model"],
            gpu=rows[0]["gpu"],
            gen_settings=rows[0]["gen_settings"],
        )
        out_path = jsonl_path.with_suffix(".json")
        with out_path.open("w") as f:
            json.dump(summary, f, indent=2)
        print(f"{jsonl_path} → {out_path}")


if __name__ == "__main__":
    main()
