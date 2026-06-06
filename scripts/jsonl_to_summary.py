"""Rebuild a results/<run>.json summary from a results/<run>.jsonl file.

Use when the summary schema changes and you want to regenerate without
re-running the benchmark. Expects current-schema per-prompt rows (aggregated
nested metrics + "repeats"); older flat-schema JSONLs are not supported.
"""

import argparse
import json
import re
from pathlib import Path

from summary import build_summary

# run_id stems are "<kind>_<UTC timestamp>"; strip the timestamp to get kind.
_TS_RE = re.compile(r"_\d{8}T\d{6}Z$")


def infer_kind(stem: str) -> str:
    return _TS_RE.sub("", stem) or "unknown"


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
            repeats=rows[0].get("repeats", 1),
        )
        out_path = jsonl_path.with_suffix(".json")
        with out_path.open("w") as f:
            json.dump(summary, f, indent=2)
        print(f"{jsonl_path} → {out_path}")


if __name__ == "__main__":
    main()
