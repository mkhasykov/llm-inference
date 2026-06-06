"""Run a matrix of benchmark cells sequentially, one subprocess per cell.

A cell is (label, script, extra_args). Each runs in its own process so GPU
memory is fully released between configs (loading six models in one process
would accumulate VRAM). A failing cell is logged and the matrix continues, so
e.g. a slow int8 run dying does not lose the others. Grow the experiment by
adding a row to MATRIX — that's the whole point of a driver.

    # full reference set on the dev model, with variance + perplexity
    python scripts/run_matrix.py --limit 80 --repeats 3 --quality --quality-max-tokens 0

    # just two cells
    python scripts/run_matrix.py --only manual_kv quant_nf4
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Reference matrix on the dev model (Qwen2.5-1.5B-Instruct). Each later
# milestone (GPTQ/AWQ, bigger models) adds rows here or overrides --model.
MATRIX = [
    ("baseline_nocache", "benchmark_baseline.py", ["--no-cache"]),
    ("manual_kv", "manual_kv_loop.py", []),
    ("static_kv", "static_kv_loop.py", []),
    ("quant_int8", "benchmark_quant.py", ["--quant", "int8"]),
    ("quant_nf4", "benchmark_quant.py", ["--quant", "nf4"]),
    ("quant_fp4", "benchmark_quant.py", ["--quant", "fp4"]),
]


def cell_format(extra: list[str]) -> str:
    """Weight format a cell exercises: the --quant value, else native bf16."""
    if "--quant" in extra:
        return extra[extra.index("--quant") + 1]
    return "bf16"


def run_quality_sweep(cells, args) -> None:
    """Measure perplexity once per UNIQUE weight format across the cells,
    instead of per speed run (lossless methods share the same number)."""
    formats = []
    for _label, _script, extra in cells:
        fmt = cell_format(extra)
        if fmt not in formats:
            formats.append(fmt)
    print(f"\n=== quality sweep: {formats} (once per format) ===", flush=True)
    for fmt in formats:
        cmd = [
            sys.executable, str(REPO / "scripts" / "eval_quality.py"),
            "--quant", ("none" if fmt == "bf16" else fmt),
            "--quality-max-tokens", str(args.quality_max_tokens),
            "--out-dir", args.out_dir,
        ]
        if args.model:
            cmd += ["--model", args.model]
        print(f"  quality[{fmt}]: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, cwd=REPO)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=80)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--quality", action="store_true")
    p.add_argument("--quality-max-tokens", type=int, default=0)
    p.add_argument("--model", default=None, help="override model for all cells")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--only", nargs="*", help="run only cells with these labels")
    return p.parse_args()


def main():
    args = parse_args()
    cells = MATRIX if not args.only else [c for c in MATRIX if c[0] in args.only]
    if not cells:
        print(f"no cells match --only {args.only}; labels: {[c[0] for c in MATRIX]}")
        sys.exit(1)

    # Speed cells never measure quality: perplexity depends only on
    # (model, weight format), not on the inference method/cache. It is
    # measured once per unique format by the quality sweep below.
    common = [
        "--limit", str(args.limit),
        "--repeats", str(args.repeats),
        "--max-new-tokens", str(args.max_new_tokens),
        "--out-dir", args.out_dir,
    ]
    if args.model:
        common += ["--model", args.model]

    print(f"matrix: {len(cells)} cells — {[c[0] for c in cells]}", flush=True)
    results = []
    for i, (label, script, extra) in enumerate(cells, 1):
        cmd = [sys.executable, str(REPO / "scripts" / script), *common, *extra]
        print(f"\n===[{i}/{len(cells)}] {label}\n     {' '.join(cmd)}", flush=True)
        start = time.time()
        rc = subprocess.run(cmd, cwd=REPO).returncode
        mins = (time.time() - start) / 60
        status = "ok" if rc == 0 else f"FAIL(rc={rc})"
        results.append((label, status, mins))
        print(f"===[{i}/{len(cells)}] {label}: {status} in {mins:.1f} min", flush=True)

    if args.quality:
        run_quality_sweep(cells, args)

    print("\n=== matrix done ===")
    for label, status, mins in results:
        print(f"  {label:<18} {status:<12} {mins:6.1f} min")


if __name__ == "__main__":
    main()
