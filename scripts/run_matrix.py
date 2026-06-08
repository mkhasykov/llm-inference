"""Run the full benchmark matrix, one subprocess per cell.

Each cell runs in its own process so GPU memory is fully released between
configs (loading a dozen models in one process would accumulate VRAM and
pollute peak_vram). A failing cell is logged and the matrix continues, so one
dying config does not lose the others. Grow the experiment by editing a list.

    # full matrix on the dev model
    python scripts/run_matrix.py --limit 40 --repeats 3

    # just a couple of speed cells, quick
    python scripts/run_matrix.py --only baseline_cache spec --limit 5 --repeats 1

    # the 7B run (speed + batch + quality)
    python scripts/run_matrix.py --model Qwen/Qwen2.5-7B-Instruct --limit 40 --repeats 3
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Single-stream speed cells: (label, script, extra_args). Common args
# (model/limit/repeats/max-new-tokens/out-dir + --fixed-length) are appended.
SPEED = [
    ("baseline_nocache", "benchmark_baseline.py", ["--no-cache"]),
    ("baseline_cache",   "benchmark_baseline.py", []),
    ("manual_kv",        "manual_kv_loop.py",     []),
    ("quant_int8",       "benchmark_quant.py",    ["--quant", "int8"]),
    ("quant_nf4",        "benchmark_quant.py",    ["--quant", "nf4"]),
    ("quant_fp4",        "benchmark_quant.py",    ["--quant", "fp4"]),
    ("quant_awq",        "benchmark_quant.py",    ["--quant", "awq"]),
    ("quant_gptq_int4",  "benchmark_quant.py",    ["--quant", "gptq-int4"]),
    ("spec",             "benchmark_spec.py",     []),
    ("spec_nf4",         "benchmark_spec.py",     ["--quant", "nf4"]),
    ("spec_awq",         "benchmark_spec.py",     ["--quant", "awq"]),
]

# Batch-throughput cells: (label, benchmark_batch.py args).
BATCH = [
    ("batch_bf16_b1",  ["--quant", "none", "--batch-size", "1"]),
    ("batch_bf16_b4",  ["--quant", "none", "--batch-size", "4"]),
    ("batch_bf16_b16", ["--quant", "none", "--batch-size", "16"]),
    ("batch_nf4_b4",   ["--quant", "nf4",  "--batch-size", "4"]),
    ("batch_nf4_b16",  ["--quant", "nf4",  "--batch-size", "16"]),
    ("batch_awq_b4",   ["--quant", "awq",  "--batch-size", "4"]),
    ("batch_awq_b16",  ["--quant", "awq",  "--batch-size", "16"]),
]

# Quality cells (eval_quality.py): the lossy formats + the bf16 reference.
QUALITY = ["none", "int8", "nf4", "awq", "gptq-int4"]


def run_cell(label: str, cmd: list[str], timeout: float | None = None) -> tuple[str, bool, float]:
    print(f"\n{'=' * 70}\n[cell] {label}\n  $ {' '.join(cmd)}\n{'=' * 70}", flush=True)
    t0 = time.perf_counter()
    try:
        rc = subprocess.run(cmd, cwd=REPO, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        # A hung cell (e.g. a stalled download or deadlock) is killed so the
        # matrix continues instead of blocking the whole unattended run.
        rc = -1
        print(f"[cell] {label}: TIMEOUT after {timeout:.0f}s — killed", flush=True)
    dt = time.perf_counter() - t0
    ok = rc == 0
    print(f"[cell] {label}: {'OK' if ok else f'FAILED (rc={rc})'} in {dt:.0f}s", flush=True)
    return label, ok, dt


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--limit", type=int, default=40)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--only", nargs="*", help="run only these cell labels")
    p.add_argument("--skip-cells", nargs="*", help="exclude these cell labels")
    p.add_argument("--cell-timeout", type=float, default=None,
                   help="kill a cell after this many seconds (safety net for unattended runs)")
    p.add_argument("--skip-speed", action="store_true")
    p.add_argument("--skip-batch", action="store_true")
    p.add_argument("--skip-quality", action="store_true")
    p.add_argument("--no-fixed-length", action="store_true",
                   help="do not force fixed-length generation (speed cells)")
    p.add_argument("--mmlu-limit", type=int, default=10)
    p.add_argument("--gsm8k-limit", type=int, default=200)
    p.add_argument("--ppl-max-tokens", type=int, default=50000)
    return p.parse_args()


def main():
    args = parse_args()
    py = sys.executable
    common = ["--model", args.model, "--limit", str(args.limit),
              "--repeats", str(args.repeats),
              "--max-new-tokens", str(args.max_new_tokens), "--out-dir", args.out_dir]
    fixed = [] if args.no_fixed_length else ["--fixed-length"]

    cells = []  # (label, cmd)
    if not args.skip_speed:
        for label, script, extra in SPEED:
            cells.append((label, [py, f"scripts/{script}", *common, *fixed, *extra]))
    if not args.skip_batch:
        for label, extra in BATCH:
            # benchmark_batch forces fixed length internally; no --fixed-length flag
            cells.append((label, [py, "scripts/benchmark_batch.py", *common, *extra]))
    if not args.skip_quality:
        for fmt in QUALITY:
            cells.append((f"quality_{fmt}", [
                py, "scripts/eval_quality.py", "--model", args.model, "--quant", fmt,
                "--out-dir", args.out_dir, "--mmlu-limit", str(args.mmlu_limit),
                "--gsm8k-limit", str(args.gsm8k_limit), "--ppl-max-tokens", str(args.ppl_max_tokens),
            ]))

    if args.only:
        cells = [(l, c) for l, c in cells if l in set(args.only)]
        missing = set(args.only) - {l for l, _ in cells}
        if missing:
            print(f"warning: unknown cell labels ignored: {sorted(missing)}", file=sys.stderr)
    if args.skip_cells:
        cells = [(l, c) for l, c in cells if l not in set(args.skip_cells)]

    print(f"matrix: {len(cells)} cells on {args.model} (limit={args.limit}, repeats={args.repeats})")
    results = [run_cell(label, cmd, args.cell_timeout) for label, cmd in cells]

    print(f"\n{'=' * 70}\nMATRIX SUMMARY\n{'=' * 70}")
    total = 0.0
    for label, ok, dt in results:
        total += dt
        print(f"  {'OK  ' if ok else 'FAIL'}  {label:<20} {dt:6.0f}s")
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results)} cells, {n_fail} failed, {total:.0f}s total")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
