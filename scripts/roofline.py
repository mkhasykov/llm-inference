"""Roofline analysis over benchmark result summaries.

Reads one or more results/<run>.json summaries and reports, per run, the
memory-bandwidth ceiling on decode throughput, the achieved fraction (MBU),
and the arithmetic-intensity regime. This is an offline analysis — no GPU or
model load required; it works from the recorded summaries alone.

    python scripts/roofline.py results/*.json
    python scripts/roofline.py results/quant_nf4_*.json --bandwidth-gb-s 936.2

GPU specs are looked up from the recorded device name; override with
--bandwidth-gb-s / --peak-fp16-tflops for an unlisted card.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_inference.roofline import analyze, format_table


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("summaries", type=Path, nargs="+", help="results/<run>.json file(s)")
    p.add_argument("--bandwidth-gb-s", type=float, default=None, help="override GPU memory bandwidth")
    p.add_argument("--peak-fp16-tflops", type=float, default=None, help="override GPU peak fp16 TFLOPS")
    return p.parse_args()


def main():
    args = parse_args()
    analyses = []
    for path in args.summaries:
        summary = json.loads(path.read_text())
        analyses.append(
            analyze(
                summary,
                bandwidth_gb_s=args.bandwidth_gb_s,
                peak_fp16_tflops=args.peak_fp16_tflops,
            )
        )

    print(format_table(analyses))

    # One-line interpretation anchored on the first run with a known ceiling.
    ref = next((a for a in analyses if a["mbu_pct"] is not None), None)
    if ref:
        print(
            f"\nDecode is {ref['regime']} (intensity {ref['arith_intensity']} "
            f"<< ridge {ref['ridge_point']} FLOP/byte): throughput is capped by "
            f"reading {ref['weight_gb']}GB of weights per token, not by compute.\n"
            f"Headroom to the bandwidth ceiling = {100 - ref['mbu_pct']:.1f}%."
        )


if __name__ == "__main__":
    main()
