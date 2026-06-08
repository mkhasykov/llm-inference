"""Method-centric figures: each method vs the baseline, and variants within a
method compared to each other. Complements plots.py (quant/batch/pareto).

  vs_baseline.png   — speed and VRAM of every single-stream config relative to
                      the baseline (baseline_cache = 1.0x reference line)
  spec_variants.png — speculative-decoding variants (draft / prompt-lookup /
                      quantized-target) vs baseline, annotated with block size
  length.png        — KV-cache behavior vs generation length (cache vs no-cache
                      throughput, and the cache speedup that grows with length)

    python scripts/plots_methods.py --results-dir results/qwen7b \
        --length-dir results/length_sweep/qwen7b
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from aggregate import load_results  # noqa: E402


def dedupe_latest(speed):
    by = {}
    for s in speed:
        k = s["kind"]
        if k not in by or s.get("timestamp_utc", "") > by[k].get("timestamp_utc", ""):
            by[k] = s
    return list(by.values())


def single_stream(speed):
    """batch=1, non-batch configs with a tok/s number."""
    out = []
    for s in speed:
        gs = s.get("gen_settings", {})
        if gs.get("batch_size", 1) != 1 or s["kind"].startswith("batch_"):
            continue
        tps = (s.get("tokens_per_sec") or {}).get("mean")
        if not tps:
            continue
        out.append({
            "kind": s["kind"],
            "tps": tps,
            "vram": (s.get("peak_vram_gb") or {}).get("max"),
            "block": (s.get("mean_block_size") or {}).get("mean"),
        })
    return out


def fig_vs_baseline(rows, out, baseline="baseline_cache"):
    by = {r["kind"]: r for r in rows}
    base = by.get(baseline)
    if not base or not base["tps"]:
        return None
    items = sorted([r for r in rows if r["kind"] != baseline], key=lambda r: r["tps"] / base["tps"])
    names = [r["kind"] for r in items]
    spd = [r["tps"] / base["tps"] for r in items]
    vr = [(r["vram"] / base["vram"]) if r["vram"] and base["vram"] else 0 for r in items]

    fig, axes = plt.subplots(1, 2, figsize=(13, max(3, 0.5 * len(items))))
    axes[0].barh(names, spd, color=["seagreen" if x >= 1 else "indianred" for x in spd])
    axes[0].axvline(1, color="k", lw=1, ls="--")
    axes[0].set_xlabel(f"speed vs {baseline} (×)")
    axes[0].set_title("Speed relative to baseline (>1 = faster)")
    axes[1].barh(names, vr, color=["seagreen" if 0 < x <= 1 else "indianred" for x in vr])
    axes[1].axvline(1, color="k", lw=1, ls="--")
    axes[1].set_xlabel(f"VRAM vs {baseline} (×)")
    axes[1].set_title("Memory relative to baseline (<1 = less)")
    fig.suptitle(f"Each method vs baseline ({baseline}, batch=1)")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    return out


def fig_spec(rows, out):
    by = {r["kind"]: r for r in rows}
    order = ["baseline_cache", "spec_lookup", "spec", "spec_awq", "spec_nf4"]
    items = [by[k] for k in order if k in by]
    if len(items) < 2:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar([i["kind"] for i in items], [i["tps"] for i in items],
                  color=["gray"] + ["steelblue"] * (len(items) - 1))
    base = by.get("baseline_cache")
    if base:
        ax.axhline(base["tps"], color="k", ls="--", lw=1, label="baseline")
        ax.legend()
    for b, it in zip(bars, items):
        if it["block"]:
            ax.annotate(f"blk {it['block']:.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("decode tok/s (batch=1)")
    ax.set_title("Speculative-decoding variants vs baseline")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    return out


def fig_length(length_dir, out):
    data = {}  # kind -> {length: tps}
    for p in glob.glob(f"{length_dir}/*.json"):
        if p.endswith(".jsonl") or "summary_table" in p:
            continue
        s = json.load(open(p))
        tps = (s.get("tokens_per_sec") or {}).get("mean")
        L = s.get("gen_settings", {}).get("max_new_tokens")
        if tps and L:
            data.setdefault(s["kind"], {})[L] = tps
    if not data:
        return None
    lens = sorted({L for k in data for L in data[k]})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for kind in sorted(data):
        axes[0].plot(lens, [data[kind].get(L) for L in lens], "o-", label=kind)
    axes[0].set_xlabel("generated tokens"); axes[0].set_ylabel("decode tok/s")
    axes[0].set_title("Throughput vs generation length"); axes[0].legend(); axes[0].grid(alpha=.3)
    if "baseline_cache" in data and "baseline_nocache" in data:
        ratio = [data["baseline_cache"].get(L, 0) / data["baseline_nocache"].get(L, 1) for L in lens]
        axes[1].plot(lens, ratio, "o-", color="purple")
        axes[1].set_xlabel("generated tokens"); axes[1].set_ylabel("cache speedup (×)")
        axes[1].set_title("KV-cache advantage grows with length"); axes[1].grid(alpha=.3)
    else:
        axes[1].set_visible(False)
    fig.suptitle("KV-cache: behavior vs generation length")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--length-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out_dir = Path(args.out_dir or f"{args.results_dir}/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    speed, _ = load_results(args.results_dir)
    rows = single_stream(dedupe_latest(speed))
    for name, made in [
        ("vs_baseline.png", fig_vs_baseline(rows, out_dir / "vs_baseline.png")),
        ("spec_variants.png", fig_spec(rows, out_dir / "spec_variants.png")),
    ]:
        print(f"  {'wrote' if made else 'skip (insufficient data):'} {out_dir / name}")
    if args.length_dir:
        made = fig_length(args.length_dir, out_dir / "length.png")
        print(f"  {'wrote' if made else 'skip:'} {out_dir / 'length.png'}")


if __name__ == "__main__":
    main()
