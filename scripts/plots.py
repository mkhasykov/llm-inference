"""Per-method figures from results/*.json (matplotlib, saved as PNG).

The thesis story is method-by-method, not one big table, so this draws one
figure per question:
  * methods   — single-stream decode speed by method (where each one helps)
  * quant     — speed, VRAM, and perplexity across weight formats
  * batch     — throughput and VRAM vs batch size, per format (the batch sweep)
  * pareto    — speed vs quality and speed vs VRAM trade-offs

Each figure is skipped (with a note) if its data isn't present yet, so it works
on a partially-filled results dir.

    python scripts/plots.py
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from aggregate import load_results, speed_format  # noqa: E402
from plotstyle import FORMAT_LABEL, label, family_color, annotate_v, annotate_h, HIDE_CONFIGS  # noqa: E402

# bf16 baseline + each quant format for the format-comparison figures.
# 4-bit AWQ/GPTQ use the optimized Marlin kernels; the slower Triton variants
# are left out of the figures (Marlin is the 4-bit result we present).
QUANT_CONFIGS = [
    ("baseline_cache", "bf16"),
    ("quant_int8", "bnb-int8"),
    ("quant_fp4", "bnb-fp4"),
    ("quant_nf4", "bnb-nf4"),
    ("quant_awq_marlin", "AWQ-int4"),
    ("quant_gptq-int4_marlin", "GPTQ-int4"),
]


def dedupe_latest(speed):
    by = {}
    for s in speed:
        k = s["kind"]
        if k not in by or s.get("timestamp_utc", "") > by[k].get("timestamp_utc", ""):
            by[k] = s
    return list(by.values())


def to_rows(speed, quality):
    rows = []
    for s in speed:
        gs = s.get("gen_settings", {})
        fmt = speed_format(s)
        q = quality.get((s["model"], fmt))
        tps = s.get("tokens_per_sec") or {}
        vram = s.get("peak_vram_gb") or {}
        blk = s.get("mean_block_size") or {}
        rows.append({
            "config": s["kind"], "format": fmt, "batch": gs.get("batch_size", 1),
            "tps": tps.get("mean"), "tps_std": tps.get("std") or 0,
            "vram": vram.get("max"), "block": blk.get("mean"),
            "ppl": q["perplexity"]["perplexity"] if q else None,
            "mmlu": q["mmlu"]["acc"] if q else None,
            "gsm8k": q["gsm8k"]["exact_match"] if q else None,
        })
    return rows


def fig_methods(rows, out):
    r = [x for x in rows if x["batch"] == 1 and not x["config"].startswith("batch_")
         and x["config"] not in HIDE_CONFIGS and x["tps"]]
    if len(r) < 2:
        return None
    r.sort(key=lambda x: x["tps"])
    fig, ax = plt.subplots(figsize=(8.5, max(3, 0.5 * len(r))))
    bars = ax.barh([label(x["config"]) for x in r], [x["tps"] for x in r],
                   xerr=[x["tps_std"] for x in r],
                   color=[family_color(x["config"]) for x in r])
    annotate_h(ax, bars, "{:.0f}")
    ax.set_xlim(0, max(x["tps"] for x in r) * 1.15)
    ax.set_xlabel("decode tokens/sec (batch=1)")
    ax.set_title("Single-stream speed by method")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    return out


def _quant_rows(rows):
    """bf16 baseline + one row per quant config at batch=1, in QUANT_CONFIGS order.

    Includes both the Triton and the Marlin 4-bit variants so the quant/pareto
    figures show that the 4-bit speed-up is kernel-dependent (same weights, same
    memory and quality, very different speed).
    """
    by = {x["config"]: x for x in rows if x["batch"] == 1}
    out = []
    for cfg, lbl in QUANT_CONFIGS:
        if cfg in by:
            r = dict(by[cfg]); r["qlabel"] = lbl
            out.append(r)
    return out


def fig_quant(rows, out):
    r = _quant_rows(rows)
    if len(r) < 2:
        return None
    labels = [x["qlabel"] for x in r]
    # speed + memory only; quality (ppl / MMLU / GSM8K) lives in fig_quality
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
    b0 = axes[0].bar(labels, [x["tps"] or 0 for x in r], color="steelblue")
    annotate_v(axes[0], b0, "{:.0f}")
    axes[0].set_ylabel("decode tok/s"); axes[0].set_title("Speed (higher = better)")
    b1 = axes[1].bar(labels, [x["vram"] or 0 for x in r], color="seagreen")
    annotate_v(axes[1], b1, "{:.1f}")
    axes[1].set_ylabel("peak VRAM (GB)"); axes[1].set_title("Memory (lower = better)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Quantization: speed and memory by format")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    return out


def fig_quality(rows, out):
    """Quality across formats on three axes: WikiText-2 perplexity (lower=better)
    plus MMLU and GSM8K accuracy (higher=better). Perplexity alone hides the real
    impact — e.g. GPTQ-int4 drops GSM8K while its perplexity barely moves."""
    r = _quant_rows(rows)
    r = [x for x in r if any(x[k] is not None for k in ("ppl", "mmlu", "gsm8k"))]
    if len(r) < 2:
        return None
    labels = [x["qlabel"] for x in r]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    ppl = [x["ppl"] for x in r]
    b0 = axes[0].bar(labels, [p or 0 for p in ppl], color="indianred")
    annotate_v(axes[0], b0, "{:.2f}")
    axes[0].set_ylabel("WikiText-2 perplexity")
    axes[0].set_title("Perplexity (lower = better)")
    if any(ppl):
        axes[0].set_ylim(0, max(p for p in ppl if p) * 1.2)

    for ax, key, color, title in [
        (axes[1], "mmlu", "steelblue", "MMLU accuracy (higher = better)"),
        (axes[2], "gsm8k", "seagreen", "GSM8K accuracy (higher = better)"),
    ]:
        vals = [(x[key] * 100 if x[key] is not None else 0) for x in r]
        bars = ax.bar(labels, vals, color=color)
        annotate_v(ax, bars, "{:.1f}")
        ax.set_ylabel("accuracy, %")
        ax.set_title(title)
        if vals and vals[0]:  # dashed line at the bf16 baseline (first config)
            ax.axhline(vals[0], color="k", ls="--", lw=1)
        ax.set_ylim(0, (max(vals) if any(vals) else 1) * 1.18)

    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Quantization quality: perplexity, MMLU, GSM8K by format")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    return out


def fig_batch(rows, out):
    r = [x for x in rows if x["config"].startswith("batch_") and x["tps"]]
    if not r:
        return None
    fmts = {}
    for x in r:
        fmts.setdefault(x["format"], []).append(x)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for fmt, pts in sorted(fmts.items()):
        pts.sort(key=lambda x: x["batch"])
        bs = [p["batch"] for p in pts]
        axes[0].plot(bs, [p["tps"] for p in pts], "o-", label=FORMAT_LABEL.get(fmt, fmt))
        axes[1].plot(bs, [p["vram"] for p in pts], "o-", label=FORMAT_LABEL.get(fmt, fmt))
    axes[0].set_xlabel("batch size"); axes[0].set_ylabel("aggregate tok/s"); axes[0].set_title("Throughput vs batch")
    axes[1].set_xlabel("batch size"); axes[1].set_ylabel("peak VRAM (GB)"); axes[1].set_title("VRAM vs batch")
    for ax in axes:
        ax.legend(); ax.grid(True, alpha=0.3)
    fig.suptitle("Static batching sweep")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    return out


def fig_pareto(rows, out):
    r = _quant_rows(rows)
    have_ppl = [x for x in r if x["ppl"] is not None and x["tps"]]
    have_vram = [x for x in r if x["vram"] is not None and x["tps"]]
    if len(have_vram) < 2:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    if len(have_ppl) >= 2:
        for x in have_ppl:
            axes[0].scatter(x["ppl"], x["tps"])
            axes[0].annotate(x["qlabel"], (x["ppl"], x["tps"]),
                             textcoords="offset points", xytext=(5, 4), fontsize=8)
        axes[0].set_xlabel("perplexity (lower=better)"); axes[0].set_ylabel("decode tok/s")
        axes[0].set_title("Speed vs quality")
    else:
        axes[0].set_visible(False)
    for x in have_vram:
        axes[1].scatter(x["vram"], x["tps"])
        axes[1].annotate(x["qlabel"], (x["vram"], x["tps"]),
                         textcoords="offset points", xytext=(5, 4), fontsize=8)
    axes[1].set_xlabel("peak VRAM (GB)"); axes[1].set_ylabel("decode tok/s")
    axes[1].set_title("Speed vs memory")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle("Trade-off frontiers (by weight format)")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="results/plots")
    args = ap.parse_args()

    speed, quality = load_results(args.results_dir)
    if not speed:
        print(f"no speed summaries in {args.results_dir}/")
        return
    rows = to_rows(dedupe_latest(speed), quality)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "methods.png": fig_methods,
        "quant.png": fig_quant,
        "quality.png": fig_quality,
        "batch.png": fig_batch,
        "pareto.png": fig_pareto,
    }
    for name, fn in figures.items():
        result = fn(rows, out_dir / name)
        print(f"  {'wrote' if result else 'skip (insufficient data):'} {out_dir / name}")


if __name__ == "__main__":
    main()
