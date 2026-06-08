"""Shared plot styling: readable labels, family colors, value annotations.

Imported by plots.py and plots_methods.py so every figure uses the same names,
colors, and number labels — for thesis-ready figures.
"""

import matplotlib as _mpl

_mpl.rcParams.update({
    "figure.dpi": 120,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
})

# config kind -> human-readable label
LABEL = {
    "baseline_cache": "bf16 + cache",
    "baseline_nocache": "bf16, no cache",
    "manual_kv": "manual KV-cache",
    "quant_int8": "bnb-int8",
    "quant_nf4": "bnb-nf4",
    "quant_fp4": "bnb-fp4",
    "quant_awq": "AWQ-int4",
    "quant_gptq-int4": "GPTQ-int4",
    "spec": "spec (draft 0.5B)",
    "spec_nf4": "spec + nf4 target",
    "spec_awq": "spec + AWQ target",
    "spec_lookup": "prompt-lookup",
}

# weight-format label (for batch/quant figures)
FORMAT_LABEL = {
    "none": "bf16", "bf16": "bf16", "bfloat16": "bf16",
    "int8": "bnb-int8", "nf4": "bnb-nf4", "fp4": "bnb-fp4",
    "awq": "AWQ-int4", "gptq-int4": "GPTQ-int4", "gptq-int8": "GPTQ-int8",
}

# method-family colors (consistent across figures)
GRAY, GREEN, ORANGE, BLUE = "#7f7f7f", "#2ca02c", "#ff7f0e", "#1f77b4"


def label(kind):
    return LABEL.get(kind, kind)


def family_color(kind):
    if kind == "manual_kv" or kind.startswith("baseline"):
        return GRAY
    if kind.startswith("quant"):
        return GREEN
    if kind.startswith("spec"):
        return ORANGE
    return BLUE


def annotate_v(ax, bars, fmt="{:.0f}", pad=2):
    """Value labels above vertical bars."""
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                    textcoords="offset points", xytext=(0, pad),
                    ha="center", va="bottom", fontsize=8)


def annotate_h(ax, bars, fmt="{:.2f}", pad=3):
    """Value labels at the end of horizontal bars."""
    for b in bars:
        w = b.get_width()
        ax.annotate(fmt.format(w), (w, b.get_y() + b.get_height() / 2),
                    textcoords="offset points", xytext=(pad, 0),
                    ha="left", va="center", fontsize=8)
