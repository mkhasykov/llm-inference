"""Roofline analysis: measured decode throughput vs the hardware ceiling.

Single-stream decode reads the full weight set from VRAM to produce each
token, so it is memory-bandwidth-bound: the ceiling is

    tokens/sec_max = memory_bandwidth / bytes_read_per_token

with bytes_read_per_token ≈ the model's weight footprint. Comparing the
measured rate to this ceiling gives MBU (memory-bandwidth utilization) — how
much of the only resource that matters we actually use. The arithmetic
intensity (FLOP per byte) vs the GPU ridge point confirms the regime:
decode (~2/bytes_per_param FLOP/byte) sits far below the ridge → memory-bound;
prefill, with big matmuls, is the compute-bound counterpart.

Weight footprint is taken from the run's recorded `model_info.param_bytes`
when present (exact, incl. the packed size of quantized weights); otherwise it
falls back to a known param count × analytic bytes-per-param.
"""

# Match by substring of torch.cuda.get_device_name(0).
KNOWN_GPUS = {
    "RTX 3090": {"mem_bandwidth_gb_s": 936.2, "peak_fp16_tflops": 71.0},
    "RTX 4090": {"mem_bandwidth_gb_s": 1008.0, "peak_fp16_tflops": 165.2},
    "A100": {"mem_bandwidth_gb_s": 1555.0, "peak_fp16_tflops": 312.0},
}

# Fallback param counts when a summary predates model_info.
KNOWN_MODEL_PARAMS = {
    "Qwen/Qwen2.5-1.5B-Instruct": 1_543_714_304,
}

# Analytic effective weight bytes per parameter.
BYTES_PER_PARAM = {
    "float32": 4.0,
    "bfloat16": 2.0,
    "float16": 2.0,
    "int8": 1.0,
    "nf4": 0.5,
    "fp4": 0.5,
}


def gpu_specs(name: str, *, bandwidth_gb_s=None, peak_fp16_tflops=None) -> dict:
    spec = {"mem_bandwidth_gb_s": None, "peak_fp16_tflops": None}
    for key, known in KNOWN_GPUS.items():
        if key in name:
            spec = dict(known)
            break
    if bandwidth_gb_s is not None:
        spec["mem_bandwidth_gb_s"] = bandwidth_gb_s
    if peak_fp16_tflops is not None:
        spec["peak_fp16_tflops"] = peak_fp16_tflops
    return spec


def _bytes_per_param(gen_settings: dict) -> float | None:
    # Quantization scheme wins over compute dtype.
    if gen_settings.get("quant"):
        return BYTES_PER_PARAM.get(gen_settings["quant"])
    dtype = gen_settings.get("dtype") or gen_settings.get("compute_dtype")
    return BYTES_PER_PARAM.get(dtype)


def _weight_footprint(summary: dict, bpp_analytic: float | None) -> tuple[int | None, str]:
    """Bytes of weights read per decode step → (weight_bytes, source).

    Prefer the recorded `param_bytes` (exact real footprint, including the
    packed quantized weights + scales + full-precision embeddings). Fall back
    to a known logical param count × analytic bytes-per-param. Note: for a
    quantized model `model_info.n_params` is the PACKED element count, so we
    never derive bytes-per-param from it — that comes analytically from the
    weight format below.
    """
    info = summary.get("model_info")
    if info and info.get("param_bytes"):
        return info["param_bytes"], "measured"
    n_params = KNOWN_MODEL_PARAMS.get(summary.get("model"))
    if n_params and bpp_analytic:
        return int(n_params * bpp_analytic), "analytic"
    return None, "unknown"


def analyze(summary: dict, *, bandwidth_gb_s=None, peak_fp16_tflops=None) -> dict:
    spec = gpu_specs(
        summary.get("gpu", ""),
        bandwidth_gb_s=bandwidth_gb_s,
        peak_fp16_tflops=peak_fp16_tflops,
    )
    bw = spec["mem_bandwidth_gb_s"]
    # bytes-per-param is a property of the weight FORMAT (used for arithmetic
    # intensity); the footprint used for the ceiling is the real byte count.
    bpp = _bytes_per_param(summary.get("gen_settings", {}))
    weight_bytes, source = _weight_footprint(summary, bpp)
    tps = summary.get("tokens_per_sec")
    achieved = tps["mean"] if tps else None

    out = {
        "run_id": summary.get("run_id"),
        "kind": summary.get("kind"),
        "weight_gb": round(weight_bytes / 1e9, 3) if weight_bytes else None,
        "bytes_per_param": bpp,
        "footprint_source": source,
        "bandwidth_gb_s": bw,
        "achieved_tok_s": achieved,
        "ceiling_tok_s": None,
        "mbu_pct": None,
        "arith_intensity": round(2.0 / bpp, 2) if bpp else None,
        "ridge_point": (
            round(spec["peak_fp16_tflops"] * 1e12 / (bw * 1e9), 1)
            if bw and spec["peak_fp16_tflops"]
            else None
        ),
        "regime": None,
    }

    if bw and weight_bytes:
        ceiling = (bw * 1e9) / weight_bytes  # tokens/sec
        out["ceiling_tok_s"] = round(ceiling, 1)
        if achieved:
            out["mbu_pct"] = round(100.0 * achieved / ceiling, 1)
    if out["arith_intensity"] is not None and out["ridge_point"] is not None:
        out["regime"] = (
            "memory-bound" if out["arith_intensity"] < out["ridge_point"] else "compute-bound"
        )
    return out


def format_table(analyses: list[dict]) -> str:
    header = (
        f"{'run':<26} {'weights':>8} {'achieved':>9} {'ceiling':>9} "
        f"{'MBU':>6} {'intens':>7} {'regime':>13}"
    )
    lines = [header, "-" * len(header)]
    for a in analyses:
        wt = f"{a['weight_gb']:.2f}GB" if a["weight_gb"] is not None else "?"
        ach = f"{a['achieved_tok_s']:.1f}" if a["achieved_tok_s"] is not None else "?"
        ceil = f"{a['ceiling_tok_s']:.0f}" if a["ceiling_tok_s"] is not None else "?"
        mbu = f"{a['mbu_pct']:.1f}%" if a["mbu_pct"] is not None else "?"
        inten = f"{a['arith_intensity']:.1f}" if a["arith_intensity"] is not None else "?"
        regime = a["regime"] or "?"
        lines.append(
            f"{a['run_id'][:26]:<26} {wt:>8} {ach:>9} {ceil:>9} "
            f"{mbu:>6} {inten:>7} {regime:>13}"
        )
    return "\n".join(lines)
