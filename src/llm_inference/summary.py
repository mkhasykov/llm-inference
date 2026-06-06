"""Repetition aggregation and per-run summary building.

Two levels of aggregation:
  1. `aggregate_repeats` — collapse R repeats of the SAME prompt into one row.
     Timing metrics become mean ± std (run-to-run jitter); peak VRAM is the
     max; structural values (token counts, cache bytes) are taken from the
     first repeat (greedy decode is deterministic).
  2. `build_summary` — collapse all per-prompt rows into the run summary:
     spread ACROSS prompts (mean/std/min/max) plus the mean within-prompt
     jitter, so significance of a method comparison can be judged.
"""

import datetime as dt
import statistics
import subprocess

# Metrics averaged over repeats (with std). Everything else is structural.
TIMING_KEYS = (
    "total_ms",
    "ttft_ms",
    "tokens_per_sec",
    "ms_per_token_mean",
    "ms_per_token_p50",
    "ms_per_token_p95",
)
# Metrics that are a peak, not an average — take the worst case over repeats.
MAX_KEYS = ("peak_vram_bytes",)


def _repeat_stat(values: list[float]) -> dict | None:
    if not values:
        return None
    n = len(values)
    return {
        "mean": round(statistics.fmean(values), 3),
        "std": round(statistics.stdev(values), 3) if n >= 2 else 0.0,
        "n": n,
        "values": [round(v, 3) for v in values],
    }


def aggregate_repeats(reps: list[dict]) -> dict:
    """Collapse R per-repeat metric dicts (from finish_measure) into one row."""
    first = reps[0]
    out: dict = {}
    for key in first:
        if key in TIMING_KEYS:
            vals = [r[key] for r in reps if r.get(key) is not None]
            out[key] = _repeat_stat(vals)
        elif key in MAX_KEYS:
            out[key] = max(r[key] for r in reps)
        else:
            out[key] = first[key]  # structural / deterministic
    return out


def _agg(values: list[float]) -> dict | None:
    if not values:
        return None
    n = len(values)
    return {
        "mean": round(statistics.fmean(values), 3),
        "std": round(statistics.stdev(values), 3) if n >= 2 else 0.0,
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "n": n,
    }


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _prompt_means(rows: list[dict], key: str) -> list[float]:
    out = []
    for r in rows:
        v = r.get(key)
        if isinstance(v, dict) and v.get("mean") is not None:
            out.append(v["mean"])
    return out


def _prompt_stds(rows: list[dict], key: str) -> list[float]:
    out = []
    for r in rows:
        v = r.get(key)
        if isinstance(v, dict) and v.get("std") is not None:
            out.append(v["std"])
    return out


def build_summary(
    rows: list[dict],
    *,
    run_id: str,
    kind: str,
    model: str,
    gpu: str,
    gen_settings: dict,
    repeats: int,
    quality: dict | None = None,
) -> dict:
    tps = _prompt_means(rows, "tokens_per_sec")
    ttft = _prompt_means(rows, "ttft_ms")
    vram = [r["peak_vram_bytes"] for r in rows if "peak_vram_bytes" in r]
    tps_jitter = _prompt_stds(rows, "tokens_per_sec")

    summary = {
        "run_id": run_id,
        "kind": kind,
        "model": model,
        "gpu": gpu,
        "git_commit": _git_commit(),
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "n_prompts": len(rows),
        "repeats": repeats,
        "gen_settings": gen_settings,
        "tokens_per_sec": _agg(tps),
        "ttft_ms": _agg(ttft),
        "peak_vram_gb": {"max": round(max(vram) / 1e9, 3)} if vram else None,
        # mean run-to-run std of tokens/sec across prompts — the noise floor
        # against which a method's speedup must be judged significant.
        "tokens_per_sec_jitter_mean": (
            round(statistics.fmean(tps_jitter), 3) if tps_jitter else None
        ),
    }

    if rows and isinstance(rows[0].get("cache_kv_bytes"), int):
        kv = [r["cache_kv_bytes"] for r in rows]
        summary["kv_cache_mb"] = {
            "max": round(max(kv) / 1e6, 2),
            "mean": round(statistics.fmean(kv) / 1e6, 2),
        }

    if quality is not None:
        summary["quality"] = quality

    return summary


def print_summary(summary: dict) -> None:
    """Human-readable stdout block mirroring the JSON summary."""
    r = summary.get("repeats", 1)
    if summary["tokens_per_sec"]:
        s = summary["tokens_per_sec"]
        print(f"tokens/sec  mean={s['mean']:.2f}  std={s['std']:.2f}  "
              f"min={s['min']:.2f}  max={s['max']:.2f}  (across {s['n']} prompts)")
    if summary.get("tokens_per_sec_jitter_mean") is not None:
        print(f"            run-to-run jitter (mean std over {r} repeats)="
              f"{summary['tokens_per_sec_jitter_mean']:.2f}")
    if summary["ttft_ms"]:
        s = summary["ttft_ms"]
        print(f"ttft (ms)   mean={s['mean']:.2f}  std={s['std']:.2f}  "
              f"min={s['min']:.2f}  max={s['max']:.2f}")
    if summary.get("peak_vram_gb"):
        print(f"peak VRAM   max={summary['peak_vram_gb']['max']:.2f}GB")
    if "kv_cache_mb" in summary:
        s = summary["kv_cache_mb"]
        print(f"KV cache    max={s['max']:.2f}MB  mean={s['mean']:.2f}MB")
    if summary.get("quality"):
        q = summary["quality"]
        print(f"quality     perplexity={q['perplexity']:.3f} "
              f"({q['corpus']}, {q['n_tokens']} tok)")
