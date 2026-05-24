"""Build the per-run summary dict written to results/<run>.json.

Imported by benchmark_baseline.py, manual_kv_loop.py, and
jsonl_to_summary.py. Not a runnable script.
"""

import datetime as dt
import statistics
import subprocess


def _agg(values: list[float]) -> dict | None:
    if not values:
        return None
    return {
        "mean": round(statistics.fmean(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
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


def build_summary(
    rows: list[dict],
    run_id: str,
    kind: str,
    model: str,
    gpu: str,
    gen_settings: dict,
) -> dict:
    tps = [r["tokens_per_sec"] for r in rows if r["tokens_per_sec"]]
    ttft = [r["ttft_ms"] for r in rows if r["ttft_ms"]]
    vram = [r["peak_vram_bytes"] for r in rows]

    summary = {
        "run_id": run_id,
        "kind": kind,
        "model": model,
        "gpu": gpu,
        "git_commit": _git_commit(),
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "n_prompts": len(rows),
        "gen_settings": gen_settings,
        "tokens_per_sec": _agg(tps),
        "ttft_ms": _agg(ttft),
        "peak_vram_gb": {"max": round(max(vram) / 1e9, 3)} if vram else None,
    }
    if rows and "cache_kv_bytes" in rows[0]:
        kv = [r["cache_kv_bytes"] for r in rows]
        summary["kv_cache_mb"] = {
            "max": round(max(kv) / 1e6, 2),
            "mean": round(statistics.fmean(kv) / 1e6, 2),
        }
    return summary


def print_summary(summary: dict) -> None:
    """Stdout report mirroring the JSON summary (so the run still prints
    a human-readable line block at the end)."""
    if summary["tokens_per_sec"]:
        s = summary["tokens_per_sec"]
        print(f"tokens/sec  mean={s['mean']:.2f}  min={s['min']:.2f}  max={s['max']:.2f}")
    if summary["ttft_ms"]:
        s = summary["ttft_ms"]
        print(f"ttft (ms)   mean={s['mean']:.2f}  min={s['min']:.2f}  max={s['max']:.2f}")
    if summary.get("peak_vram_gb"):
        print(f"peak VRAM   max={summary['peak_vram_gb']['max']:.2f}GB")
    if "kv_cache_mb" in summary:
        s = summary["kv_cache_mb"]
        print(f"KV cache    max={s['max']:.2f}MB  mean={s['mean']:.2f}MB")
