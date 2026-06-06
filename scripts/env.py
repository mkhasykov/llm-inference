"""GPU runtime-state snapshot, for detecting thermal/clock drift.

A long matrix run (hours) can throttle the GPU as it heats up, so later
configs look slower for reasons unrelated to the method. Recording temperature
and clocks per run lets us spot drift and report the measurement conditions.
"""

import subprocess


def gpu_state(index: int = 0) -> dict | None:
    """Snapshot temperature / clocks / power via nvidia-smi. Returns None if
    nvidia-smi is unavailable (e.g. some WSL setups), so callers can degrade
    gracefully."""
    query = "temperature.gpu,clocks.gr,clocks.mem,power.draw"
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}",
             "--format=csv,noheader,nounits", "-i", str(index)],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None

    parts = [p.strip() for p in out.split(",")]
    if len(parts) < 4:
        return None

    def _f(x):
        try:
            return float(x)
        except ValueError:
            return None

    return {
        "temperature_c": _f(parts[0]),
        "clock_graphics_mhz": _f(parts[1]),
        "clock_memory_mhz": _f(parts[2]),
        "power_w": _f(parts[3]),
    }
