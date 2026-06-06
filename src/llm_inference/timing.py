"""CUDA-event timing and per-generation metric computation.

Both execution paths (model.generate() with a streamer, and the manual
prefill+decode loop) produce the same raw material: a `gen_start` event, one
event per generated token, and a `gen_end` event. `finish_measure` turns that
into the metric dict, so the ~30-line latency/throughput block lives in one
place instead of being copied into every script.
"""

import statistics

import torch
from transformers.generation.streamers import BaseStreamer


class CudaEventStreamer(BaseStreamer):
    """Record a CUDA event for every token generate() emits.

    generate() calls put() once with the prompt input_ids, then once per
    newly generated token. We skip the first call and record an event for
    each subsequent token. Used only for timing — text is never decoded.
    """

    def __init__(self):
        self.events: list[torch.cuda.Event] = []
        self._saw_prompt = False

    def put(self, value):
        if not self._saw_prompt:
            self._saw_prompt = True
            return
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self.events.append(ev)

    def end(self):
        pass


def begin_measure() -> torch.cuda.Event:
    """Reset peak-VRAM tracking and record the generation start event."""
    torch.cuda.reset_peak_memory_stats()
    gen_start = torch.cuda.Event(enable_timing=True)
    gen_start.record()
    return gen_start


def finish_measure(
    gen_start: torch.cuda.Event,
    events: list[torch.cuda.Event],
    prompt_tokens: int,
    generated_tokens: int,
    extra: dict | None = None,
) -> dict:
    """Record gen_end, synchronize once, and derive the metric dict.

    `events` holds one timing event per generated token (the first is the
    prefill end → TTFT). Throughput is measured over the decode phase only
    (tokens 2..N), so prefill is excluded from tokens/sec.
    """
    gen_end = torch.cuda.Event(enable_timing=True)
    gen_end.record()
    torch.cuda.synchronize()

    peak_vram = int(torch.cuda.max_memory_allocated())
    total_ms = gen_start.elapsed_time(gen_end)
    token_ms = [gen_start.elapsed_time(ev) for ev in events]

    ttft_ms = token_ms[0] if token_ms else None

    if len(token_ms) >= 2:
        decode_intervals = [token_ms[i] - token_ms[i - 1] for i in range(1, len(token_ms))]
        decode_total_s = (token_ms[-1] - token_ms[0]) / 1000.0
        decode_tokens = len(token_ms) - 1
        tokens_per_sec = decode_tokens / decode_total_s if decode_total_s > 0 else None
        ms_per_token_mean = statistics.fmean(decode_intervals)
        ms_per_token_p50 = statistics.median(decode_intervals)
        sorted_ints = sorted(decode_intervals)
        p95_idx = max(0, int(round(0.95 * (len(sorted_ints) - 1))))
        ms_per_token_p95 = sorted_ints[p95_idx]
    else:
        tokens_per_sec = ms_per_token_mean = ms_per_token_p50 = ms_per_token_p95 = None

    metrics = {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "events_recorded": len(events),
        "total_ms": total_ms,
        "ttft_ms": ttft_ms,
        "tokens_per_sec": tokens_per_sec,
        "ms_per_token_mean": ms_per_token_mean,
        "ms_per_token_p50": ms_per_token_p50,
        "ms_per_token_p95": ms_per_token_p95,
        "peak_vram_bytes": peak_vram,
    }
    if extra:
        metrics.update(extra)
    return metrics
