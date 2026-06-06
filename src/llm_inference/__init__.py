"""Shared inference-benchmark harness.

Pure helpers extracted from the per-experiment scripts so that adding a new
optimization (or a new measurement axis) does not duplicate dataset loading,
CUDA-event timing, metric computation, repetition/variance aggregation, or
summary writing. Each benchmark script supplies only its generation strategy
(how the model is loaded, which cache, greedy vs sampling) and calls into
this package for everything else.
"""
