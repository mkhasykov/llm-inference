"""Pre-allocated KV cache: one big buffer per layer, in-place writes.

Contrast with `DynamicCache`, which calls `torch.cat([past, new], dim=-2)`
on every decode step → reallocates the underlying storage every token.
Here we allocate `(batch, n_kv_heads, max_cache_len, head_dim)` once on the
first `update`, then write each new chunk into `[:, :, current_len:end, :]`.

Same `transformers.cache_utils.Cache` interface as `DynamicCache`, so it
drops into `model(past_key_values=...)` without further changes.
"""

import torch
from transformers.cache_utils import Cache, CacheLayerMixin


class PreallocatedKVLayer(CacheLayerMixin):
    """Single-layer pre-allocated buffer. K/V live in fixed tensors of
    length `max_cache_len`; `cumulative_length` tracks the valid prefix."""

    is_sliding = False

    def __init__(self, max_cache_len: int):
        super().__init__()
        self.max_cache_len = max_cache_len
        self.cumulative_length = 0

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        # First call gives us dtype/device and the per-layer shapes.
        self.dtype = key_states.dtype
        self.device = key_states.device
        batch, n_heads = key_states.shape[:2]
        k_head_dim = key_states.shape[-1]
        v_head_dim = value_states.shape[-1]

        self.keys = torch.zeros(
            (batch, n_heads, self.max_cache_len, k_head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.values = torch.zeros(
            (batch, n_heads, self.max_cache_len, v_head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.is_initialized = True

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        n_new = key_states.shape[-2]
        start = self.cumulative_length
        end = start + n_new
        if end > self.max_cache_len:
            raise RuntimeError(
                f"PreallocatedKVLayer overflow: tried to write tokens "
                f"[{start}:{end}] into buffer of size {self.max_cache_len}"
            )

        # In-place write. No torch.cat, no realloc.
        self.keys[:, :, start:end, :] = key_states
        self.values[:, :, start:end, :] = value_states
        self.cumulative_length = end

        # Return a view of the valid prefix. Mirrors DynamicCache semantics
        # (no padded slots leak into attention). Note: this view is
        # NON-CONTIGUOUS in the seq dim — see the perf note in README.
        return self.keys[:, :, :end, :], self.values[:, :, :end, :]

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        # Called BEFORE update — return the kv length we'll have AFTER
        # writing `query_length` new tokens. Mirrors DynamicLayer.
        return self.cumulative_length + query_length, 0

    def get_max_cache_shape(self) -> int:
        return self.max_cache_len


class PreallocatedKVCache(Cache):
    """Cache holding one `PreallocatedKVLayer` per transformer layer."""

    def __init__(self, max_cache_len: int, n_layers: int):
        layers = [PreallocatedKVLayer(max_cache_len) for _ in range(n_layers)]
        super().__init__(layers=layers)
        # `max_cache_len` is exposed by the base class as a property over layers.

    def buffer_bytes(self) -> int:
        """Total bytes pinned by the pre-allocated buffers (not just the
        valid prefix). This is the real VRAM footprint of the cache."""
        total = 0
        for layer in self.layers:
            if layer.is_initialized:
                total += layer.keys.numel() * layer.keys.element_size()
                total += layer.values.numel() * layer.values.element_size()
        return total

    def used_bytes(self) -> int:
        """Bytes that would be held by a `DynamicCache` at the same point —
        only the valid prefix. Useful for apples-to-apples comparison."""
        total = 0
        for layer in self.layers:
            if layer.is_initialized:
                n = layer.cumulative_length
                total += n * layer.keys.shape[1] * layer.keys.shape[-1] * layer.keys.element_size()
                total += n * layer.values.shape[1] * layer.values.shape[-1] * layer.values.element_size()
        return total
