"""Model/tokenizer loading and small model-introspection helpers."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(
    model_id: str,
    *,
    quantization_config=None,
    dtype="auto",
    device_map: str = "cuda",
):
    """Load a causal-LM and its tokenizer in eval mode.

    `quantization_config` (a BitsAndBytesConfig or None) is the only knob the
    quantization benchmark needs; everything else loads the same way as the
    fp16/bf16 baseline. When quantizing, callers typically pass
    `device_map="auto"` (bitsandbytes places the layers).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    kwargs = {"device_map": device_map}
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    else:
        kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return model, tokenizer


def dtype_str(model) -> str:
    """Compute dtype of the model's parameters as a short string (e.g.
    'bfloat16'). For quantized models this is the compute dtype of the
    non-quantized params, so callers should record the quant scheme too."""
    return str(next(model.parameters()).dtype).replace("torch.", "")


def model_stats(model) -> dict:
    """Parameter count and on-device weight footprint.

    `param_bytes` sums params + buffers as actually stored, so for a
    bitsandbytes model it reflects the packed 4-bit/8-bit weights (plus quant
    scales) — the real number of bytes read per decode step, which is what the
    roofline ceiling divides bandwidth by.
    """
    n_params = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    param_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    return {"n_params": int(n_params), "param_bytes": int(param_bytes)}


def get_eos_ids(model, tokenizer) -> set[int]:
    """Collect every token id that should stop generation. Manual decode
    loops need this because they don't go through generate()'s stopping
    criteria."""
    eos = getattr(model.generation_config, "eos_token_id", None)
    if eos is None:
        eos = tokenizer.eos_token_id
    if isinstance(eos, int):
        return {eos}
    return set(eos)
