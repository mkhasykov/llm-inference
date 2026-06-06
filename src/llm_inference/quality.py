"""Quality metric: perplexity on WikiText-2.

Quantization (and any lossy optimization) trades quality for speed/VRAM, so a
quality axis is mandatory to compare methods fairly — a faster run that
degrades the model is not a free win. Perplexity on WikiText-2 is the standard
LM-quality proxy: deterministic, cheap, no judge model, and comparable to
published numbers.

Sliding-window, token-weighted recipe (the accepted HF formulation): score
each token exactly once, weighting each window's mean NLL by the number of
tokens it actually scores, then exponentiate the corpus-level mean NLL.
"""

import torch


def load_wikitext_text() -> str:
    """Concatenate the WikiText-2 (raw) test split into one string."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return "\n\n".join(t for t in ds["text"] if t.strip())


def compute_perplexity(
    model,
    tokenizer,
    text: str,
    *,
    window: int = 1024,
    stride: int = 512,
    max_tokens: int | None = None,
    device=None,
) -> dict:
    """Sliding-window perplexity over `text`.

    `window` is the context length per forward pass; `stride` is how far the
    window advances (stride < window overlaps, giving each scored token more
    context). `max_tokens` caps corpus length for quick runs; None = full.
    """
    if device is None:
        device = model.get_input_embeddings().weight.device

    input_ids_full = tokenizer(text, return_tensors="pt").input_ids
    if max_tokens is not None:
        input_ids_full = input_ids_full[:, :max_tokens]
    seq_len = input_ids_full.shape[1]

    nll_sum = 0.0
    n_tokens = 0
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + window, seq_len)
        trg_len = end - prev_end  # tokens scored for the first time in this window
        input_ids = input_ids_full[:, begin:end].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100  # only score the fresh tail

        with torch.inference_mode():
            out = model(input_ids, labels=target_ids)

        # out.loss is mean NLL over the scored tokens of this window.
        nll_sum += out.loss.double().item() * trg_len
        n_tokens += trg_len
        prev_end = end
        if end == seq_len:
            break

    ppl = float(torch.exp(torch.tensor(nll_sum / n_tokens)))
    return {
        "perplexity": round(ppl, 4),
        "corpus": "wikitext-2-raw-v1:test",
        "n_tokens": n_tokens,
        "window": window,
        "stride": stride,
    }
