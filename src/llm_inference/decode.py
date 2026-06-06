"""Explicit prefill + greedy decode loop, shared by the manual-cache scripts.

No model.generate(): prefill is one forward pass over the prompt; decode is N
forward passes over a single new token, mutating the cache in place. The only
thing that varies between experiments is the cache object passed in
(DynamicCache vs our PreallocatedKVCache), so the loop lives here once.

Call inside `torch.inference_mode()`, after `begin_measure()` has recorded the
start event. Records one CUDA event per generated token (the first is the
prefill end → TTFT) and returns (events, n_generated).
"""

import torch


def manual_decode(model, prompt_ids, cache, max_new_tokens: int, eos_ids: set[int]):
    # Prefill: full prompt → logits for last position → first new token.
    out = model(input_ids=prompt_ids, past_key_values=cache, use_cache=True)
    next_token = out.logits[:, -1:, :].argmax(-1)

    first_ev = torch.cuda.Event(enable_timing=True)
    first_ev.record()
    events = [first_ev]

    if int(next_token.item()) in eos_ids:
        return events, 1

    # Decode loop: one new token per step.
    for _ in range(max_new_tokens - 1):
        out = model(input_ids=next_token, past_key_values=cache, use_cache=True)
        next_token = out.logits[:, -1:, :].argmax(-1)

        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        events.append(ev)

        if int(next_token.item()) in eos_ids:
            break

    return events, len(events)
