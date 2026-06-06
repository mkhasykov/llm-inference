"""Dataset loading and prompt construction (MT-Bench)."""

import json
from pathlib import Path


def load_dataset(path: Path, limit: int) -> list[dict]:
    """Read up to `limit` MT-Bench questions from a JSONL file."""
    items = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
            if len(items) >= limit:
                break
    return items


def build_prompt(tokenizer, user_text: str) -> str:
    """Apply the tokenizer chat template to a single user turn (MT-Bench
    turns[0]). Returns the formatted prompt string (not yet tokenized)."""
    messages = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
