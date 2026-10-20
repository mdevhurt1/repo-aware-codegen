"""
Budget fill rule.

Fill to cutoff: take units in rank order and append until the budget is reached; the
unit that crosses N is truncated at the token boundary so the block is <= N tokens.
"""

from __future__ import annotations

from typing import Any


def fill_to_cutoff(units: list[str], n: int, tokenizer: Any, sep: str = "\n") -> str:
    """Concatenate ranked units, truncating the crossing unit to fit n tokens.

    Count separators using the model tokenizer without special tokens.
    Reject negative budgets and return an empty string for a zero budget.
    """
    if n < 0:
        raise ValueError("Token budget must be non-negative.")

    if n == 0 or not units:
        return ""

    text = ""

    for index, unit in enumerate(units):
        candidate = text + sep + unit if index > 0 else unit
        token_ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]

        if len(token_ids) <= n:
            text = candidate
            continue

        # Keep the leading model tokens, including any separator tokens.
        kept_ids = token_ids[:n]

        while kept_ids:
            truncated = tokenizer.decode(
                kept_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

            # Decoding and re-encoding can change the number of tokens.
            used = len(tokenizer(truncated, add_special_tokens=False)["input_ids"])
            if used <= n:
                return truncated

            kept_ids = kept_ids[:-1]

        return ""

    return text