"""
Token-to-rupiah cost conversion.

One pure function used in two places: the per-query metadata badge under an
answer and the cumulative session block in the sidebar. Rates are a static
snapshot in ``config`` (decided D-6.3): predictable, no external dependency or
latency, and sub-percent FX accuracy is irrelevant for a capstone demo.
"""

from __future__ import annotations

from src import config

_PER_MILLION = 1_000_000


def tokens_to_idr(input_tokens: int, output_tokens: int) -> float:
    """Convert an input/output token count into an IDR cost.

    gpt-4o-mini bills input and output at different per-million-token rates,
    so the two are priced separately, summed in USD, then converted with the
    static USD->IDR rate. Pure: same inputs always give the same number, which
    keeps the demo's cost figures reproducible.
    """
    usd = (
        input_tokens / _PER_MILLION * config.INPUT_TOKEN_PRICE_USD_PER_1M
        + output_tokens / _PER_MILLION * config.OUTPUT_TOKEN_PRICE_USD_PER_1M
    )
    return usd * config.USD_TO_IDR
