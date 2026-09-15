from __future__ import annotations

# USD per million tokens: (input, output). Source: platform.claude.com/docs pricing.
# Anything not listed here (including any Ollama model) is treated as $0.
_PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-sonnet-5": (2.0, 10.0),
}


def estimate_cost_usd(model: str, input_tokens: int | None, output_tokens: int | None) -> float:
    """USD cost of one call. $0 for Ollama or any model without a listed price."""
    pricing = _PRICING_USD_PER_MTOK.get(model)
    if pricing is None or input_tokens is None or output_tokens is None:
        return 0.0

    input_price_per_mtok, output_price_per_mtok = pricing
    return (input_tokens / 1_000_000) * input_price_per_mtok + (output_tokens / 1_000_000) * output_price_per_mtok
