from __future__ import annotations

from app.config import MODEL_DEFAULTS


def estimate_cost_usd(model: str, input_tokens: int | None, output_tokens: int | None) -> float:
    """USD cost of one call. $0 for Ollama or any model without a listed price."""
    pricing = MODEL_DEFAULTS.pricing_usd_per_mtok.get(model)
    if pricing is None or input_tokens is None or output_tokens is None:
        return 0.0

    input_price_per_mtok, output_price_per_mtok = pricing
    return (input_tokens / 1_000_000) * input_price_per_mtok + (output_tokens / 1_000_000) * output_price_per_mtok
