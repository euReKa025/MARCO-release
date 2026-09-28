from __future__ import annotations


INVALID_MODEL_TYPES = {"invalid_format", "invalid_parse", "invalid_eval"}


def is_model_invalid(status: str) -> bool:
    return status in INVALID_MODEL_TYPES
