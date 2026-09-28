from __future__ import annotations

from typing import Any


def build_score_payload(*, prompt: str, response: str, ground_truth: str, extra_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "response": response,
        "ground_truth": ground_truth,
        "extra_info": dict(extra_info),
    }

