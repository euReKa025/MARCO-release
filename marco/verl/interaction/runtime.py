from __future__ import annotations

import math
from typing import Any

from marco.reward.reward_fn import score_single_sample
from marco.trainer_types import Sample
from marco.utils import get_env_float, get_env_int


def _extract_followup_user_content(next_state: dict[str, Any]) -> str | None:
    messages = list(next_state.get("messages") or [])
    if not messages:
        return None
    last_message = messages[-1]
    if not isinstance(last_message, dict) or last_message.get("role") != "user":
        return None
    return str(last_message.get("content", ""))


def _coerce_sample_status(stop_reason: str | None) -> Sample.Status:
    return Sample.Status.TRUNCATED if stop_reason == "length" else Sample.Status.COMPLETED


def _maybe_apply_t3_stop(
    *,
    turn_result: dict[str, Any],
    response_length: int,
    invalid_streak: int,
    consecutive_invalid_limit: int,
    truncated_response_fraction: float,
    rollout_max_response_len: int,
) -> tuple[dict[str, Any], int]:
    status = str(turn_result.get("status", "ok"))
    is_model_invalid = status in {"invalid_format", "invalid_parse", "invalid_eval"}
    next_invalid_streak = invalid_streak + 1 if is_model_invalid else 0

    truncated_threshold = 0
    if rollout_max_response_len > 0 and truncated_response_fraction > 0.0:
        truncated_threshold = max(1, int(math.ceil(float(rollout_max_response_len) * truncated_response_fraction)))
    is_truncated_invalid = (
        status == "invalid_format"
        and str(turn_result.get("error_detail", "")) == "truncated_without_closed_answer_tag"
        and truncated_threshold > 0
        and response_length >= truncated_threshold
    )

    if is_truncated_invalid:
        updated = dict(turn_result)
        updated["should_stop"] = True
        updated["message"] = "t3_truncated_invalid"
        return updated, next_invalid_streak

    if consecutive_invalid_limit > 0 and next_invalid_streak >= consecutive_invalid_limit:
        updated = dict(turn_result)
        updated["should_stop"] = True
        updated["message"] = "t3_invalid_streak"
        return updated, next_invalid_streak

    return turn_result, next_invalid_streak


def evaluate_marco_response_turn(
    *,
    state_payload: dict[str, Any],
    response_text: str,
    turn_id: int,
    response_length: int,
    stop_reason: str | None,
    invalid_streak: int,
    rollout_max_response_len: int,
) -> dict[str, Any]:
    sample = Sample(
        response=response_text,
        response_length=int(response_length),
        tokens=[],
        metadata={
            "rl_state": state_payload,
            "turn_id": int(turn_id),
        },
    )
    sample.status = _coerce_sample_status(stop_reason)
    score_single_sample(sample)

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    turn_result = dict(metadata.get("turn_result") or {})
    turn_result, next_invalid_streak = _maybe_apply_t3_stop(
        turn_result=turn_result,
        response_length=int(response_length),
        invalid_streak=int(invalid_streak),
        consecutive_invalid_limit=get_env_int("MARCO_T3_CONSECUTIVE_INVALID_LIMIT", 0),
        truncated_response_fraction=get_env_float("MARCO_T3_TRUNCATED_RESPONSE_FRACTION", 0.0),
        rollout_max_response_len=int(rollout_max_response_len),
    )

    next_state = dict(metadata.get("next_state") or {})
    if turn_result.get("should_stop", False):
        messages = list(next_state.get("messages") or [])
        if messages and isinstance(messages[-1], dict) and messages[-1].get("role") == "user":
            next_state["messages"] = messages[:-1]

    turn_payload = {
        "turn_id": int(turn_id),
        "max_turns": int(state_payload.get("max_turns", turn_id)),
        "invalid_type": str(turn_result.get("status", "ok")),
        "similarity": float(turn_result.get("similarity", 0.0)),
        "progress_score": float(turn_result.get("progress_score", 0.0)),
        "met_all_targets": bool(turn_result.get("met_all_targets", False)),
        "error_detail": str(turn_result.get("error_detail", "")),
        "stop_reason": str(turn_result.get("message", "")),
        "response_length": int(response_length),
        "response_text": response_text,
        "turn_result": turn_result,
    }
    return {
        "turn_result": turn_result,
        "feedback_message": str(metadata.get("feedback_message", "")),
        "next_state": next_state,
        "next_user_content": _extract_followup_user_content(next_state),
        "invalid_streak": next_invalid_streak,
        "turn_payload": turn_payload,
    }
