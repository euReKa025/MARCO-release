from __future__ import annotations

import json
from typing import Any, Optional

from verl.interactions.base import BaseInteraction

from marco.prompts.system_prompt import PROMPT_MODE_THINK_ANSWER
from marco.verl.interaction.runtime import evaluate_marco_response_turn
from marco.verl.interaction.trajectory import bootstrap_trajectory_state
from marco.verl.reward.settlement import settle_trajectory_rewards_from_turn_payloads
from marco.utils import get_env_float


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _set_trace_field_if_present(target: dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return
        target[key] = text
        return
    target[key] = value


def _extract_instruction_from_state_payload(state_payload: dict[str, Any]) -> str | None:
    instruction = state_payload.get("instruction")
    if isinstance(instruction, str) and instruction.strip():
        return instruction.strip()
    messages = state_payload.get("messages")
    if not isinstance(messages, list):
        return None
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                text = content.strip()
                if text:
                    return text
    return None


def _build_validation_trace_turn(
    turn_payload: dict[str, Any],
    *,
    turn_index: int,
    feedback_message: str | None = None,
) -> dict[str, Any]:
    turn_id = int(turn_payload.get("turn_id", turn_index + 1) or (turn_index + 1))
    turn_result = turn_payload.get("turn_result") if isinstance(turn_payload.get("turn_result"), dict) else {}
    trace_turn: dict[str, Any] = {"turn_id": turn_id}
    _set_trace_field_if_present(trace_turn, "output_response", turn_payload.get("response_text"))
    if "response_length" in turn_payload:
        trace_turn["response_length"] = int(turn_payload.get("response_length", 0) or 0)
    _set_trace_field_if_present(
        trace_turn,
        "candidate_smiles",
        turn_result.get("candidate_smiles") if isinstance(turn_result, dict) else None,
    )
    if "similarity" in turn_payload:
        trace_turn["similarity"] = _to_float(turn_payload.get("similarity", 0.0), 0.0)
    if isinstance(turn_result.get("predictions"), dict) and turn_result.get("predictions"):
        trace_turn["predictions"] = dict(turn_result.get("predictions"))
    if "progress_score" in turn_payload:
        trace_turn["progress_score"] = _to_float(turn_payload.get("progress_score", 0.0), 0.0)
    if isinstance(turn_result.get("directional_improvements"), dict) and turn_result.get("directional_improvements"):
        trace_turn["directional_improvements"] = dict(turn_result.get("directional_improvements"))
    if isinstance(turn_result.get("active_property_names"), list) and turn_result.get("active_property_names"):
        trace_turn["active_property_names"] = list(turn_result.get("active_property_names"))
    trace_turn["met_all_targets"] = bool(turn_payload.get("met_all_targets", False))
    _set_trace_field_if_present(trace_turn, "invalid_type", turn_payload.get("invalid_type"))
    _set_trace_field_if_present(trace_turn, "stop_reason", turn_payload.get("stop_reason"))
    _set_trace_field_if_present(trace_turn, "error_detail", turn_payload.get("error_detail"))
    _set_trace_field_if_present(trace_turn, "feedback", feedback_message)
    return trace_turn


def _build_validation_trace_json(
    *,
    trajectory_id: str,
    trace_context: dict[str, Any],
    trace_turns: list[dict[str, Any]],
) -> str:
    trace_payload: dict[str, Any] = {
        "trajectory_id": str(trajectory_id),
        "turns": list(trace_turns),
    }
    _set_trace_field_if_present(trace_payload, "episode_id", trace_context.get("episode_id"))
    _set_trace_field_if_present(trace_payload, "group_id", trace_context.get("group_id"))
    _set_trace_field_if_present(trace_payload, "subtask", trace_context.get("subtask"))
    _set_trace_field_if_present(trace_payload, "x0_smiles", trace_context.get("x0_smiles"))
    _set_trace_field_if_present(trace_payload, "instruction", trace_context.get("instruction"))
    if isinstance(trace_context.get("properties"), list):
        trace_payload["properties"] = list(trace_context.get("properties"))
    if "max_turns" in trace_context:
        try:
            trace_payload["max_turns"] = int(trace_context.get("max_turns"))
        except (TypeError, ValueError):
            pass
    return json.dumps(trace_payload, ensure_ascii=False, separators=(",", ":"))


def _snapshot_trace_context(state_payload: dict[str, Any]) -> dict[str, Any]:
    context: dict[str, Any] = {}
    _set_trace_field_if_present(context, "episode_id", state_payload.get("episode_id"))
    _set_trace_field_if_present(context, "group_id", state_payload.get("group_id"))
    _set_trace_field_if_present(context, "subtask", state_payload.get("subtask"))
    _set_trace_field_if_present(context, "x0_smiles", state_payload.get("x0_smiles"))
    _set_trace_field_if_present(context, "instruction", _extract_instruction_from_state_payload(state_payload))
    property_targets = state_payload.get("property_targets")
    if isinstance(property_targets, list):
        context["properties"] = list(property_targets)
    if "max_turns" in state_payload:
        try:
            context["max_turns"] = int(state_payload.get("max_turns"))
        except (TypeError, ValueError):
            pass
    return context


def _summarize_marco_turn_payloads(turn_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    if not turn_payloads:
        return {
            "success": 0.0,
            "constraint_success": 0.0,
            "selected_similarity": 0.0,
            "turns_used": 0,
            "invalid_turns": 0,
            "oracle_calls": 0,
        }

    similarity_min = get_env_float("MARCO_SIMILARITY_TARGET_LOW", 0.6)
    similarity_copy_threshold = get_env_float("MARCO_SIMILARITY_COPY_THRESHOLD", 0.99)
    horizon_turn = turn_payloads[-1]
    success_turn: dict[str, Any] | None = None
    for turn_payload in turn_payloads:
        if bool(turn_payload.get("met_all_targets", False)):
            success_turn = turn_payload
            break

    selected_turn = success_turn or horizon_turn
    selected_similarity = _to_float(selected_turn.get("similarity", 0.0), 0.0)
    success = 1.0 if success_turn is not None else 0.0
    constraint_success = (
        1.0
        if any(
            bool(turn_payload.get("met_all_targets", False))
            and similarity_min <= _to_float(turn_payload.get("similarity", 0.0), 0.0) < similarity_copy_threshold
            for turn_payload in turn_payloads
        )
        else 0.0
    )

    invalid_turns = sum(
        1
        for turn_payload in turn_payloads
        if str(turn_payload.get("invalid_type", "")) in {"invalid_format", "invalid_parse", "invalid_eval"}
    )
    oracle_calls = sum(
        1
        for turn_payload in turn_payloads
        if str(turn_payload.get("invalid_type", "")) in {"ok", "invalid_eval", "env_error"}
    )
    turns_used = int(horizon_turn.get("turn_id", len(turn_payloads)) or len(turn_payloads))
    return {
        "success": success,
        "constraint_success": constraint_success,
        "selected_similarity": selected_similarity,
        "turns_used": turns_used,
        "invalid_turns": invalid_turns,
        "oracle_calls": oracle_calls,
    }


class MarcoTrajectoryInteraction(BaseInteraction):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._instance_dict: dict[str, dict[str, Any]] = {}

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        *,
        record: dict[str, Any] | None = None,
        initial_state_payload: dict[str, Any] | None = None,
        max_turns: int | None = None,
        prompt_mode: str = PROMPT_MODE_THINK_ANSWER,
        **kwargs: Any,
    ) -> str:
        instance_id = await super().start_interaction(instance_id=instance_id, **kwargs)
        if initial_state_payload is None:
            if record is None:
                raise ValueError("record or initial_state_payload is required for MARCO interaction")
            row = {"extra_info": {"canonical_record": record}}
            initial_state_payload = bootstrap_trajectory_state(
                row,
                max_turns=int(max_turns or 5),
                prompt_mode=prompt_mode,
            )

        self._instance_dict[instance_id] = {
            "state_payload": dict(initial_state_payload),
            "trace_context": _snapshot_trace_context(dict(initial_state_payload)),
            "turn_payloads": [],
            "trace_turns": [],
            "invalid_streak": 0,
        }
        return instance_id

    async def generate_response(
        self,
        instance_id: str,
        messages: list[dict[str, Any]],
        *,
        stop_reason: str | None = None,
        response_length: int = 0,
        rollout_max_response_len: int = 0,
        force_terminate: bool = False,
        **kwargs: Any,
    ) -> tuple[bool, str, float, dict[str, Any]]:
        del kwargs
        session = self._instance_dict[instance_id]
        state_payload = dict(session["state_payload"])

        assistant_message = ""
        for item in reversed(messages):
            if item.get("role") == "assistant":
                assistant_message = str(item.get("content", ""))
                break

        turn_id = int(state_payload.get("turn_id", 0)) + 1
        result = evaluate_marco_response_turn(
            state_payload=state_payload,
            response_text=assistant_message,
            turn_id=turn_id,
            response_length=int(response_length),
            stop_reason=stop_reason,
            invalid_streak=int(session.get("invalid_streak", 0)),
            rollout_max_response_len=int(rollout_max_response_len),
        )
        turn_result = dict(result["turn_result"] or {})
        forced_by_loop_limit = force_terminate and not bool(turn_result.get("should_stop", False))
        if forced_by_loop_limit:
            turn_result["should_stop"] = True
            turn_result["message"] = "agent_loop_limit"
        turn_payload = dict(result["turn_payload"] or {})
        turn_payload["turn_result"] = turn_result
        if forced_by_loop_limit:
            turn_payload["stop_reason"] = "agent_loop_limit"
        feedback_message = str(result["feedback_message"])
        trace_turn = _build_validation_trace_turn(
            turn_payload,
            turn_index=len(list(session["turn_payloads"])),
            feedback_message=feedback_message,
        )

        session["state_payload"] = dict(result["next_state"] or state_payload)
        session["invalid_streak"] = int(result["invalid_streak"])
        session["turn_payloads"].append(turn_payload)
        session["trace_turns"].append(trace_turn)

        should_stop = bool(turn_result.get("should_stop", False))
        additional_data: dict[str, Any] = {
            "turn_payload": turn_payload,
            "turn_result": turn_result,
            "feedback_message": feedback_message,
        }

        if should_stop:
            settlement = settle_trajectory_rewards_from_turn_payloads(
                list(session["turn_payloads"]),
                trajectory_id=instance_id,
            )
            additional_data.update(
                {
                    "trajectory_reward": float(settlement["trajectory_reward"]),
                    "trajectory_return": float(settlement["trajectory_reward"]),
                    "step_rewards": dict(settlement["step_rewards"]),
                    "reward_breakdowns": dict(settlement["breakdowns"]),
                }
            )
            summary = _summarize_marco_turn_payloads(list(session["turn_payloads"]))
            summary["validation_trace_json"] = _build_validation_trace_json(
                trajectory_id=instance_id,
                trace_context=dict(session["trace_context"]),
                trace_turns=list(session["trace_turns"]),
            )
            additional_data["reward_extra_info"] = summary

        next_user_content = "" if should_stop else str(result["next_user_content"] or "")
        return should_stop, next_user_content, 0.0, additional_data

    async def finalize_interaction(self, instance_id: str, **kwargs: Any) -> None:
        self._instance_dict.pop(instance_id, None)
