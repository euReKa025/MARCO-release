from __future__ import annotations

import json
from typing import Any, Optional

from verl.interactions.base import BaseInteraction

from marco.verl.reward.repo_single_turn import compute_score


def _extract_last_assistant_message(messages: list[dict[str, Any]]) -> str:
    for item in reversed(messages):
        if item.get("role") == "assistant":
            return str(item.get("content", ""))
    return ""


def _extract_base_prompt(record: dict[str, Any] | None, extra_info: dict[str, Any] | None) -> str:
    row = record if isinstance(record, dict) else {}
    prompt = row.get("prompt")
    if isinstance(prompt, list):
        for message in reversed(prompt):
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content", ""))

    info = extra_info if isinstance(extra_info, dict) else {}
    if isinstance(info.get("instruction"), str):
        return str(info.get("instruction"))
    return ""


def _normalize_reward_payload(payload: dict[str, Any]) -> dict[str, Any]:
    status = str(payload.get("status", "unknown"))
    similarity = float(payload.get("similarity", payload.get("similar", 0.0)) or 0.0)
    improvement = float(payload.get("improvement_score", payload.get("improve", 0.0)) or 0.0)
    total = float(payload.get("total", payload.get("reward", payload.get("score", 0.0))) or 0.0)
    improved_count = int(payload.get("improved_count", 0) or 0)
    required_count = int(payload.get("required_count", 0) or 0)

    normalized = dict(payload)
    normalized["status"] = status
    normalized["similarity"] = similarity
    normalized["similar"] = similarity
    normalized["improvement_score"] = improvement
    normalized["improve"] = improvement
    normalized["total"] = total
    normalized["reward"] = total
    normalized["score"] = total
    normalized["improved_count"] = improved_count
    normalized["required_count"] = required_count
    return normalized


def _reward_success(payload: dict[str, Any]) -> bool:
    status = str(payload.get("status", "unknown"))
    improved_count = int(payload.get("improved_count", 0) or 0)
    required_count = int(payload.get("required_count", 0) or 0)
    if required_count > 0:
        return status == "ok" and improved_count >= required_count
    return status == "ok" and float(payload.get("improve", 0.0) or 0.0) >= (1.0 - 1e-12)


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


def _first_nonempty_string(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
    return None


def _repo_reward_payload(turn_payload: dict[str, Any]) -> dict[str, Any]:
    payload = turn_payload.get("repo_reward")
    return payload if isinstance(payload, dict) else {}


def _build_validation_trace_turn(turn_payload: dict[str, Any], turn_index: int) -> dict[str, Any]:
    turn_id = int(turn_payload.get("turn_id", turn_index + 1) or (turn_index + 1))
    repo_reward = _repo_reward_payload(turn_payload)
    trace_turn: dict[str, Any] = {"turn_id": turn_id}
    _set_trace_field_if_present(trace_turn, "output_response", turn_payload.get("response_text"))
    if "response_length" in turn_payload:
        trace_turn["response_length"] = int(turn_payload.get("response_length", 0) or 0)
    if "similarity" in repo_reward or "similar" in repo_reward:
        trace_turn["similarity"] = _to_float(repo_reward.get("similarity", repo_reward.get("similar", 0.0)), 0.0)
    if "success" in repo_reward:
        trace_turn["success"] = _to_float(repo_reward.get("success", 0.0), 0.0)
    if "constraint_success" in repo_reward:
        trace_turn["constraint_success"] = _to_float(repo_reward.get("constraint_success", 0.0), 0.0)
    _set_trace_field_if_present(trace_turn, "stop_reason", turn_payload.get("stop_reason"))
    return trace_turn


def _build_validation_trace_json(*, trajectory_id: str, session: dict[str, Any], turn_payloads: list[dict[str, Any]]) -> str:
    reward_context = session.get("reward_context")
    context = reward_context if isinstance(reward_context, dict) else {}
    extra_info = context.get("extra_info")
    info = extra_info if isinstance(extra_info, dict) else {}
    ground_truth = context.get("ground_truth")
    gt = ground_truth if isinstance(ground_truth, dict) else {}

    trace_payload: dict[str, Any] = {
        "trajectory_id": str(trajectory_id),
        "turns": [_build_validation_trace_turn(turn_payload, idx) for idx, turn_payload in enumerate(turn_payloads)],
    }
    _set_trace_field_if_present(trace_payload, "sample_id", info.get("sample_id"))
    _set_trace_field_if_present(trace_payload, "episode_id", info.get("episode_id"))
    _set_trace_field_if_present(trace_payload, "group_id", info.get("group_id"))
    _set_trace_field_if_present(trace_payload, "subtask", info.get("subtask"))
    _set_trace_field_if_present(
        trace_payload,
        "x0_smiles",
        _first_nonempty_string(info.get("x0_smiles"), gt.get("x0_smiles")),
    )
    _set_trace_field_if_present(trace_payload, "instruction", session.get("base_prompt"))

    max_turns = session.get("max_turns")
    if isinstance(max_turns, int):
        trace_payload["max_turns"] = max_turns
    else:
        try:
            trace_payload["max_turns"] = int(max_turns)
        except (TypeError, ValueError):
            pass
    return json.dumps(trace_payload, ensure_ascii=False, separators=(",", ":"))


def _summarize_turn_payloads(turn_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    if not turn_payloads:
        return {
            "success": 0.0,
            "constraint_success": 0.0,
            "selected_similarity": 0.0,
            "turns_used": 0,
            "invalid_turns": 0,
        }

    horizon_turn = turn_payloads[-1]
    success_turn: dict[str, Any] | None = None
    for turn_payload in turn_payloads:
        repo_reward = _repo_reward_payload(turn_payload)
        if _to_float(repo_reward.get("success", 0.0), 0.0) > 0.0:
            success_turn = turn_payload
            break

    selected_turn = success_turn or horizon_turn
    selected_reward = _repo_reward_payload(selected_turn)
    success_value = 1.0 if success_turn is not None else 0.0
    constraint_success = 1.0 if any(
        _to_float(repo_reward.get("constraint_success", 0.0), 0.0) > 0.0
        for repo_reward in (_repo_reward_payload(turn_payload) for turn_payload in turn_payloads)
    ) else 0.0
    selected_similarity = _to_float(
        selected_reward.get("similarity", selected_reward.get("similar", 0.0)),
        0.0,
    )
    turns_used = int(horizon_turn.get("turn_id", len(turn_payloads)) or len(turn_payloads))
    invalid_turns = sum(
        1
        for turn_payload in turn_payloads
        if _to_float(_repo_reward_payload(turn_payload).get("invalid", 0.0), 0.0) > 0.0
    )
    return {
        "success": success_value,
        "constraint_success": constraint_success,
        "selected_similarity": selected_similarity,
        "turns_used": turns_used,
        "invalid_turns": invalid_turns,
    }


def _build_followup_prompt(
    base_prompt: str,
    previous_response: str,
    reward_payload: dict[str, Any],
    turn_id: int,
    max_turns: int,
) -> str:
    improve = float(reward_payload.get("improve", reward_payload.get("improvement_score", 0.0)) or 0.0)
    similar = float(reward_payload.get("similar", reward_payload.get("similarity", 0.0)) or 0.0)
    total = float(reward_payload.get("total", reward_payload.get("reward", 0.0)) or 0.0)
    return (
        f"{base_prompt}\n\n"
        f"Turn {turn_id}/{max_turns}\n"
        f"Previous response:\n{previous_response}\n\n"
        f"Previous reward summary: improve={improve:.4f}, similar={similar:.4f}, total={total:.4f}\n"
        "Revise the molecule based on the previous result. Return only one molecule in the same tag format."
    )


def _coerce_reward_context(
    record: dict[str, Any] | None,
    extra_info: dict[str, Any] | None,
) -> dict[str, Any]:
    row = record if isinstance(record, dict) else {}
    info = dict(extra_info or row.get("extra_info") or {})
    reward_model = row.get("reward_model") if isinstance(row.get("reward_model"), dict) else {}
    ground_truth = reward_model.get("ground_truth")
    if not isinstance(ground_truth, dict):
        ground_truth = {}

    if "x0_smiles" not in ground_truth and isinstance(info.get("x0_smiles"), str):
        ground_truth["x0_smiles"] = info.get("x0_smiles")

    properties = ground_truth.get("properties")
    if not isinstance(properties, list):
        properties = info.get("properties")
        if not isinstance(properties, list):
            properties = info.get("property_targets")
        if isinstance(properties, list):
            ground_truth["properties"] = properties

    if "x0_smiles" not in info and isinstance(ground_truth.get("x0_smiles"), str):
        info["x0_smiles"] = ground_truth.get("x0_smiles")
    if "properties" not in info and isinstance(ground_truth.get("properties"), list):
        info["properties"] = ground_truth.get("properties")
    return {"ground_truth": ground_truth, "extra_info": info}


class RepoTrajectoryInteraction(BaseInteraction):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._instance_dict: dict[str, dict[str, Any]] = {}

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        *,
        record: dict[str, Any] | None = None,
        extra_info: dict[str, Any] | None = None,
        max_turns: int | None = None,
        base_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        instance_id = await super().start_interaction(instance_id=instance_id, **kwargs)
        resolved_max_turns = int(max_turns or ((extra_info or {}).get("max_turns") or 4))
        reward_context = _coerce_reward_context(record, extra_info)
        self._instance_dict[instance_id] = {
            "turn_id": 0,
            "max_turns": max(1, resolved_max_turns),
            "base_prompt": base_prompt if isinstance(base_prompt, str) else _extract_base_prompt(record, extra_info),
            "reward_context": reward_context,
            "step_rewards": {},
            "reward_breakdowns": {},
            "turn_payloads": [],
        }
        return instance_id

    async def generate_response(
        self,
        instance_id: str,
        messages: list[dict[str, Any]],
        *,
        response_length: int = 0,
        force_terminate: bool = False,
        **kwargs: Any,
    ) -> tuple[bool, str, float, dict[str, Any]]:
        del kwargs
        session = self._instance_dict[instance_id]
        turn_id = int(session["turn_id"]) + 1
        max_turns = int(session["max_turns"])
        assistant_message = _extract_last_assistant_message(messages)

        reward_context = dict(session["reward_context"])
        reward_payload = _normalize_reward_payload(
            compute_score(
                data_source="repo_multi_turn_baseline",
                solution_str=assistant_message,
                ground_truth=reward_context.get("ground_truth"),
                extra_info=reward_context.get("extra_info"),
            )
        )
        step_reward = float(reward_payload.get("total", 0.0))
        success = _reward_success(reward_payload)
        hit_max_turns = turn_id >= max_turns
        should_stop = bool(success or hit_max_turns or force_terminate)

        turn_payload = {
            "turn_id": turn_id,
            "max_turns": max_turns,
            "response_text": assistant_message,
            "response_length": int(response_length),
            "repo_reward": reward_payload,
            "step_reward": step_reward,
            "reward_status": str(reward_payload.get("status", "unknown")),
            "success": bool(success),
            "stop_reason": (
                "success"
                if success
                else ("max_turns" if hit_max_turns else ("agent_loop_limit" if force_terminate else ""))
            ),
        }

        step_rewards = dict(session["step_rewards"])
        step_rewards[turn_id] = step_reward
        reward_breakdowns = dict(session["reward_breakdowns"])
        reward_breakdowns[turn_id] = {
            "status": str(reward_payload.get("status", "unknown")),
            "improve": float(reward_payload.get("improve", 0.0)),
            "similar": float(reward_payload.get("similar", 0.0)),
            "total": step_reward,
        }

        session["turn_id"] = turn_id
        session["step_rewards"] = step_rewards
        session["reward_breakdowns"] = reward_breakdowns
        session["turn_payloads"] = [*list(session["turn_payloads"]), turn_payload]

        additional_data: dict[str, Any] = {
            "turn_payload": dict(turn_payload),
            "turn_result": {
                "should_stop": should_stop,
                "success": bool(success),
                "hit_max_turns": bool(hit_max_turns),
                "forced_stop": bool(force_terminate),
            },
            "feedback_message": "",
        }

        if should_stop:
            trajectory_return = float(sum(step_rewards.values()))
            additional_data.update(
                {
                    "trajectory_reward": trajectory_return,
                    "trajectory_return": trajectory_return,
                    "step_rewards": dict(step_rewards),
                    "reward_breakdowns": dict(reward_breakdowns),
                }
            )
            summary = _summarize_turn_payloads(list(session["turn_payloads"]))
            summary["validation_trace_json"] = _build_validation_trace_json(
                trajectory_id=instance_id,
                session=session,
                turn_payloads=list(session["turn_payloads"]),
            )
            additional_data["reward_extra_info"] = summary
            return True, "", 0.0, additional_data

        next_user_content = _build_followup_prompt(
            base_prompt=str(session["base_prompt"]),
            previous_response=assistant_message,
            reward_payload=reward_payload,
            turn_id=turn_id + 1,
            max_turns=max_turns,
        )
        additional_data["feedback_message"] = next_user_content
        return False, next_user_content, 0.0, additional_data

    async def finalize_interaction(self, instance_id: str, **kwargs: Any) -> None:
        del kwargs
        self._instance_dict.pop(instance_id, None)
