from __future__ import annotations

import re
from typing import Any

from marco.core_types import EpisodeState, PropertyTarget
from marco.env.molecule_env import MoleculeEnv
from marco.env.molecule_validation import validate_model_output
from marco.env.predictor_client import PredictorClient
from marco.prompts.feedback_formatter import format_feedback
from marco.prompts.prompt_builder import build_turn_prompt
from marco.reward.config import build_reward_config_from_env
from marco.trainer_types import Sample

_SHARED_ENV: MoleculeEnv | None = None
_CLOSED_ANSWER_RE = re.compile(r"<(?P<tag>answer|SMILES)>.*?</(?P=tag)>", re.IGNORECASE | re.DOTALL)


def _get_env() -> MoleculeEnv:
    global _SHARED_ENV
    if _SHARED_ENV is None:
        cfg = build_reward_config_from_env()
        client = PredictorClient()
        _SHARED_ENV = MoleculeEnv(client, cfg)
    return _SHARED_ENV


def _state_from_payload(payload: dict[str, Any]) -> EpisodeState:
    property_targets = [PropertyTarget(**item) for item in payload["property_targets"]]
    return EpisodeState(
        episode_id=payload["episode_id"],
        subtask=payload["subtask"],
        x0_smiles=payload["x0_smiles"],
        current_smiles=payload["current_smiles"],
        property_targets=property_targets,
        x0_predictions={k: float(v) for k, v in payload["x0_predictions"].items()},
        current_predictions={k: float(v) for k, v in payload["current_predictions"].items()},
        current_total_gap=float(payload.get("current_total_gap", 0.0)),
        current_directional_improvements={
            k: float(v) for k, v in (payload.get("current_directional_improvements") or {}).items()
        },
        current_progress_score=float(payload.get("current_progress_score", 0.0)),
        success_turn=(int(payload["success_turn"]) if payload.get("success_turn") is not None else None),
        turn_id=int(payload.get("turn_id", 0)),
        max_turns=int(payload.get("max_turns", 4)),
        history=list(payload.get("history", [])),
    )


def _step_result_to_dict(result) -> dict[str, Any]:
    return {
        "status": result.status,
        "turn_id": result.turn_id,
        "candidate_smiles": result.candidate_smiles,
        "predictions": result.predictions,
        "property_gaps": result.property_gaps,
        "total_gap": result.total_gap,
        "similarity": result.similarity,
        "met_all_targets": result.met_all_targets,
        "reward_total": result.reward_total,
        "reward_gap": result.reward_gap,
        "reward_similarity": result.reward_similarity,
        "reward_success": result.reward_success,
        "should_stop": result.should_stop,
        "message": result.message,
        "error_message": result.error_message,
        "error_detail": result.error_detail,
        "recovery_hint": result.recovery_hint,
        "directional_improvements": result.directional_improvements,
        "progress_components": result.progress_components,
        "progress_score": result.progress_score,
        "active_property_names": result.active_property_names,
        "success_this_turn": result.success_this_turn,
    }


def _feedback_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "predictions": result.get("predictions", {}),
        "property_gaps": result.get("property_gaps", {}),
        "directional_improvements": result.get("directional_improvements", {}),
        "progress_score": float(result.get("progress_score", 0.0)),
        "similarity": float(result.get("similarity", 0.0)),
        "met_all_targets": bool(result.get("met_all_targets", False)),
        "invalid_type": str(result.get("status", "ok")),
        "error_message": str(result.get("error_message", "")),
        "error_detail": str(result.get("error_detail", "")),
        "recovery_hint": str(result.get("recovery_hint", "")),
        "message": str(result.get("message", "")),
    }


def _feedback_message(result: dict[str, Any]) -> str:
    return format_feedback(**_feedback_payload(result))


def _prompt_targets(targets: list[PropertyTarget]) -> list[dict[str, Any]]:
    return [
        {
            "name": target.name,
            "direction": target.direction,
            "delta": target.delta,
        }
        for target in targets
    ]


def _append_conversation_messages(
    state_payload: dict[str, Any],
    state: EpisodeState,
    *,
    response_text: str,
    result_dict: dict[str, Any],
    turn_id: int,
) -> list[dict[str, str]]:
    messages = [dict(message) for message in list(state_payload.get("messages") or []) if isinstance(message, dict)]
    if response_text:
        messages.append({"role": "assistant", "content": response_text})

    if not bool(result_dict.get("should_stop", False)) and turn_id < state.max_turns:
        next_user_content = build_turn_prompt(
            x0_smiles=state.x0_smiles,
            current_smiles=state.current_smiles,
            property_targets=_prompt_targets(state.property_targets),
            turn_id=turn_id + 1,
            history=[],
            previous_feedback=_feedback_payload(result_dict),
        )
        messages.append({"role": "user", "content": next_user_content})

    return messages


def _has_closed_answer_tag(text: str | None) -> bool:
    if not text:
        return False
    return _CLOSED_ANSWER_RE.search(text) is not None


def _empty_state(turn_id: int) -> EpisodeState:
    return EpisodeState(
        episode_id="missing",
        subtask="unknown",
        x0_smiles="",
        current_smiles="",
        property_targets=[],
        x0_predictions={},
        current_predictions={},
        current_total_gap=0.0,
        current_directional_improvements={},
        current_progress_score=0.0,
        turn_id=0,
        max_turns=turn_id,
        history=[],
    )


def score_single_sample(sample: Sample) -> float:
    env = _get_env()
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}

    state_payload = metadata.get("rl_state")
    turn_id = int(metadata.get("turn_id", 1))
    response_text = sample.response if isinstance(sample.response, str) else ""

    # Strict format rule: when output is truncated, it must still contain a closed tagged molecule block.
    if sample.status == Sample.Status.TRUNCATED and not _has_closed_answer_tag(response_text):
        state = _state_from_payload(state_payload) if isinstance(state_payload, dict) else _empty_state(turn_id=turn_id)
        result = env.step_invalid(
            state=state,
            turn_id=turn_id,
            invalid_type="invalid_format",
            message="truncated_without_closed_answer_tag",
        )
        result_dict = _step_result_to_dict(result)
        metadata["turn_result"] = result_dict
        metadata["feedback_message"] = _feedback_message(result_dict)
        if isinstance(state_payload, dict):
            env.apply_step(state, result)
            next_state = env.state_to_payload(state)
            next_state["messages"] = _append_conversation_messages(
                state_payload=state_payload,
                state=state,
                response_text=response_text,
                result_dict=result_dict,
                turn_id=turn_id,
            )
            metadata["next_state"] = next_state
        sample.metadata = metadata
        return 0.0

    if not isinstance(state_payload, dict):
        # Missing state is treated as invalid-eval for safety, but reward is still
        # resolved post-hoc at trajectory level.
        result = env.step_invalid(
            state=_empty_state(turn_id=turn_id),
            turn_id=turn_id,
            invalid_type="invalid_eval",
            message="missing_rl_state",
        )
        result_dict = _step_result_to_dict(result)
        metadata["turn_result"] = result_dict
        metadata["feedback_message"] = _feedback_message(result_dict)
        sample.metadata = metadata
        return 0.0

    state = _state_from_payload(state_payload)

    validation = validate_model_output(response_text)
    metadata["parsed_smiles"] = validation.smiles
    if validation.status != "ok":
        result = env.step_invalid(
            state=state,
            turn_id=turn_id,
            invalid_type=validation.status,
            message=validation.message,
        )
    else:
        result = env.step(state=state, candidate_smiles=str(validation.smiles), turn_id=turn_id)

    env.apply_step(state, result)

    result_dict = _step_result_to_dict(result)
    metadata["turn_result"] = result_dict
    metadata["feedback_message"] = _feedback_message(result_dict)
    next_state = env.state_to_payload(state)
    next_state["messages"] = _append_conversation_messages(
        state_payload=state_payload,
        state=state,
        response_text=response_text,
        result_dict=result_dict,
        turn_id=turn_id,
    )
    metadata["next_state"] = next_state
    metadata["invalid_type"] = result.status
    metadata["reward_breakdown"] = {
        "reward_total": 0.0,
        "reward_gap": 0.0,
        "reward_similarity": 0.0,
        "reward_success": 0.0,
        "reward_invalid": 0.0,
        "posthoc": True,
    }
    sample.metadata = metadata

    return 0.0


async def batched_custom_rm(args, samples):
    if isinstance(samples, list):
        rewards = []
        for sample in samples:
            rewards.append(score_single_sample(sample))
        return rewards
    # Compatibility path: some callers may pass a single sample instead of a list.
    return score_single_sample(samples)


async def custom_rm(args, sample):
    return score_single_sample(sample)
