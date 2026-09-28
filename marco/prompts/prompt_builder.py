from __future__ import annotations

from typing import Any

from .feedback_formatter import format_feedback
from .system_prompt import (
    PROMPT_MODE_THINK_ANSWER,
    build_system_user_messages,
    format_similarity_target_range,
    get_user_response_instruction,
)


def _format_targets(property_targets: list[dict[str, Any]]) -> str:
    lines = []
    for item in property_targets:
        name = item["name"]
        direction = item["direction"]
        lines.append(f"- {name}: {direction}")
    return "\n".join(lines)


def build_turn_prompt(
    *,
    x0_smiles: str,
    current_smiles: str,
    property_targets: list[dict[str, Any]],
    turn_id: int,
    history: list[dict[str, Any]],
    previous_feedback: dict[str, Any] | None,
    prompt_mode: str | None = None,
) -> str:
    history_block = ""
    if history:
        lines = ["Previous turns summary:"]
        for item in history[-3:]:
            cand = item.get("candidate_smiles") or "<invalid>"
            invalid_type = item.get("invalid_type", "ok")
            progress_score = float(item.get("progress_score", 0.0))
            similarity = float(item.get("similarity", 0.0))
            lines.append(
                f"- turn {item.get('turn_id')}: candidate={cand}, invalid={invalid_type}, "
                f"progress_score={progress_score:.6f}, similarity={similarity:.6f}"
            )
        history_block = "\n".join(lines)

    feedback_block = ""
    if previous_feedback is not None:
        feedback_block = format_feedback(
            predictions=previous_feedback.get("predictions", {}),
            property_gaps=previous_feedback.get("property_gaps", {}),
            directional_improvements=previous_feedback.get("directional_improvements", {}),
            progress_score=float(previous_feedback.get("progress_score", 0.0)),
            similarity=float(previous_feedback.get("similarity", 0.0)),
            met_all_targets=bool(previous_feedback.get("met_all_targets", False)),
            invalid_type=str(previous_feedback.get("invalid_type", "ok")),
            error_message=str(previous_feedback.get("error_message", "")),
            error_detail=str(previous_feedback.get("error_detail", "")),
            recovery_hint=str(previous_feedback.get("recovery_hint", "")),
            message=str(previous_feedback.get("message", "")),
            prompt_mode=prompt_mode,
        )

    similarity_block = (
        "Similarity target:\n"
        f"- Keep similarity to the source molecule in the target range {format_similarity_target_range()}."
    )

    return (
        f"Turn: {turn_id}\n"
        f"Source molecule (x0): {x0_smiles}\n"
        f"Current molecule (x_t): {current_smiles}\n"
        "Target property directions:\n"
        f"{_format_targets(property_targets)}\n\n"
        f"{similarity_block}\n\n"
        f"{history_block}\n\n"
        f"{feedback_block}\n"
        f"{get_user_response_instruction(prompt_mode)}"
    ).strip()


def build_turn_messages(
    *,
    x0_smiles: str,
    current_smiles: str,
    property_targets: list[dict[str, Any]],
    turn_id: int,
    history: list[dict[str, Any]],
    previous_feedback: dict[str, Any] | None,
    prompt_mode: str = PROMPT_MODE_THINK_ANSWER,
) -> list[dict[str, str]]:
    user_content = build_turn_prompt(
        x0_smiles=x0_smiles,
        current_smiles=current_smiles,
        property_targets=property_targets,
        turn_id=turn_id,
        history=history,
        previous_feedback=previous_feedback,
        prompt_mode=prompt_mode,
    )
    return build_system_user_messages(user_content, prompt_mode=prompt_mode)
