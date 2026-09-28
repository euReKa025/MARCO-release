from __future__ import annotations

from .system_prompt import format_similarity_target_range, get_feedback_response_instruction, get_similarity_target_range


def _similarity_status_and_distance(similarity: float) -> tuple[str, float]:
    low, high = get_similarity_target_range()
    if similarity < low:
        return "below", float(low - similarity)
    if similarity <= high:
        return "in_range", 0.0
    return "above", float(similarity - high)


def _special_invalid_feedback(
    *,
    invalid_type: str,
    error_message: str,
    error_detail: str,
    prompt_mode: str | None,
) -> tuple[str, str, str] | None:
    generic_hint = get_feedback_response_instruction(prompt_mode, require_valid_smiles=True)
    normalized_error = f"{error_message}\n{error_detail}".lower()

    if invalid_type == "invalid_parse" and "multi_fragment_smiles" in normalized_error:
        reason = 'The candidate contains disconnected fragments separated by ".".'
        detail = "multi_fragment_smiles"
        hint = (
            "Return exactly one connected molecule. "
            "Do not output salts, solvents, or extra fragments. "
            f"{generic_hint}"
        )
        return reason, detail, hint

    return None


def format_feedback(
    *,
    predictions: dict[str, float],
    property_gaps: dict[str, float],
    directional_improvements: dict[str, float],
    progress_score: float,
    similarity: float,
    met_all_targets: bool,
    invalid_type: str,
    error_message: str = "",
    error_detail: str = "",
    recovery_hint: str = "",
    message: str = "",
    prompt_mode: str | None = None,
) -> str:
    target_range = format_similarity_target_range()
    if invalid_type != "ok":
        specialized = _special_invalid_feedback(
            invalid_type=invalid_type,
            error_message=error_message,
            error_detail=error_detail,
            prompt_mode=prompt_mode,
        )
        if specialized is not None:
            reason, detail, hint = specialized
        else:
            reason = error_message or message or "The previous answer was invalid."
            detail = error_detail or message or reason
            hint = recovery_hint or get_feedback_response_instruction(prompt_mode, require_valid_smiles=True)
        return (
            "Environment feedback:\n"
            f"- invalid_type: {invalid_type}\n"
            f"- target_similarity_range: {target_range}\n"
            f"- error_message: {reason}\n"
            f"- error_detail: {detail}\n"
            f"- recovery_hint: {hint}\n"
        )

    similarity_status, similarity_distance = _similarity_status_and_distance(similarity)
    lines = ["Environment feedback:"]
    for prop, value in sorted(predictions.items()):
        improvement = float(directional_improvements.get(prop, 0.0))
        if property_gaps:
            gap = float(property_gaps.get(prop, 0.0))
            lines.append(
                f"- {prop}: pred={value:.6f}, directional_improvement={improvement:.6f}, legacy_gap={gap:.6f}"
            )
        else:
            lines.append(f"- {prop}: pred={value:.6f}, directional_improvement={improvement:.6f}")
    lines.append(f"- progress_score: {progress_score:.6f}")
    lines.append(f"- target_similarity_range: {target_range}")
    lines.append(f"- similarity_to_source: {similarity:.6f}")
    lines.append(f"- similarity_status: {similarity_status}")
    lines.append(f"- similarity_distance_to_range: {similarity_distance:.6f}")
    lines.append(f"- met_all_targets: {met_all_targets}")
    return "\n".join(lines)
