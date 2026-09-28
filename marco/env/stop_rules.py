from __future__ import annotations

from marco.core_types import StopResult


def should_stop_episode(
    *,
    turn_id: int,
    max_turns: int,
    invalid_type: str,
    met_all_targets: bool,
    similarity_acceptable: bool,
) -> StopResult:
    # In V1 we do not terminate on model-invalid actions. They receive penalty and
    # continue rollout until success or max turns.
    if invalid_type in {"invalid_format", "invalid_parse", "invalid_eval"}:
        pass
    if met_all_targets and similarity_acceptable:
        return StopResult(should_stop=True, reason="success")
    if turn_id >= max_turns:
        return StopResult(should_stop=True, reason="max_turns")
    return StopResult(should_stop=False, reason="continue")
