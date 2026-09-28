from __future__ import annotations

from typing import Any

from marco.rollout.sample_conversion import _settle_posthoc_rewards
from marco.trainer_types import Sample


def settle_trajectory_rewards_from_samples(
    samples: list[Any],
) -> tuple[dict[int, float], dict[int, dict[str, float]], dict[str, float]]:
    step_rewards, breakdowns, trajectory_returns, _, _ = _settle_posthoc_rewards(samples)
    return step_rewards, breakdowns, trajectory_returns


def settle_trajectory_rewards_from_turn_payloads(
    turn_payloads: list[dict[str, Any]],
    *,
    trajectory_id: str,
    group_id: str | None = None,
) -> dict[str, Any]:
    samples: list[Sample] = []
    for payload in turn_payloads:
        sample = Sample(
            response=str(payload.get("response_text", "")),
            response_length=int(payload.get("response_length", 0) or 0),
            tokens=[],
            metadata={},
        )
        sample.train_metadata = {
            "trajectory_id": trajectory_id,
            "group_id": group_id or trajectory_id,
            "turn_id": int(payload.get("turn_id", len(samples) + 1)),
            "max_turns": int(payload.get("max_turns", len(turn_payloads) or 1)),
            "invalid_type": str(payload.get("invalid_type", "ok")),
            "similarity": float(payload.get("similarity", 0.0)),
            "progress_score": float(payload.get("progress_score", 0.0)),
            "met_all_targets": bool(payload.get("met_all_targets", False)),
            "error_detail": str(payload.get("error_detail", "")),
        }
        sample.status = Sample.Status.COMPLETED
        samples.append(sample)

    step_rewards, breakdowns, trajectory_returns = settle_trajectory_rewards_from_samples(samples)
    trajectory_reward = float(trajectory_returns.get(trajectory_id, 0.0))
    return {
        "samples": samples,
        "step_rewards": step_rewards,
        "breakdowns": breakdowns,
        "trajectory_returns": trajectory_returns,
        "trajectory_reward": trajectory_reward,
    }
