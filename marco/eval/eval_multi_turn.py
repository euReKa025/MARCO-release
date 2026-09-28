from __future__ import annotations

from collections import defaultdict
from typing import Any

from marco.utils import get_env_float

_MODEL_INVALID_TYPES = {"invalid_format", "invalid_parse", "invalid_eval"}
_ORACLE_CALL_TYPES = {"ok", "invalid_eval", "env_error"}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _active_names(turn: dict[str, Any]) -> list[str]:
    improvements = turn.get("directional_improvements") or {}
    return list(turn.get("active_property_names") or improvements.keys())


def is_similarity_acceptable(
    similarity: float,
    *,
    similarity_target_low: float,
    similarity_copy_threshold: float,
) -> bool:
    return similarity >= similarity_target_low and similarity < similarity_copy_threshold


def select_multi_turn_reporting_turn(
    turns: list[dict[str, Any]],
    *,
    similarity_target_low: float,
    similarity_copy_threshold: float,
) -> dict[str, Any]:
    if not turns:
        return {
            "horizon_idx": -1,
            "selected_idx": -1,
            "success_idx": -1,
            "success": False,
            "constraint_success": False,
        }

    horizon_idx = len(turns) - 1
    for i, turn in enumerate(turns):
        if str(turn.get("stop_reason", "")) in {"success", "max_turns"}:
            horizon_idx = i
            break

    used_turns = turns[: horizon_idx + 1]
    success_idx = -1
    constraint_success = False
    for i, turn in enumerate(used_turns):
        if bool(turn.get("met_all_targets", False)):
            if success_idx < 0:
                success_idx = i
            similarity = _to_float(turn.get("similarity", 0.0), 0.0)
            if is_similarity_acceptable(
                similarity,
                similarity_target_low=similarity_target_low,
                similarity_copy_threshold=similarity_copy_threshold,
            ):
                constraint_success = True

    selected_idx = success_idx if success_idx >= 0 else horizon_idx
    return {
        "horizon_idx": horizon_idx,
        "selected_idx": selected_idx,
        "success_idx": success_idx,
        "success": success_idx >= 0,
        "constraint_success": constraint_success,
    }


def evaluate_multi_turn_trajectories(
    trajectories: list[dict[str, Any]],
    *,
    similarity_threshold: float,
    similarity_target_low: float | None = None,
    similarity_target_high: float | None = None,
    similarity_copy_threshold: float | None = None,
    max_success_examples: int = 5,
) -> dict[str, Any]:
    # Keep backward-compatibility for call sites that still pass target_high.
    resolved_copy_threshold = similarity_copy_threshold
    if resolved_copy_threshold is None and similarity_target_high is not None:
        resolved_copy_threshold = similarity_target_high

    similarity_target_low = float(
        similarity_target_low if similarity_target_low is not None else get_env_float("MARCO_SIMILARITY_TARGET_LOW", 0.6)
    )
    similarity_copy_threshold = float(
        resolved_copy_threshold
        if resolved_copy_threshold is not None
        else get_env_float("MARCO_SIMILARITY_COPY_THRESHOLD", 0.99)
    )
    if similarity_target_low > similarity_copy_threshold:
        similarity_target_low, similarity_copy_threshold = similarity_copy_threshold, similarity_target_low

    rows: list[dict[str, Any]] = []
    success_examples: list[dict[str, Any]] = []
    per_prop_values: dict[str, list[float]] = defaultdict(list)

    for traj in trajectories:
        turns = list(traj.get("turns") or [])
        if not turns:
            continue

        selection = select_multi_turn_reporting_turn(
            turns,
            similarity_target_low=similarity_target_low,
            similarity_copy_threshold=similarity_copy_threshold,
        )
        horizon_idx = int(selection["horizon_idx"])
        selected_idx = int(selection["selected_idx"])
        success = float(bool(selection["success"]))
        constraint_success = float(bool(selection["constraint_success"]))

        used_turns = turns[: horizon_idx + 1]
        horizon_turn = used_turns[-1]
        selected_turn = turns[selected_idx]

        turns_used = int(horizon_turn.get("turn_id", horizon_idx + 1))
        invalid_turns = sum(1 for t in used_turns if str(t.get("invalid_type", "ok")) in _MODEL_INVALID_TYPES)
        oracle_calls = sum(1 for t in used_turns if str(t.get("invalid_type", "ok")) in _ORACLE_CALL_TYPES)
        final_similarity = _to_float(selected_turn.get("similarity", 0.0), 0.0)
        valid_selected = float(
            str(selected_turn.get("invalid_type", "ok")) == "ok" and bool(selected_turn.get("candidate_smiles"))
        )

        improvements = selected_turn.get("directional_improvements") or {}
        active_names = _active_names(selected_turn)
        for name in active_names:
            per_prop_values[name].append(_to_float(improvements.get(name, 0.0), 0.0))

        row = {
            "trajectory_id": str(traj.get("trajectory_id", "")),
            "success": success,
            "constraint_success": constraint_success,
            "turns_used": float(turns_used),
            "oracle_calls": float(oracle_calls),
            "final_similarity": final_similarity,
            "valid_selected_candidate": valid_selected,
            "meets_similarity_threshold": float(final_similarity >= similarity_threshold),
            "invalid_turns": float(invalid_turns),
            "per_property_improvement": {name: float(improvements.get(name, 0.0)) for name in active_names},
        }
        rows.append(row)

        if success > 0.0 and len(success_examples) < max_success_examples:
            success_examples.append(
                {
                    "trajectory_id": row["trajectory_id"],
                    "turns_used": int(turns_used),
                    "final_similarity": final_similarity,
                    "final_candidate_smiles": selected_turn.get("candidate_smiles"),
                    "x0_smiles": traj.get("x0_smiles"),
                    "subtask": traj.get("subtask"),
                    "per_property_improvement": row["per_property_improvement"],
                }
            )

    if not rows:
        return {
            "rows": [],
            "metrics": {
                "num_trajectories": 0.0,
                "success_rate": 0.0,
                "constraint_success_rate": 0.0,
                "avg_turns_used": 0.0,
                "avg_oracle_calls": 0.0,
                "valid_selected_similarity_mean": 0.0,
                "valid_selected_similarity_count": 0.0,
                "valid_selected_similarity_rate": 0.0,
                "invalid_turns_per_trajectory": 0.0,
                "meets_similarity_threshold_rate": 0.0,
            },
            "success_examples": [],
            "similarity_target_low": similarity_target_low,
            "similarity_copy_threshold": similarity_copy_threshold,
        }

    n = float(len(rows))
    valid_selected_rows = [r for r in rows if float(r.get("valid_selected_candidate", 0.0)) > 0.0]
    valid_selected_count = float(len(valid_selected_rows))
    metrics = {
        "num_trajectories": n,
        "success_rate": sum(float(r["success"]) for r in rows) / n,
        "constraint_success_rate": sum(float(r["constraint_success"]) for r in rows) / n,
        "avg_turns_used": sum(float(r["turns_used"]) for r in rows) / n,
        "avg_oracle_calls": sum(float(r["oracle_calls"]) for r in rows) / n,
        "valid_selected_similarity_mean": (
            sum(float(r["final_similarity"]) for r in valid_selected_rows) / valid_selected_count
            if valid_selected_count > 0.0
            else 0.0
        ),
        "valid_selected_similarity_count": valid_selected_count,
        "valid_selected_similarity_rate": valid_selected_count / n,
        "invalid_turns_per_trajectory": sum(float(r["invalid_turns"]) for r in rows) / n,
        "meets_similarity_threshold_rate": sum(float(r["meets_similarity_threshold"]) for r in rows) / n,
    }

    for prop, values in sorted(per_prop_values.items()):
        if values:
            metrics[f"final_directional_improvement/{prop}"] = sum(values) / float(len(values))

    return {
        "rows": rows,
        "metrics": metrics,
        "success_examples": success_examples,
        "similarity_target_low": similarity_target_low,
        "similarity_copy_threshold": similarity_copy_threshold,
    }
