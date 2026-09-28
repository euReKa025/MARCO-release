from __future__ import annotations

from typing import Any


def _count_improved(turn: dict[str, Any]) -> tuple[int, int]:
    improvements = turn.get("directional_improvements") or {}
    active_names = list(turn.get("active_property_names") or improvements.keys())
    if not active_names:
        return 0, 0
    improved = 0
    for name in active_names:
        if float(improvements.get(name, 0.0)) > 0.0:
            improved += 1
    return improved, len(active_names)


def evaluate_single_turn_trajectories(
    trajectories: list[dict[str, Any]],
    *,
    similarity_threshold: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    for traj in trajectories:
        turns = list(traj.get("turns") or [])
        if not turns:
            continue
        first = turns[0]
        invalid_type = str(first.get("invalid_type", "ok"))
        similarity = float(first.get("similarity", 0.0))
        property_success = bool(first.get("met_all_targets", False))
        sim_ok = similarity >= similarity_threshold
        success = float(property_success)
        constraint_success = float(invalid_type == "ok" and property_success and sim_ok)
        improved_count, total_count = _count_improved(first)

        rows.append(
            {
                "trajectory_id": str(traj.get("trajectory_id", "")),
                "success": success,
                "constraint_success": constraint_success,
                "invalid_type": invalid_type,
                "similarity": similarity,
                "improved_count": improved_count,
                "active_property_count": total_count,
            }
        )

    if not rows:
        return {
            "rows": [],
            "metrics": {
                "num_trajectories": 0.0,
                "success_rate": 0.0,
                "constraint_success_rate": 0.0,
                "invalid_rate": 0.0,
                "mean_similarity": 0.0,
            },
        }

    n = float(len(rows))
    invalid = sum(1.0 for r in rows if r["invalid_type"] != "ok")
    success = sum(float(r["success"]) for r in rows)
    constraint_success = sum(float(r["constraint_success"]) for r in rows)
    mean_similarity = sum(float(r["similarity"]) for r in rows) / n

    return {
        "rows": rows,
        "metrics": {
            "num_trajectories": n,
            "success_rate": success / n,
            "constraint_success_rate": constraint_success / n,
            "invalid_rate": invalid / n,
            "mean_similarity": mean_similarity,
        },
    }
