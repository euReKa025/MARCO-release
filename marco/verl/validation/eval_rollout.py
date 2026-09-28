from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from marco.env.molecule_validation import is_valid_candidate_smiles
from marco.env.similarity import tanimoto_similarity
from marco.prompts.answer_parser import parse_answer_smiles
from marco.utils import get_env_float


_TRACE_KEYS = (
    "validation_trace_json",
    "trajectory_trace_json",
    "validation_trace",
    "trajectory_trace",
    "trace_json",
)
_METRIC_KEYS = ("metrics", "metric_blob", "validation_metrics")
_MAX_SUCCESS_EXAMPLES_PER_DATASET = 5
_GLOBAL_METRIC_NAMESPACES = {"val", "validation", "eval", "overall", "summary"}


def _parse_json_payload(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _try_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_probability(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "y"}:
            return 1.0
        if lowered in {"false", "no", "n"}:
            return 0.0
    return _to_float(value, default)


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _prompt_like_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        for item in reversed(value):
            if isinstance(item, dict) and item.get("role") == "user":
                content = item.get("content")
                if isinstance(content, str):
                    return content
    return ""


def _extract_reward_extra_info(row: dict[str, Any]) -> dict[str, Any]:
    payload = _parse_json_payload(row.get("reward_extra_info"))
    return payload if isinstance(payload, dict) else {}


def _extract_gts_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = _parse_json_payload(row.get("gts"))
    return payload if isinstance(payload, dict) else {}


def _extract_metrics_blob(row: dict[str, Any]) -> dict[str, Any]:
    for key in _METRIC_KEYS:
        payload = _parse_json_payload(row.get(key))
        if isinstance(payload, dict):
            return payload
    reward_extra_info = _extract_reward_extra_info(row)
    for key in _METRIC_KEYS:
        payload = _parse_json_payload(reward_extra_info.get(key))
        if isinstance(payload, dict):
            return payload
    return {}


def _extract_trace_payload(row: dict[str, Any]) -> dict[str, Any] | None:
    for key in _TRACE_KEYS:
        payload = _parse_json_payload(row.get(key))
        if isinstance(payload, dict):
            return payload

    reward_extra_info = _extract_reward_extra_info(row)
    for key in _TRACE_KEYS:
        payload = _parse_json_payload(reward_extra_info.get(key))
        if isinstance(payload, dict):
            return payload

    if isinstance(row.get("turns"), list):
        return row
    if isinstance(reward_extra_info.get("turns"), list):
        return reward_extra_info
    return None


def _set_if_present(target: dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str) and not value.strip():
        return
    target[key] = value


def _normalize_turn(turn: dict[str, Any], *, row: dict[str, Any], turn_index: int) -> dict[str, Any]:
    normalized: dict[str, Any] = dict(turn)
    normalized["turn_id"] = _to_int(_first_present(turn.get("turn_id"), turn_index + 1), turn_index + 1)

    prompt_value = _first_present(
        turn.get("prompt"),
        turn.get("input"),
        row.get("input"),
        row.get("instruction"),
        _prompt_like_to_text(row.get("prompt")),
    )
    _set_if_present(normalized, "prompt", prompt_value)

    output_value = _first_present(turn.get("output_response"), turn.get("output"), row.get("output"), row.get("response"))
    _set_if_present(normalized, "output_response", output_value)

    candidate_smiles = _first_present(turn.get("candidate_smiles"), turn.get("parsed_smiles"), row.get("candidate_smiles"))
    if candidate_smiles is None and isinstance(output_value, str):
        parsed = parse_answer_smiles(output_value)
        candidate_smiles = parsed.smiles
    _set_if_present(normalized, "candidate_smiles", candidate_smiles)
    predictions = _parse_json_payload(_first_present(turn.get("predictions"), row.get("predicted_values")))
    if isinstance(predictions, dict) and predictions:
        normalized["predictions"] = dict(predictions)
    directional_improvements = _parse_json_payload(
        _first_present(
            turn.get("directional_improvements"),
            row.get("directional_improvements"),
        )
    )
    if isinstance(directional_improvements, dict) and directional_improvements:
        normalized["directional_improvements"] = dict(directional_improvements)
    active_property_names = turn.get("active_property_names")
    if isinstance(active_property_names, list) and active_property_names:
        normalized["active_property_names"] = list(active_property_names)
    progress_score = turn.get("progress_score")
    if progress_score is not None:
        normalized["progress_score"] = _to_float(progress_score, 0.0)

    similarity = _first_present(turn.get("similarity"), turn.get("selected_similarity"), row.get("similarity"))
    if similarity is not None:
        normalized["similarity"] = _to_float(similarity, 0.0)

    success = _first_present(turn.get("success"), turn.get("met_all_targets"), row.get("success"))
    if success is not None:
        normalized["success"] = _to_probability(success, 0.0)

    constraint_success = _first_present(
        turn.get("constraint_success"),
        row.get("constraint_success"),
    )
    if constraint_success is not None:
        normalized["constraint_success"] = _to_probability(constraint_success, 0.0)

    stop_reason = _first_present(
        turn.get("stop_reason"),
        (turn.get("turn_result") or {}).get("message") if isinstance(turn.get("turn_result"), dict) else None,
        row.get("stop_reason"),
    )
    _set_if_present(normalized, "stop_reason", stop_reason)
    return normalized


def _build_single_turn_trajectory(
    row: dict[str, Any],
    *,
    row_index: int,
    rollout_id: str,
    default_subtask: str | None = None,
) -> dict[str, Any]:
    gts_payload = _extract_gts_payload(row)
    trajectory: dict[str, Any] = {
        "trajectory_id": str(
            _first_present(
                row.get("trajectory_id"),
                row.get("sample_id"),
                row.get("episode_id"),
                f"{rollout_id}-traj-{row_index}",
            )
        ),
    }
    _set_if_present(trajectory, "episode_id", row.get("episode_id"))
    _set_if_present(trajectory, "group_id", row.get("group_id"))
    _set_if_present(trajectory, "subtask", _first_present(row.get("subtask"), gts_payload.get("subtask"), default_subtask))
    _set_if_present(trajectory, "sample_id", row.get("sample_id"))
    _set_if_present(trajectory, "x0_smiles", _first_present(row.get("x0_smiles"), gts_payload.get("x0_smiles")))
    _set_if_present(
        trajectory,
        "reference_smiles",
        _first_present(row.get("reference_smiles"), gts_payload.get("reference_smiles")),
    )
    properties = _first_present(row.get("properties"), gts_payload.get("properties"))
    if isinstance(properties, list):
        trajectory["properties"] = properties
    _set_if_present(
        trajectory,
        "instruction",
        _first_present(row.get("instruction"), row.get("input"), _prompt_like_to_text(row.get("prompt"))),
    )
    _set_if_present(trajectory, "max_turns", row.get("max_turns"))

    trajectory["turns"] = [_normalize_turn({}, row=row, turn_index=0)]
    return trajectory


def _build_trace_trajectory(
    trace_payload: dict[str, Any],
    *,
    row: dict[str, Any],
    row_index: int,
    rollout_id: str,
    default_subtask: str | None = None,
) -> dict[str, Any]:
    gts_payload = _extract_gts_payload(row)
    turns_raw = trace_payload.get("turns")
    if not isinstance(turns_raw, list):
        return _build_single_turn_trajectory(
            row,
            row_index=row_index,
            rollout_id=rollout_id,
            default_subtask=default_subtask,
        )

    turns = []
    for idx, turn in enumerate(turns_raw):
        if isinstance(turn, dict):
            turns.append(_normalize_turn(turn, row=row, turn_index=idx))

    if not turns:
        return _build_single_turn_trajectory(
            row,
            row_index=row_index,
            rollout_id=rollout_id,
            default_subtask=default_subtask,
        )

    trajectory: dict[str, Any] = {
        "trajectory_id": str(
            _first_present(
                trace_payload.get("trajectory_id"),
                trace_payload.get("sample_id"),
                row.get("trajectory_id"),
                row.get("sample_id"),
                row.get("episode_id"),
                f"{rollout_id}-traj-{row_index}",
            )
        ),
        "turns": turns,
    }
    _set_if_present(trajectory, "episode_id", _first_present(trace_payload.get("episode_id"), row.get("episode_id")))
    _set_if_present(trajectory, "group_id", _first_present(trace_payload.get("group_id"), row.get("group_id")))
    _set_if_present(trajectory, "sample_id", _first_present(trace_payload.get("sample_id"), row.get("sample_id")))
    _set_if_present(
        trajectory,
        "subtask",
        _first_present(trace_payload.get("subtask"), row.get("subtask"), gts_payload.get("subtask"), default_subtask),
    )
    _set_if_present(
        trajectory,
        "x0_smiles",
        _first_present(trace_payload.get("x0_smiles"), row.get("x0_smiles"), gts_payload.get("x0_smiles")),
    )
    _set_if_present(
        trajectory,
        "reference_smiles",
        _first_present(
            trace_payload.get("reference_smiles"),
            row.get("reference_smiles"),
            gts_payload.get("reference_smiles"),
        ),
    )
    properties = _first_present(trace_payload.get("properties"), row.get("properties"), gts_payload.get("properties"))
    if isinstance(properties, list):
        trajectory["properties"] = properties
    _set_if_present(
        trajectory,
        "instruction",
        _first_present(
            trace_payload.get("instruction"),
            row.get("instruction"),
            row.get("input"),
            _prompt_like_to_text(row.get("prompt")),
        ),
    )
    _set_if_present(trajectory, "max_turns", _first_present(trace_payload.get("max_turns"), row.get("max_turns")))
    return trajectory


def _dataset_name(row: dict[str, Any]) -> str:
    value = _first_present(
        row.get("data_source"),
        row.get("dataset"),
        row.get("source"),
        row.get("split"),
    )
    if value is None:
        return "all"
    return str(value)


def _select_reporting_turn(turns: list[dict[str, Any]]) -> dict[str, Any]:
    for turn in turns:
        if _to_probability(turn.get("success"), 0.0) > 0.0:
            return turn
    return turns[-1] if turns else {}


def _eval_similarity_is_acceptable(similarity: float | None) -> bool:
    if similarity is None:
        return False
    low = get_env_float("MARCO_SIMILARITY_TARGET_LOW", 0.6)
    high = get_env_float(
        "MARCO_SIMILARITY_TARGET_HIGH",
        get_env_float("MARCO_SIMILARITY_COPY_THRESHOLD", 0.99),
    )
    if high < low:
        low, high = high, low
    return float(similarity) >= low and float(similarity) < high


def _compute_eval_similarity(x0_smiles: Any, candidate_smiles: Any) -> float | None:
    if not isinstance(x0_smiles, str) or not x0_smiles.strip():
        return None
    if not isinstance(candidate_smiles, str) or not candidate_smiles.strip():
        return None
    if not is_valid_candidate_smiles(candidate_smiles):
        return None
    try:
        return float(tanimoto_similarity(candidate_smiles.strip(), x0_smiles.strip()))
    except Exception:  # noqa: BLE001
        return None


def _trajectory_rollup(trajectory: dict[str, Any], row: dict[str, Any], *, trace_based: bool) -> dict[str, Any]:
    turns = [turn for turn in (trajectory.get("turns") or []) if isinstance(turn, dict)]
    selected_turn = _select_reporting_turn(turns)
    horizon_turn = turns[-1] if turns else {}

    success = _to_probability(row.get("success"), -1.0)
    if success < 0.0:
        success = 1.0 if any(_to_probability(turn.get("success"), 0.0) > 0.0 for turn in turns) else 0.0

    turns_used = _to_int(_first_present(horizon_turn.get("turn_id"), len(turns)), len(turns))
    final_candidate = _first_present(selected_turn.get("candidate_smiles"), row.get("candidate_smiles"))
    final_similarity = _compute_eval_similarity(trajectory.get("x0_smiles"), final_candidate)
    final_predicted_values = selected_turn.get("predictions") if isinstance(selected_turn.get("predictions"), dict) else None
    if trace_based:
        selected_success = _to_probability(
            _first_present(selected_turn.get("success"), selected_turn.get("met_all_targets")),
            0.0,
        )
        constraint_success = 1.0 if (selected_success > 0.0 and _eval_similarity_is_acceptable(final_similarity)) else 0.0
    else:
        constraint_success = _to_probability(row.get("constraint_success"), -1.0)
        if constraint_success < 0.0:
            constraint_success = (
                1.0 if any(_to_probability(turn.get("constraint_success"), 0.0) > 0.0 for turn in turns) else 0.0
            )

    return {
        "success": success,
        "constraint_success": constraint_success,
        "turns_used": turns_used,
        "final_similarity": final_similarity,
        "final_candidate_smiles": final_candidate,
        "final_predicted_values": dict(final_predicted_values) if final_predicted_values else None,
    }


def _summary_from_stats(
    *,
    step: int,
    num_rows: int,
    num_trajectories: int,
    success_count: float,
    constraint_success_count: float,
    valid_similarity_sum: float,
    valid_similarity_count: int,
) -> dict[str, Any]:
    denominator = float(num_trajectories) if num_trajectories > 0 else 0.0
    success_rate = (success_count / denominator) if denominator else 0.0
    constraint_success_rate = (constraint_success_count / denominator) if denominator else 0.0
    valid_similarity_mean = (
        valid_similarity_sum / float(valid_similarity_count) if valid_similarity_count > 0 else 0.0
    )
    return {
        "step": int(step),
        "num_rows": int(num_rows),
        "num_trajectories": int(num_trajectories),
        "valid_selected_similarity_count": int(valid_similarity_count),
        "valid_selected_similarity_rate": (
            float(valid_similarity_count) / float(num_trajectories) if num_trajectories > 0 else 0.0
        ),
        "success_rate": success_rate,
        "constraint_success_rate": constraint_success_rate,
        "valid_selected_similarity_mean": valid_similarity_mean,
        "success_similarity_product": success_rate * valid_similarity_mean,
    }


def _parse_metric_key(key: Any) -> tuple[str, str] | None:
    if not isinstance(key, str):
        return None
    if key in {"success_rate", "constraint_success_rate"}:
        return "all", key

    parts = [part for part in key.split("/") if part]
    if not parts:
        return None

    def _is_global_namespace(prefix_parts: list[str]) -> bool:
        if not prefix_parts:
            return True
        if len(prefix_parts) == 1 and prefix_parts[0].lower() in _GLOBAL_METRIC_NAMESPACES:
            return True
        return False

    if parts[-1] in {"success_rate", "constraint_success_rate"}:
        metric_name = parts[-1]
        prefix_parts = parts[:-1]
        if _is_global_namespace(prefix_parts):
            return "all", metric_name
        if prefix_parts:
            return prefix_parts[-1], metric_name
        return "all", metric_name

    for index, part in enumerate(parts):
        if part not in {"success", "constraint_success"}:
            continue
        metric_name = f"{part}_rate"
        prefix_parts = parts[:index]
        if _is_global_namespace(prefix_parts):
            return "all", metric_name
        if prefix_parts:
            return prefix_parts[-1], metric_name
        return "all", metric_name
    return None


def _extract_metric_dataset_rates(metrics: dict[str, Any]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key, value in metrics.items():
        parsed = _parse_metric_key(key)
        if parsed is None:
            continue
        numeric_value = _try_float(value)
        if numeric_value is None:
            continue
        dataset, metric_name = parsed
        dataset_metrics = out.setdefault(dataset, {})
        dataset_metrics[metric_name] = numeric_value
    return out


def _overall_metric_rate(metric_dataset_rates: dict[str, dict[str, float]], metric_name: str) -> float:
    return _to_float((metric_dataset_rates.get("all") or {}).get(metric_name), 0.0)


def _has_usable_metric_rates(metric_dataset_rates: dict[str, dict[str, float]]) -> bool:
    has_success = any("success_rate" in rates for rates in metric_dataset_rates.values())
    has_constraint = any("constraint_success_rate" in rates for rates in metric_dataset_rates.values())
    return has_success and has_constraint


def _metric_derived_per_dataset(
    *,
    metrics: dict[str, Any],
    dataset_counts: dict[str, int],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, float]], bool]:
    metric_dataset_rates = _extract_metric_dataset_rates(metrics)
    if not _has_usable_metric_rates(metric_dataset_rates):
        return {}, metric_dataset_rates, False

    overall_success_rate = (metric_dataset_rates.get("all") or {}).get("success_rate")
    overall_constraint_success_rate = (metric_dataset_rates.get("all") or {}).get("constraint_success_rate")
    has_specific_datasets = any(dataset != "all" for dataset in metric_dataset_rates)

    per_dataset: dict[str, dict[str, Any]] = {}
    if has_specific_datasets:
        dataset_names = list(dataset_counts.keys())
        for dataset in metric_dataset_rates:
            if dataset != "all" and dataset not in dataset_names:
                dataset_names.append(dataset)
        for dataset in dataset_names:
            rates = metric_dataset_rates.get(dataset, {})
            success_rate = rates.get("success_rate", overall_success_rate)
            constraint_success_rate = rates.get("constraint_success_rate", overall_constraint_success_rate)
            if success_rate is None or constraint_success_rate is None:
                return {}, metric_dataset_rates, False
            per_dataset[dataset] = {
                "num_trajectories": int(dataset_counts.get(dataset, 0)),
                "success_rate": _to_float(success_rate, 0.0),
                "constraint_success_rate": _to_float(constraint_success_rate, 0.0),
            }
        return per_dataset, metric_dataset_rates, True

    if overall_success_rate is None or overall_constraint_success_rate is None:
        return {}, metric_dataset_rates, False
    if dataset_counts:
        for dataset, count in dataset_counts.items():
            per_dataset[dataset] = {
                "num_trajectories": int(count),
                "success_rate": _to_float(overall_success_rate, 0.0),
                "constraint_success_rate": _to_float(overall_constraint_success_rate, 0.0),
            }
    elif metric_dataset_rates:
        per_dataset["all"] = {
            "num_trajectories": 0,
            "success_rate": _to_float(overall_success_rate, 0.0),
            "constraint_success_rate": _to_float(overall_constraint_success_rate, 0.0),
        }
    return per_dataset, metric_dataset_rates, True


def _summary_from_metric_per_dataset(
    *,
    step: int,
    num_rows: int,
    num_trajectories: int,
    per_dataset: dict[str, dict[str, Any]],
    metric_dataset_rates: dict[str, dict[str, float]],
    valid_similarity_sum: float,
    valid_similarity_count: int,
) -> dict[str, Any]:
    denominator = 0.0
    success_weighted = 0.0
    constraint_success_weighted = 0.0
    for dataset_summary in per_dataset.values():
        count = float(_to_int(dataset_summary.get("num_trajectories"), 0))
        denominator += count
        success_weighted += count * _to_float(dataset_summary.get("success_rate"), 0.0)
        constraint_success_weighted += count * _to_float(dataset_summary.get("constraint_success_rate"), 0.0)

    if denominator > 0.0:
        success_rate = success_weighted / denominator
        constraint_success_rate = constraint_success_weighted / denominator
    else:
        success_rate = _overall_metric_rate(metric_dataset_rates, "success_rate")
        constraint_success_rate = _overall_metric_rate(metric_dataset_rates, "constraint_success_rate")

    valid_similarity_mean = (
        valid_similarity_sum / float(valid_similarity_count) if valid_similarity_count > 0 else 0.0
    )
    return {
        "step": int(step),
        "num_rows": int(num_rows),
        "num_trajectories": int(num_trajectories),
        "valid_selected_similarity_count": int(valid_similarity_count),
        "valid_selected_similarity_rate": (
            float(valid_similarity_count) / float(num_trajectories) if num_trajectories > 0 else 0.0
        ),
        "success_rate": success_rate,
        "constraint_success_rate": constraint_success_rate,
        "valid_selected_similarity_mean": valid_similarity_mean,
        "success_similarity_product": success_rate * valid_similarity_mean,
    }


def build_eval_rollout_snapshot(
    *,
    rows: list[dict[str, Any]],
    step: int,
    rollout_id: str,
    default_subtask: str | None = None,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_rows = [row for row in rows if isinstance(row, dict)]

    metrics: dict[str, Any] = {}
    trajectories: list[dict[str, Any]] = []
    success_examples: dict[str, list[dict[str, Any]]] = {}
    dataset_stats: dict[str, dict[str, float]] = {}
    metrics_present = False

    total_success = 0.0
    total_constraint_success = 0.0
    total_similarity = 0.0
    total_similarity_available = 0

    for row_index, row in enumerate(normalized_rows):
        metric_blob = _extract_metrics_blob(row)
        if metric_blob:
            metrics_present = True
            metrics.update(metric_blob)

        trace_payload = _extract_trace_payload(row)
        trace_based = trace_payload is not None
        if trace_payload is not None:
            trajectory = _build_trace_trajectory(
                trace_payload,
                row=row,
                row_index=row_index,
                rollout_id=rollout_id,
                default_subtask=default_subtask,
            )
        else:
            trajectory = _build_single_turn_trajectory(
                row,
                row_index=row_index,
                rollout_id=rollout_id,
                default_subtask=default_subtask,
            )

        dataset = _dataset_name(row)
        _set_if_present(trajectory, "data_source", row.get("data_source"))
        rollup = _trajectory_rollup(trajectory, row, trace_based=trace_based)
        trajectory["success"] = float(rollup["success"])
        trajectory["constraint_success"] = float(rollup["constraint_success"])
        trajectory["turns_used"] = int(rollup["turns_used"])
        trajectory["final_similarity"] = (
            float(rollup["final_similarity"]) if rollup["final_similarity"] is not None else None
        )
        _set_if_present(trajectory, "final_candidate_smiles", rollup["final_candidate_smiles"])
        if rollup.get("final_predicted_values") is not None:
            trajectory["final_predicted_values"] = dict(rollup["final_predicted_values"])
        trajectories.append(trajectory)
        success_value = float(rollup["success"])
        constraint_success_value = float(rollup["constraint_success"])
        total_success += success_value
        total_constraint_success += constraint_success_value
        final_similarity_value = _try_float(rollup["final_similarity"])
        if final_similarity_value is not None:
            total_similarity += final_similarity_value
            total_similarity_available += 1

        stats = dataset_stats.setdefault(
            dataset,
            {
                "num_trajectories": 0.0,
                "success_count": 0.0,
                "constraint_success_count": 0.0,
                "similarity_sum": 0.0,
                "similarity_available": 0.0,
            },
        )
        stats["num_trajectories"] += 1.0
        stats["success_count"] += success_value
        stats["constraint_success_count"] += constraint_success_value
        if final_similarity_value is not None:
            stats["similarity_sum"] += final_similarity_value
            stats["similarity_available"] += 1.0

        if success_value > 0.0:
            examples = success_examples.setdefault(dataset, [])
            if len(examples) < _MAX_SUCCESS_EXAMPLES_PER_DATASET:
                examples.append(
                    {
                        "trajectory_id": trajectory.get("trajectory_id"),
                        "turns_used": int(rollup["turns_used"]),
                        "final_similarity": (
                            float(rollup["final_similarity"]) if rollup["final_similarity"] is not None else None
                        ),
                        "final_candidate_smiles": rollup["final_candidate_smiles"],
                        "x0_smiles": trajectory.get("x0_smiles"),
                        "subtask": trajectory.get("subtask"),
                    }
                )

    timestamp_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    per_dataset = {}
    for dataset, stats in dataset_stats.items():
        n = float(stats["num_trajectories"])
        per_dataset[dataset] = {
            "num_trajectories": int(n),
            "valid_selected_similarity_count": int(stats["similarity_available"]),
            "valid_selected_similarity_rate": (
                float(stats["similarity_available"]) / n if n else 0.0
            ),
            "success_rate": (float(stats["success_count"]) / n) if n else 0.0,
            "constraint_success_rate": (float(stats["constraint_success_count"]) / n) if n else 0.0,
            "valid_selected_similarity_mean": (
                float(stats["similarity_sum"]) / float(stats["similarity_available"])
                if float(stats["similarity_available"]) > 0.0
                else 0.0
            ),
        }
        per_dataset[dataset]["success_similarity_product"] = (
            per_dataset[dataset]["success_rate"] * per_dataset[dataset]["valid_selected_similarity_mean"]
        )
    summary = _summary_from_stats(
        step=step,
        num_rows=len(normalized_rows),
        num_trajectories=len(trajectories),
        success_count=total_success,
        constraint_success_count=total_constraint_success,
        valid_similarity_sum=total_similarity,
        valid_similarity_count=total_similarity_available,
    )

    snapshot = {
        "rollout_id": rollout_id,
        "timestamp_utc": timestamp_utc,
        "summary": summary,
        "per_dataset": per_dataset,
        "metrics": metrics,
        "success_examples": success_examples,
        "trajectories": trajectories,
    }
    if run_metadata is not None:
        snapshot["run_metadata"] = dict(run_metadata)
    return snapshot
