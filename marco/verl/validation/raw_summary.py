from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from marco.env.molecule_validation import is_valid_candidate_smiles
from marco.env.predictor_client import PredictorClient
from marco.env.similarity import Chem
from marco.utils import read_json, read_jsonl


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


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _is_valid_smiles(smiles: Any) -> bool:
    if not isinstance(smiles, str) or not smiles.strip():
        return False
    if Chem is None:
        return False
    return is_valid_candidate_smiles(smiles.strip())


def _normalize_property_specs(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        out.append(dict(item))
    return out


def _is_improved(direction: Any, *, reference_value: float, candidate_value: float) -> bool:
    if str(direction).strip().lower() == "decrease":
        return candidate_value < reference_value
    return candidate_value > reference_value


def _similarity_is_acceptable(
    similarity: float,
    *,
    similarity_threshold: float,
    similarity_target_low: float | None = None,
    similarity_target_high: float | None = None,
    similarity_copy_threshold: float | None = None,
) -> bool:
    low = float(similarity_target_low) if similarity_target_low is not None else float(similarity_threshold)
    high = similarity_target_high
    if high is None:
        high = similarity_copy_threshold
    if high is None:
        return similarity >= low
    high = float(high)
    if high < low:
        low, high = high, low
    return similarity >= low and similarity < high


def build_raw_reference_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        subtask = str(row.get("subtask", "")).strip()
        x0_smiles = str(row.get("x0_smiles", "")).strip()
        if not subtask or not x0_smiles:
            continue
        property_specs = _normalize_property_specs(row.get("properties"))
        properties: dict[str, float] = {}
        for prop in property_specs:
            name = str(prop.get("name", "")).strip()
            if not name or prop.get("source") is None:
                continue
            properties[name] = _to_float(prop.get("source"), 0.0)
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        partition = str(
            metadata.get("test_partition")
            or metadata.get("instr_setting")
            or row.get("split")
            or "unknown"
        )
        index[(subtask, x0_smiles)] = {
            "sample_id": row.get("sample_id"),
            "subtask": subtask,
            "x0_smiles": x0_smiles,
            "properties": properties,
            "property_specs": property_specs,
            "reference_smiles": row.get("reference_smiles"),
            "partition": partition,
        }
    return index


def load_raw_reference_index(canonical_jsonl: str | Path) -> dict[tuple[str, str], dict[str, Any]]:
    return build_raw_reference_index(read_jsonl(canonical_jsonl))


def _empty_summary(
    *,
    partition: str,
    validation_mode: str,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = {
        "data_variant": "raw",
        "partition": partition,
        "validation_mode": validation_mode,
        "metrics": {
            "num_trajectories": 0,
            "num_valid_candidates": 0,
            "valid_selected_similarity_count": 0,
            "valid_selected_similarity_rate": 0.0,
            "success_rate": 0.0,
            "constraint_success_rate": 0.0,
            "valid_selected_similarity_mean": 0.0,
            "success_similarity_product": 0.0,
        },
        "rows": [],
    }
    if run_metadata is not None:
        summary["run_metadata"] = dict(run_metadata)
    return summary


def build_raw_style_summary(
    *,
    trajectories: list[dict[str, Any]],
    raw_reference_index: dict[tuple[str, str], dict[str, Any]],
    predict_properties: Callable[[str, list[str]], dict[str, float]],
    default_subtask: str | None = None,
    partition: str,
    validation_mode: str,
    similarity_threshold: float,
    similarity_target_low: float | None = None,
    similarity_target_high: float | None = None,
    similarity_copy_threshold: float | None = None,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not trajectories:
        return _empty_summary(partition=partition, validation_mode=validation_mode, run_metadata=run_metadata)

    rows: list[dict[str, Any]] = []
    diff_sums: dict[str, float] = {}
    diff_counts: dict[str, int] = {}
    valid_similarity_sum = 0.0
    valid_similarity_available = 0
    valid_candidates = 0
    success_count = 0.0
    constraint_success_count = 0.0

    for trajectory in trajectories:
        if not isinstance(trajectory, dict):
            continue
        subtask = str(_first_present(trajectory.get("subtask"), default_subtask) or "").strip()
        x0_smiles = str(trajectory.get("x0_smiles", "")).strip()
        trajectory_id = str(trajectory.get("trajectory_id", ""))
        final_candidate_smiles = trajectory.get("final_candidate_smiles")
        if isinstance(final_candidate_smiles, str):
            final_candidate_smiles = final_candidate_smiles.strip() or None
        else:
            final_candidate_smiles = None
        final_similarity = _try_float(trajectory.get("final_similarity"))
        raw_reference = raw_reference_index.get((subtask, x0_smiles))
        property_specs = _normalize_property_specs(
            trajectory.get("properties")
            or (raw_reference.get("property_specs") if isinstance(raw_reference, dict) else None)
        )

        row: dict[str, Any] = {
            "trajectory_id": trajectory_id,
            "subtask": subtask,
            "x0_smiles": x0_smiles,
            "final_candidate_smiles": final_candidate_smiles,
            "final_similarity": final_similarity,
            "success": 0.0,
            "constraint_success": 0.0,
            "property_differences": {},
        }
        if raw_reference is not None:
            row["sample_id"] = raw_reference.get("sample_id")
            row["reference_partition"] = raw_reference.get("partition")
            if raw_reference.get("reference_smiles") is not None:
                row["reference_smiles"] = raw_reference.get("reference_smiles")

        if raw_reference is None:
            row["invalid_reason"] = "missing_reference"
            rows.append(row)
            continue

        if final_candidate_smiles is None:
            row["invalid_reason"] = "missing_candidate"
            rows.append(row)
            continue

        property_names = [str(prop.get("name")) for prop in property_specs if str(prop.get("name", "")).strip()]
        if not property_names:
            row["invalid_reason"] = "missing_properties"
            rows.append(row)
            continue

        if not _is_valid_smiles(final_candidate_smiles):
            row["invalid_reason"] = "invalid_candidate"
            rows.append(row)
            continue

        persisted_candidate_properties = (
            dict(trajectory.get("final_predicted_values"))
            if isinstance(trajectory.get("final_predicted_values"), dict) and trajectory.get("final_predicted_values")
            else None
        )
        if persisted_candidate_properties is not None:
            candidate_properties = {
                name: _to_float(persisted_candidate_properties.get(name), 0.0)
                for name in property_names
                if name in persisted_candidate_properties
            }
        else:
            try:
                candidate_properties = predict_properties(final_candidate_smiles, property_names)
            except Exception as exc:  # pragma: no cover - exercised through caller behavior
                row["invalid_reason"] = "predictor_error"
                row["predictor_error"] = str(exc)
                rows.append(row)
                continue

        valid_candidates += 1
        if final_similarity is not None:
            valid_similarity_sum += final_similarity
            valid_similarity_available += 1

        property_success = True
        property_differences: dict[str, float] = {}
        reference_values: dict[str, float] = {}
        predicted_values: dict[str, float] = {}
        for prop in property_specs:
            name = str(prop.get("name", "")).strip()
            if not name:
                continue
            reference_value = _to_float((raw_reference.get("properties") or {}).get(name), 0.0)
            candidate_value = _to_float(candidate_properties.get(name), 0.0)
            property_differences[name] = candidate_value - reference_value
            reference_values[name] = reference_value
            predicted_values[name] = candidate_value
            if not _is_improved(prop.get("direction"), reference_value=reference_value, candidate_value=candidate_value):
                property_success = False
            diff_sums[name] = diff_sums.get(name, 0.0) + property_differences[name]
            diff_counts[name] = diff_counts.get(name, 0) + 1

        similarity_ok = (
            _similarity_is_acceptable(
                final_similarity,
                similarity_threshold=similarity_threshold,
                similarity_target_low=similarity_target_low,
                similarity_target_high=similarity_target_high,
                similarity_copy_threshold=similarity_copy_threshold,
            )
            if final_similarity is not None
            else False
        )
        success = 1.0 if property_success else 0.0
        constraint_success = 1.0 if (property_success and similarity_ok) else 0.0
        success_count += success
        constraint_success_count += constraint_success

        row["reference_values"] = reference_values
        row["predicted_values"] = predicted_values
        row["property_differences"] = property_differences
        row["success"] = success
        row["constraint_success"] = constraint_success
        rows.append(row)

    num_trajectories = len(rows)
    summary = _empty_summary(partition=partition, validation_mode=validation_mode, run_metadata=run_metadata)
    summary["rows"] = rows
    summary["metrics"]["num_trajectories"] = num_trajectories
    summary["metrics"]["num_valid_candidates"] = valid_candidates
    if num_trajectories > 0:
        summary["metrics"]["success_rate"] = success_count / float(num_trajectories)
        summary["metrics"]["constraint_success_rate"] = constraint_success_count / float(num_trajectories)
        summary["metrics"]["valid_selected_similarity_rate"] = valid_similarity_available / float(num_trajectories)
    summary["metrics"]["valid_selected_similarity_count"] = valid_similarity_available
    if valid_similarity_available > 0:
        summary["metrics"]["valid_selected_similarity_mean"] = valid_similarity_sum / float(valid_similarity_available)
    product_similarity = summary["metrics"]["valid_selected_similarity_mean"]
    summary["metrics"]["success_similarity_product"] = summary["metrics"]["success_rate"] * product_similarity
    for name in sorted(diff_sums):
        count = diff_counts.get(name, 0)
        if count > 0:
            summary["metrics"][f"diff_{name}"] = diff_sums[name] / float(count)
    return summary


def resolve_raw_canonical_jsonl(
    *,
    root_dir: str | Path,
    subtask: str,
    partition: str,
) -> Path:
    partition_key = str(partition or "seen").strip().lower()
    if partition_key in {"seen", "val"}:
        filename = "canonical_mumo_val.jsonl"
    elif partition_key in {"unseen", "test_unseen"}:
        filename = "canonical_mumo_test_unseen.jsonl"
    elif partition_key in {"all", "test"}:
        filename = "canonical_mumo_test.jsonl"
    else:
        filename = "canonical_mumo_test_seen.jsonl"
    root = Path(root_dir)
    property_count = len([part for part in str(subtask).split("+") if part.strip()])
    candidate_roots = (
        ["canonical_mumoinstruct4", "canonical_mumoinstruct4_recomputed", "canonical"]
        if property_count >= 4
        else ["canonical", "canonical_recomputed"]
    )
    candidates = [root / "data" / name / "by_subtask" / subtask / filename for name in candidate_roots]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def build_raw_style_summary_from_snapshot(
    *,
    snapshot_json: str | Path,
    canonical_jsonl: str | Path | None,
    subtask: str | None,
    partition: str,
    validation_mode: str,
    similarity_threshold: float,
    similarity_target_low: float | None = None,
    similarity_target_high: float | None = None,
    similarity_copy_threshold: float | None = None,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    snapshot = read_json(snapshot_json)
    trajectories = list(snapshot.get("trajectories") or [])
    snapshot_run_metadata = snapshot.get("run_metadata") if isinstance(snapshot.get("run_metadata"), dict) else None
    effective_run_metadata = run_metadata if run_metadata is not None else snapshot_run_metadata
    raw_reference_index: dict[tuple[str, str], dict[str, Any]] = {}
    if canonical_jsonl is not None and Path(canonical_jsonl).exists():
        raw_reference_index = load_raw_reference_index(canonical_jsonl)

    predictor = PredictorClient()
    summary = build_raw_style_summary(
        trajectories=trajectories,
        raw_reference_index=raw_reference_index,
        predict_properties=predictor.predict,
        default_subtask=str(subtask).strip() if subtask else None,
        partition=partition,
        validation_mode=validation_mode,
        similarity_threshold=similarity_threshold,
        similarity_target_low=similarity_target_low,
        similarity_target_high=similarity_target_high,
        similarity_copy_threshold=similarity_copy_threshold,
        run_metadata=effective_run_metadata,
    )
    if canonical_jsonl is not None:
        summary["canonical_jsonl"] = str(canonical_jsonl)
    rollout_id = snapshot.get("rollout_id")
    if rollout_id is not None:
        summary["rollout_id"] = rollout_id
    return summary
