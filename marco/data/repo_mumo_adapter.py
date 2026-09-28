from __future__ import annotations

import glob
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from marco.core_types import CanonicalRecord, PropertyTarget
from marco.utils import read_json


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_split(raw_split: str | None) -> str:
    if raw_split is None:
        return "train"
    s = raw_split.strip().lower()
    if s in {"train", "test", "val", "valid", "validation"}:
        return "val" if s in {"val", "valid", "validation"} else s
    return "train"


def _infer_test_partition(*, metadata: dict[str, Any], source_file: str) -> str | None:
    instr_setting = metadata.get("instr_setting")
    if isinstance(instr_setting, str):
        setting = instr_setting.strip().lower()
        if setting in {"seen", "unseen"}:
            return setting

    stem = Path(source_file).stem.lower()
    if "_seen_" in stem or stem.startswith("seen_") or stem.endswith("_seen"):
        return "seen"
    if "_unseen_" in stem or stem.startswith("unseen_") or stem.endswith("_unseen"):
        return "unseen"
    return None


def _parse_property_targets(task: str, properties: dict[str, Any]) -> list[PropertyTarget]:
    selected_props = [p.strip() for p in task.split("+") if p.strip()] if task else sorted(properties.keys())
    targets: list[PropertyTarget] = []
    for prop in selected_props:
        payload = properties.get(prop, {}) if isinstance(properties, dict) else {}
        source = _to_float(payload.get("source")) if isinstance(payload, dict) else None
        target = _to_float(payload.get("target")) if isinstance(payload, dict) else None
        change = _to_float(payload.get("change")) if isinstance(payload, dict) else None

        if change is None and source is not None and target is not None:
            change = target - source
        if change is None:
            change = 0.0

        direction = "increase" if change >= 0 else "decrease"
        delta = max(abs(change), 1e-6)
        targets.append(
            PropertyTarget(
                name=prop,
                direction=direction,
                delta=delta,
                source=source,
                target=target,
            )
        )
    return targets


def _build_record(raw_item: dict[str, Any], source_file: str, row_idx: int) -> CanonicalRecord | None:
    metadata = raw_item.get("meta-data", {})
    if not isinstance(metadata, dict):
        return None

    task = str(metadata.get("task", "")).strip()
    if not task:
        return None

    x0_smiles = metadata.get("source_smiles")
    if not isinstance(x0_smiles, str) or not x0_smiles.strip():
        return None

    instruction = str(raw_item.get("instruction", "")).strip()
    reference_smiles = metadata.get("target_smiles")
    if reference_smiles is None:
        reference_smiles = raw_item.get("output")
    if not isinstance(reference_smiles, str):
        reference_smiles = None

    properties = metadata.get("properties", {})
    targets = _parse_property_targets(task=task, properties=properties if isinstance(properties, dict) else {})

    sample_id = f"{Path(source_file).stem}:{row_idx}"
    split = _normalize_split(metadata.get("split"))
    test_partition = _infer_test_partition(metadata=metadata, source_file=source_file)

    record_metadata = {
        "instr_setting": metadata.get("instr_setting"),
        "instr_idx": metadata.get("instr_idx"),
        "source_file": os.path.basename(source_file),
        "raw_instruction": raw_item.get("instruction"),
        "raw_output": raw_item.get("output"),
        "raw_meta_data": metadata,
    }
    if test_partition is not None:
        record_metadata["test_partition"] = test_partition

    record = CanonicalRecord(
        sample_id=sample_id,
        split=split,
        source_repo="RePO",
        benchmark="MuMoInstruction",
        subtask=task,
        instruction=instruction,
        x0_smiles=x0_smiles.strip(),
        reference_smiles=reference_smiles.strip() if reference_smiles else None,
        properties=targets,
        metadata=record_metadata,
    )
    return record


def iter_repo_mumo_files(repo_data_root: str) -> list[str]:
    patterns = [
        os.path.join(repo_data_root, "TRAIN_multi_prop", "*.json"),
        os.path.join(repo_data_root, "TEST_multi_prop", "*.json"),
    ]
    files: list[str] = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    return sorted(set(files))


def load_repo_mumo_records(
    repo_data_root: str,
    allowed_subtasks: list[str] | None = None,
) -> list[CanonicalRecord]:
    records: list[CanonicalRecord] = []
    allowed = set(allowed_subtasks or [])

    for json_file in iter_repo_mumo_files(repo_data_root):
        payload = read_json(json_file)
        if not isinstance(payload, list):
            continue

        for idx, item in enumerate(payload):
            if not isinstance(item, dict):
                continue
            record = _build_record(item, json_file, idx)
            if record is None:
                continue
            if allowed and record.subtask not in allowed:
                continue
            records.append(record)

    return records


def canonical_record_to_dict(record: CanonicalRecord) -> dict[str, Any]:
    out = asdict(record)
    out["properties"] = [asdict(prop) for prop in record.properties]
    return out


def canonical_record_from_dict(payload: dict[str, Any]) -> CanonicalRecord:
    props = [PropertyTarget(**item) for item in payload.get("properties", [])]
    return CanonicalRecord(
        sample_id=payload["sample_id"],
        split=payload["split"],
        source_repo=payload.get("source_repo", "RePO"),
        benchmark=payload.get("benchmark", "MuMoInstruction"),
        subtask=payload["subtask"],
        instruction=payload["instruction"],
        x0_smiles=payload["x0_smiles"],
        reference_smiles=payload.get("reference_smiles"),
        properties=props,
        metadata=payload.get("metadata", {}),
    )
