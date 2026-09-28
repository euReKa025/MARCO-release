from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Iterator

from marco.core_types import CanonicalRecord, PropertyTarget
from marco.data.build_baseline_data import build_baseline_data
from marco.data.build_canonical_mumo_jsonl import OUTPUT_SPLITS, _ensure_val_alias_from_test
from marco.data.build_repo_source_datasets import _baseline_jobs, _rlhf_jobs
from marco.data.build_recomputed_source_datasets import build_canonical_from_existing
from marco.data.repo_mumo_adapter import canonical_record_to_dict
from marco.prompts.system_prompt import PROMPT_MODE_THINK_ANSWER
from marco.utils import write_jsonl
from marco.verl.data.build_sft_parquet import build_sft_parquet


FOUR_OBJECTIVE_SUBTASKS = (
    "bbbp+drd2+plogp+qed",
    "bbbp+drd2+mutagenicity+qed",
    "bbbp+hia+mutagenicity+qed",
    "bbbp+mutagenicity+plogp+qed",
    "hia+mutagenicity+plogp+qed",
)

PROPERTY_FULL_NAMES = {
    "plogp": (
        "Penalized octanol-water partition coefficient (penalized logP)",
        "Penalized logP",
        "Penalized logP which is logP penalized by synthetic accessibility score and number of large rings",
    ),
    "qed": (
        "QED",
        "Quantitative Estimate of Drug-likeness (QED)",
        "drug-likeness quantified by QED score",
    ),
    "drd2": (
        "DRD2 inhibition",
        "Dopamine receptor D2 inhibition probability",
        "inhibition probability of Dopamine receptor D2",
    ),
    "bbbp": (
        "BBB permeability",
        "BBBP",
        "Blood-brain barrier permeability (BBBP)",
    ),
    "mutagenicity": (
        "Mutagenicity",
        "Mutagenicity predicted by Ames test",
        "probability to induce genetic alterations (mutagenicity)",
    ),
    "hia": (
        "Intestinal adsorption",
        "probability to be absorbed in the intestine",
        "human intestinal adsorption ability",
    ),
}

DEFAULT_PROPERTY_DELTAS = {
    "bbbp": 0.2,
    "drd2": 0.2,
    "hia": 0.1,
    "mutagenicity": 0.1,
    "plogp": 1.0,
    "qed": 0.1,
}

INSTRUCTION_TEMPLATES = (
    "Modify the given molecule to adjust the specified molecular properties by substituting functional groups while keeping changes to the core structure minimal. Output only the SMILES of the modified molecule, wrapped in <SMILES> </SMILES> tags.",
    "Your goal is to fine-tune the specified molecular properties of the given compound with minimal structural changes. Make the necessary adjustments and return the modified molecule in a SMILES format enclosed in <SMILES> </SMILES> tags.",
    "Adjust the structure of the given molecule to target the specified adjustments in molecular properties. Retain the core structure as much as possible. Respond with only the SMILES of the modified molecule enclosed in <SMILES> </SMILES> tags.",
    "Modify the given molecular structure to target specific property changes, aiming to keep structural adjustments minimal. Respond solely with the SMILES notation for the adjusted molecule, enclosed within <SMILES> </SMILES> tags.",
    "Alter the given molecule to meet the desired property changes with the least structural alteration possible. Output only the adjusted molecule in SMILES format, using <SMILES> </SMILES> tags.",
    "Your task is to modify the given molecule to adjust specific molecular properties while keeping structural changes as minimal as possible. Your response should only contain a valid SMILES representation of the modified molecule enclosed with <SMILES> </SMILES> tag.",
)


def _task_property_names(task: str) -> list[str]:
    return [item.strip() for item in str(task).split("+") if item.strip()]


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_properties(raw_properties: Any) -> dict[str, Any]:
    if isinstance(raw_properties, str):
        try:
            parsed = json.loads(raw_properties)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return raw_properties if isinstance(raw_properties, dict) else {}


def _direction_for_property(prop: str, change: float | None) -> str:
    if prop == "mutagenicity":
        return "decrease"
    if change is not None and change < 0:
        return "decrease"
    return "increase"


def _property_targets(task: str, raw_properties: Any) -> list[PropertyTarget]:
    properties = _normalize_properties(raw_properties)
    targets: list[PropertyTarget] = []
    for prop in _task_property_names(task):
        payload = properties.get(prop, {}) if isinstance(properties, dict) else {}
        if not isinstance(payload, dict):
            payload = {}
        source = _to_float(payload.get("source"))
        target = _to_float(payload.get("target"))
        change = _to_float(payload.get("change"))
        if change is None and source is not None and target is not None:
            change = target - source
        direction = _direction_for_property(prop, change)
        delta = abs(change) if change is not None else float(DEFAULT_PROPERTY_DELTAS.get(prop, 0.1))
        targets.append(
            PropertyTarget(
                name=prop,
                direction=direction,  # type: ignore[arg-type]
                delta=max(float(delta), 1e-6),
                source=source,
                target=target,
            )
        )
    return targets


def _property_label(prop: str, *, instr_setting: str) -> str:
    labels = PROPERTY_FULL_NAMES.get(prop)
    if labels is None:
        return prop
    return labels[-1] if instr_setting == "unseen" else labels[0]


def _format_adjust_clause(task: str, *, instr_setting: str, raw_properties: Any) -> str:
    targets = _property_targets(task, raw_properties)
    chunks = [f"{target.direction} {_property_label(target.name, instr_setting=instr_setting)}" for target in targets]
    if len(chunks) == 1:
        return chunks[0]
    return ", ".join(chunks[:-1]) + f" and {chunks[-1]}"


def _instruction(row: dict[str, Any], *, task: str, source_smiles: str, instr_setting: str, instr_idx: int) -> str:
    existing = row.get("instruction")
    if isinstance(existing, str) and existing.strip():
        return existing.strip()

    template = INSTRUCTION_TEMPLATES[instr_idx % len(INSTRUCTION_TEMPLATES)]
    adjust = _format_adjust_clause(task, instr_setting=instr_setting, raw_properties=row.get("properties"))
    return f"{template}\nInput : <SMILES> {source_smiles} </SMILES>\nAdjust: {adjust}"


def _normalize_split(row: dict[str, Any], *, hf_split: str) -> str:
    if hf_split == "test":
        return "test"
    raw = str(row.get("split") or "").strip().lower()
    if raw in {"val", "valid", "validation"}:
        return "val"
    if raw == "test":
        return "test"
    return "train"


def _test_partition(row: dict[str, Any], *, split: str) -> str | None:
    raw = str(row.get("instr_setting") or "").strip().lower()
    if raw not in {"seen", "unseen"}:
        return None
    if split in {"val", "test"}:
        return raw
    return None


def _safe_instr_idx(row: dict[str, Any]) -> int:
    try:
        return int(row.get("instr_idx", 0))
    except (TypeError, ValueError):
        return 0


def _canonical_record(row: dict[str, Any], *, hf_split: str, row_idx: int) -> dict[str, Any] | None:
    task = str(row.get("task") or "").strip()
    if not task:
        return None
    source_smiles = row.get("source_smiles")
    target_smiles = row.get("target_smiles")
    if not isinstance(source_smiles, str) or not source_smiles.strip():
        return None

    instr_setting = str(row.get("instr_setting") or "seen").strip().lower()
    if instr_setting not in {"seen", "unseen"}:
        instr_setting = "seen"
    instr_idx = _safe_instr_idx(row)
    split = _normalize_split(row, hf_split=hf_split)
    reference_smiles = target_smiles.strip() if isinstance(target_smiles, str) and target_smiles.strip() else None
    if split in {"train", "val"} and reference_smiles is None:
        return None
    partition = _test_partition(row, split=split)
    instruction = _instruction(
        row,
        task=task,
        source_smiles=source_smiles.strip(),
        instr_setting=instr_setting,
        instr_idx=instr_idx,
    )
    metadata: dict[str, Any] = {
        "instr_setting": instr_setting,
        "instr_idx": instr_idx,
        "source_file": f"{hf_split}.json",
        "hf_split": hf_split,
        "raw_instruction": instruction,
        "raw_output": reference_smiles,
        "raw_meta_data": deepcopy(row),
    }
    if partition is not None:
        metadata["test_partition"] = partition

    sample_id = f"mumoinstruct:{hf_split}:{split}:{instr_setting}:{task.replace('+', '_')}:{row_idx}"
    record = CanonicalRecord(
        sample_id=sample_id,
        split=split,  # type: ignore[arg-type]
        source_repo="GeLLMO",
        benchmark="MuMOInstruct",
        subtask=task,
        instruction=instruction,
        x0_smiles=source_smiles.strip(),
        reference_smiles=reference_smiles,
        properties=_property_targets(task, row.get("properties")),
        metadata=metadata,
    )
    return canonical_record_to_dict(record)


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _iter_json_records(path: Path) -> Iterator[dict[str, Any]]:
    with _open_text(path) as handle:
        first = ""
        while True:
            line = handle.readline()
            if not line:
                return
            if line.strip():
                first = line
                break

        if first.lstrip().startswith("["):
            payload = json.loads(first + handle.read())
            if not isinstance(payload, list):
                raise ValueError(f"Expected a JSON array in {path}")
            for item in payload:
                if isinstance(item, dict):
                    yield item
            return

        if first.lstrip().startswith("{"):
            first_item = json.loads(first)
            if isinstance(first_item, dict):
                yield first_item
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    yield item
            return

        raise ValueError(f"Unsupported JSON format in {path}")


def _find_local_split_file(root: Path, split: str) -> Path:
    candidates = (
        root / f"{split}.jsonl",
        root / f"{split}.jsonl.gz",
        root / f"{split}.json",
        root / f"{split}.json.gz",
    )
    for path in candidates:
        if path.exists():
            return path
    names = ", ".join(path.name for path in candidates)
    raise FileNotFoundError(f"Missing MuMOInstruct {split} file under {root}; tried: {names}")


def _iter_local_rows(source: Path) -> Iterator[tuple[str, int, dict[str, Any]]]:
    if source.is_file():
        split = "test" if "test" in source.name.lower() else "train"
        for idx, row in enumerate(_iter_json_records(source)):
            yield split, idx, row
        return

    for split in ("train", "test"):
        path = _find_local_split_file(source, split)
        for idx, row in enumerate(_iter_json_records(path)):
            yield split, idx, row


def _iter_hf_rows(dataset_name: str) -> Iterator[tuple[str, int, dict[str, Any]]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required to load MuMOInstruct from Hugging Face. "
            "Install datasets or pass a local directory with train/test json files."
        ) from exc

    dataset = load_dataset(dataset_name)
    for split in ("train", "test"):
        if split not in dataset:
            continue
        for idx, row in enumerate(dataset[split]):
            if isinstance(row, dict):
                yield split, idx, row


def _iter_source_rows(mumoinstruct_source: str) -> Iterator[tuple[str, int, dict[str, Any]]]:
    source_path = Path(mumoinstruct_source)
    if source_path.exists():
        yield from _iter_local_rows(source_path)
        return
    yield from _iter_hf_rows(mumoinstruct_source)


def _write_canonical_outputs(split_rows: dict[str, list[dict[str, Any]]], output_dir: str | Path) -> dict[str, Any]:
    _ensure_val_alias_from_test(split_rows)

    root = Path(output_dir)
    all_outputs = {split: str(root / f"canonical_mumo_{split}.jsonl") for split in OUTPUT_SPLITS}
    for split, path in all_outputs.items():
        write_jsonl(path, split_rows.get(split, []))

    by_subtask_rows: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for split, rows in split_rows.items():
        for row in rows:
            subtask = str(row.get("subtask") or "").strip()
            if subtask:
                by_subtask_rows[subtask][split].append(row)

    by_subtask_outputs: dict[str, dict[str, str]] = {}
    for subtask, rows_by_split in by_subtask_rows.items():
        _ensure_val_alias_from_test(rows_by_split)
        subtask_dir = root / "by_subtask" / subtask
        split_paths = {split: str(subtask_dir / f"canonical_mumo_{split}.jsonl") for split in OUTPUT_SPLITS}
        for split, path in split_paths.items():
            write_jsonl(path, rows_by_split.get(split, []))
        by_subtask_outputs[subtask] = split_paths

    return {"all": all_outputs, "by_subtask": by_subtask_outputs}


def _has_jsonl_rows(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8") as handle:
        return any(bool(line.strip()) for line in handle)


def _sft_jobs_from_baseline(baseline_root: Path, parquet_root: Path) -> dict[str, Any]:
    merged: dict[str, str] = {}
    train_src = baseline_root / "sft" / "sft_train.jsonl"
    if _has_jsonl_rows(train_src):
        train_dst = parquet_root / "train.parquet"
        build_sft_parquet(input_jsonl=str(train_src), output_parquet=str(train_dst))
        merged["train"] = str(train_dst)

    by_subtask: dict[str, dict[str, str]] = {}
    subtask_root = baseline_root / "sft" / "by_subtask"
    if subtask_root.exists():
        for subdir in sorted(path for path in subtask_root.iterdir() if path.is_dir()):
            src = subdir / "sft_train.jsonl"
            if not _has_jsonl_rows(src):
                continue
            dst = parquet_root / "by_subtask" / subdir.name / "train.parquet"
            build_sft_parquet(input_jsonl=str(src), output_parquet=str(dst))
            by_subtask[subdir.name] = {"train": str(dst)}

    return {"merged": merged, "by_subtask": by_subtask}


def build_mumoinstruct_four_objective_datasets(
    *,
    mumoinstruct_source: str,
    canonical_output_dir: str,
    baseline_jsonl_output_dir: str,
    sft_parquet_output_dir: str,
    baseline_parquet_output_dir: str,
    rlhf_parquet_output_dir: str,
    subtasks: list[str] | None = None,
    baseline_prompt_mode: str = PROMPT_MODE_THINK_ANSWER,
    rlhf_prompt_mode: str = PROMPT_MODE_THINK_ANSWER,
    max_turns: int = 5,
) -> dict[str, Any]:
    allowed_subtasks = set(subtasks or FOUR_OBJECTIVE_SUBTASKS)
    if not allowed_subtasks:
        raise ValueError("At least one subtask is required")
    for subtask in allowed_subtasks:
        if len(_task_property_names(subtask)) != 4:
            raise ValueError(f"Expected four-objective subtask, got: {subtask}")

    split_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in OUTPUT_SPLITS}
    rows_seen = 0
    rows_kept = 0
    for hf_split, row_idx, row in _iter_source_rows(mumoinstruct_source):
        rows_seen += 1
        task = str(row.get("task") or "").strip()
        if task not in allowed_subtasks:
            continue
        record = _canonical_record(row, hf_split=hf_split, row_idx=row_idx)
        if record is None:
            continue
        split = str(record.get("split") or "train")
        if split not in split_rows:
            continue
        partition = None
        metadata = record.get("metadata")
        if isinstance(metadata, dict):
            partition = metadata.get("test_partition")
        split_rows[split].append(record)
        if split == "test" and partition in {"seen", "unseen"}:
            split_rows[f"test_{partition}"].append(record)
        rows_kept += 1

    canonical_outputs = _write_canonical_outputs(split_rows, canonical_output_dir)
    baseline_jsonl_outputs = build_baseline_data(
        canonical_dir=canonical_output_dir,
        output_dir=baseline_jsonl_output_dir,
    )
    sft_parquet_outputs = _sft_jobs_from_baseline(Path(baseline_jsonl_output_dir), Path(sft_parquet_output_dir))
    baseline_parquet_outputs = _baseline_jobs(
        Path(baseline_jsonl_output_dir),
        Path(baseline_parquet_output_dir),
        prompt_mode=baseline_prompt_mode,
    )
    rlhf_parquet_outputs = _rlhf_jobs(
        Path(canonical_output_dir),
        Path(rlhf_parquet_output_dir),
        max_turns=max_turns,
        prompt_mode=rlhf_prompt_mode,
    )

    return {
        "mumoinstruct_source": str(mumoinstruct_source),
        "subtasks": sorted(allowed_subtasks),
        "rows_seen": rows_seen,
        "rows_kept": rows_kept,
        "canonical": canonical_outputs,
        "baseline_jsonl": baseline_jsonl_outputs,
        "sft_parquet": sft_parquet_outputs,
        "baseline_parquet": baseline_parquet_outputs,
        "rlhf_parquet": rlhf_parquet_outputs,
    }


def build_mumoinstruct_four_objective_datasets_from_canonical(
    *,
    canonical_source_dir: str,
    canonical_output_dir: str,
    baseline_jsonl_output_dir: str,
    sft_parquet_output_dir: str,
    baseline_parquet_output_dir: str,
    rlhf_parquet_output_dir: str,
    subtasks: list[str] | None = None,
    baseline_prompt_mode: str = PROMPT_MODE_THINK_ANSWER,
    rlhf_prompt_mode: str = PROMPT_MODE_THINK_ANSWER,
    max_turns: int = 5,
) -> dict[str, Any]:
    allowed_subtasks = set(subtasks or FOUR_OBJECTIVE_SUBTASKS)
    if not allowed_subtasks:
        raise ValueError("At least one subtask is required")
    for subtask in allowed_subtasks:
        if len(_task_property_names(subtask)) != 4:
            raise ValueError(f"Expected four-objective subtask, got: {subtask}")

    canonical_outputs = build_canonical_from_existing(
        existing_canonical_dir=canonical_source_dir,
        output_dir=canonical_output_dir,
        allowed_subtasks=sorted(allowed_subtasks),
    )
    baseline_jsonl_outputs = build_baseline_data(
        canonical_dir=canonical_output_dir,
        output_dir=baseline_jsonl_output_dir,
    )
    sft_parquet_outputs = _sft_jobs_from_baseline(Path(baseline_jsonl_output_dir), Path(sft_parquet_output_dir))
    baseline_parquet_outputs = _baseline_jobs(
        Path(baseline_jsonl_output_dir),
        Path(baseline_parquet_output_dir),
        prompt_mode=baseline_prompt_mode,
    )
    rlhf_parquet_outputs = _rlhf_jobs(
        Path(canonical_output_dir),
        Path(rlhf_parquet_output_dir),
        max_turns=max_turns,
        prompt_mode=rlhf_prompt_mode,
    )

    return {
        "canonical_source_dir": str(canonical_source_dir),
        "subtasks": sorted(allowed_subtasks),
        "canonical": canonical_outputs,
        "baseline_jsonl": baseline_jsonl_outputs,
        "sft_parquet": sft_parquet_outputs,
        "baseline_parquet": baseline_parquet_outputs,
        "rlhf_parquet": rlhf_parquet_outputs,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build MARCO-verl datasets for four-objective tasks from NingLab/MuMOInstruct pair data."
    )
    parser.add_argument(
        "--mumoinstruct-source",
        default="NingLab/MuMOInstruct",
        help="Hugging Face dataset id or local directory/file containing train/test JSON or JSONL data.",
    )
    parser.add_argument(
        "--canonical-source-dir",
        default=None,
        help="Optional existing canonical JSONL root. If set, derive downstream MuMOInstruct4 datasets from it instead of loading Hugging Face pair data.",
    )
    parser.add_argument("--canonical-output-dir", default="data/canonical_mumoinstruct4")
    parser.add_argument("--baseline-jsonl-output-dir", default="data/baselines_mumoinstruct4")
    parser.add_argument("--sft-parquet-output-dir", default="data/sft_mumoinstruct4")
    parser.add_argument("--baseline-parquet-output-dir", default="data/grpo_single_turn_mumoinstruct4")
    parser.add_argument("--rlhf-parquet-output-dir", default="data/rlhf_mumoinstruct4")
    parser.add_argument("--subtasks", default=",".join(FOUR_OBJECTIVE_SUBTASKS))
    parser.add_argument("--baseline-prompt-mode", default=PROMPT_MODE_THINK_ANSWER)
    parser.add_argument("--rlhf-prompt-mode", default=PROMPT_MODE_THINK_ANSWER)
    parser.add_argument("--max-turns", type=int, default=5)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    subtasks = [item.strip() for item in args.subtasks.split(",") if item.strip()]
    if args.canonical_source_dir:
        outputs = build_mumoinstruct_four_objective_datasets_from_canonical(
            canonical_source_dir=args.canonical_source_dir,
            canonical_output_dir=args.canonical_output_dir,
            baseline_jsonl_output_dir=args.baseline_jsonl_output_dir,
            sft_parquet_output_dir=args.sft_parquet_output_dir,
            baseline_parquet_output_dir=args.baseline_parquet_output_dir,
            rlhf_parquet_output_dir=args.rlhf_parquet_output_dir,
            subtasks=subtasks,
            baseline_prompt_mode=args.baseline_prompt_mode,
            rlhf_prompt_mode=args.rlhf_prompt_mode,
            max_turns=args.max_turns,
        )
    else:
        outputs = build_mumoinstruct_four_objective_datasets(
            mumoinstruct_source=args.mumoinstruct_source,
            canonical_output_dir=args.canonical_output_dir,
            baseline_jsonl_output_dir=args.baseline_jsonl_output_dir,
            sft_parquet_output_dir=args.sft_parquet_output_dir,
            baseline_parquet_output_dir=args.baseline_parquet_output_dir,
            rlhf_parquet_output_dir=args.rlhf_parquet_output_dir,
            subtasks=subtasks,
            baseline_prompt_mode=args.baseline_prompt_mode,
            rlhf_prompt_mode=args.rlhf_prompt_mode,
            max_turns=args.max_turns,
        )
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
