from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from marco.utils import read_jsonl
from marco.verl.data.schema import dataset_instruction_from_record


LEGACY_FIRST_TURN_MARKERS = (
    "Turn: 1",
    "Source molecule (x0)",
    "Current molecule (x_t)",
)


def _canonical_files(canonical_dir: Path) -> list[Path]:
    return sorted(path for path in canonical_dir.rglob("canonical_mumo_*.jsonl") if path.is_file())


def _parquet_files(root: Path | None) -> list[Path]:
    if root is None:
        return []
    if root.is_file():
        return [root]
    return sorted(path for path in root.rglob("*.parquet") if path.is_file())


def _load_expected_instructions(canonical_dir: Path) -> dict[str, str]:
    if not canonical_dir.exists():
        raise FileNotFoundError(f"canonical_dir does not exist: {canonical_dir}")
    expected: dict[str, str] = {}
    for path in _canonical_files(canonical_dir):
        for record in read_jsonl(str(path)):
            sample_id = str(record.get("sample_id", "")).strip()
            if not sample_id:
                continue
            instruction = dataset_instruction_from_record(record)
            previous = expected.get(sample_id)
            if previous is not None and previous != instruction:
                raise ValueError(f"conflicting canonical instruction for sample_id={sample_id}")
            expected[sample_id] = instruction
    if not expected:
        raise ValueError(f"no canonical records found under {canonical_dir}")
    return expected


def _first_user_content(messages: Any) -> str | None:
    if not isinstance(messages, list):
        return None
    for message in messages:
        if not isinstance(message, dict):
            continue
        if str(message.get("role", "")).strip().lower() == "user":
            content = message.get("content")
            return content if isinstance(content, str) else None
    return None


def _contains_legacy_first_turn(text: str) -> bool:
    return any(marker in text for marker in LEGACY_FIRST_TURN_MARKERS)


def _sample_id_from_rlhf_row(row: dict[str, Any]) -> str:
    extra_info = row.get("extra_info")
    if isinstance(extra_info, dict):
        for key in ("sample_id", "index"):
            value = extra_info.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    return ""


def _sample_id_from_sft_row(row: dict[str, Any]) -> str:
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        value = metadata.get("sample_id")
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _check_prompt(
    *,
    dataset: str,
    file_path: Path,
    row_idx: int,
    sample_id: str,
    user_content: str | None,
    expected: dict[str, str],
) -> dict[str, Any] | None:
    location = f"{file_path}:{row_idx}"
    if not sample_id:
        return {
            "dataset": dataset,
            "location": location,
            "error_type": "missing_sample_id",
            "error": "row has no sample_id",
        }
    expected_instruction = expected.get(sample_id)
    if expected_instruction is None:
        return {
            "dataset": dataset,
            "location": location,
            "sample_id": sample_id,
            "error_type": "missing_canonical_record",
            "error": "sample_id is not present in canonical records",
        }
    if user_content is None:
        return {
            "dataset": dataset,
            "location": location,
            "sample_id": sample_id,
            "error_type": "missing_user_message",
            "error": "row has no first user message",
        }
    if _contains_legacy_first_turn(user_content):
        return {
            "dataset": dataset,
            "location": location,
            "sample_id": sample_id,
            "error_type": "legacy_structured_prompt",
            "error": "legacy structured first-turn prompt detected",
        }
    if user_content.strip() != expected_instruction.strip():
        return {
            "dataset": dataset,
            "location": location,
            "sample_id": sample_id,
            "error_type": "instruction_mismatch",
            "error": "first user message does not match canonical dataset instruction",
            "expected_prefix": expected_instruction[:160],
            "actual_prefix": user_content[:160],
        }
    return None


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def verify_instruction_prompt_alignment(
    *,
    canonical_dir: str | Path,
    rlhf_parquet_dir: str | Path | None = None,
    sft_parquet_dir: str | Path | None = None,
    max_failures: int = 20,
) -> dict[str, Any]:
    canonical_path = Path(canonical_dir)
    expected = _load_expected_instructions(canonical_path)
    failures: list[dict[str, Any]] = []
    checked_rows = {"rlhf": 0, "sft": 0}
    checked_files = {"rlhf": 0, "sft": 0}

    for path in _parquet_files(Path(rlhf_parquet_dir) if rlhf_parquet_dir is not None else None):
        checked_files["rlhf"] += 1
        for row_idx, row in enumerate(_read_parquet_rows(path)):
            checked_rows["rlhf"] += 1
            failure = _check_prompt(
                dataset="rlhf",
                file_path=path,
                row_idx=row_idx,
                sample_id=_sample_id_from_rlhf_row(row),
                user_content=_first_user_content(row.get("prompt")),
                expected=expected,
            )
            if failure is not None:
                failures.append(failure)
                if len(failures) >= max_failures:
                    break
        if len(failures) >= max_failures:
            break

    if len(failures) < max_failures:
        for path in _parquet_files(Path(sft_parquet_dir) if sft_parquet_dir is not None else None):
            checked_files["sft"] += 1
            for row_idx, row in enumerate(_read_parquet_rows(path)):
                checked_rows["sft"] += 1
                failure = _check_prompt(
                    dataset="sft",
                    file_path=path,
                    row_idx=row_idx,
                    sample_id=_sample_id_from_sft_row(row),
                    user_content=_first_user_content(row.get("messages")),
                    expected=expected,
                )
                if failure is not None:
                    failures.append(failure)
                    if len(failures) >= max_failures:
                        break
            if len(failures) >= max_failures:
                break

    report: dict[str, Any] = {
        "status": "ok" if not failures else "failed",
        "canonical_dir": str(canonical_path),
        "num_canonical_records": len(expected),
        "checked_files": checked_files,
        "checked_rows": checked_rows,
        "num_failures": len(failures),
        "failures": failures,
    }
    if failures:
        first = failures[0]
        raise ValueError(
            "instruction prompt alignment failed: "
            f"{first.get('error')} at {first.get('location')} sample_id={first.get('sample_id', '')}"
        )
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify MARCO SFT/RLHF parquet first-turn prompts use dataset instructions.")
    parser.add_argument("--canonical-dir", required=True)
    parser.add_argument("--rlhf-parquet-dir")
    parser.add_argument("--sft-parquet-dir")
    parser.add_argument("--max-failures", type=int, default=20)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    report = verify_instruction_prompt_alignment(
        canonical_dir=args.canonical_dir,
        rlhf_parquet_dir=args.rlhf_parquet_dir,
        sft_parquet_dir=args.sft_parquet_dir,
        max_failures=args.max_failures,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
