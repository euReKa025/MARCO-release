from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Any

from .repo_mumo_adapter import canonical_record_to_dict, load_repo_mumo_records
from marco.utils import write_jsonl


OUTPUT_SPLITS = ("train", "val", "test", "test_seen", "test_unseen")


def _rows_with_split(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        cloned = dict(row)
        cloned["split"] = split
        out.append(cloned)
    return out


def _ensure_val_alias_from_test(split_rows: dict[str, list[dict[str, Any]]]) -> None:
    if split_rows.get("val"):
        return
    seen_rows = split_rows.get("test_seen", [])
    if seen_rows:
        split_rows["val"] = _rows_with_split(seen_rows, "val")
        return
    test_rows = split_rows.get("test", [])
    if test_rows:
        split_rows["val"] = _rows_with_split(test_rows, "val")


def _test_partition(payload: dict[str, Any]) -> str | None:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return None
    raw = metadata.get("test_partition")
    if not isinstance(raw, str):
        return None
    value = raw.strip().lower()
    if value in {"seen", "unseen"}:
        return value
    return None


def build_canonical(
    repo_data_root: str,
    output_dir: str,
    allowed_subtasks: list[str] | None = None,
    write_by_subtask: bool = True,
) -> dict[str, Any]:
    records = load_repo_mumo_records(repo_data_root=repo_data_root, allowed_subtasks=allowed_subtasks)

    by_split: dict[str, list[dict]] = defaultdict(list)
    by_subtask_split: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        payload = canonical_record_to_dict(record)
        by_split[record.split].append(payload)
        by_subtask_split[record.subtask][record.split].append(payload)
        if record.split == "test":
            partition = _test_partition(payload)
            if partition in {"seen", "unseen"}:
                split_name = f"test_{partition}"
                by_split[split_name].append(payload)
                by_subtask_split[record.subtask][split_name].append(payload)

    _ensure_val_alias_from_test(by_split)
    for split_map in by_subtask_split.values():
        _ensure_val_alias_from_test(split_map)

    all_outputs = {split: os.path.join(output_dir, f"canonical_mumo_{split}.jsonl") for split in OUTPUT_SPLITS}

    for split, path in all_outputs.items():
        write_jsonl(path, by_split.get(split, []))

    subtask_outputs: dict[str, dict[str, str]] = {}
    if write_by_subtask:
        for subtask, split_map in by_subtask_split.items():
            subtask_dir = os.path.join(output_dir, "by_subtask", subtask)
            split_paths = {
                split: os.path.join(subtask_dir, f"canonical_mumo_{split}.jsonl")
                for split in OUTPUT_SPLITS
            }
            for split, path in split_paths.items():
                write_jsonl(path, split_map.get(split, []))
            subtask_outputs[subtask] = split_paths

    return {
        "all": all_outputs,
        "by_subtask": subtask_outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical MuMo JSONL from RePO data")
    parser.add_argument(
        "--repo-data-root",
        default="../RePO/data",
    )
    parser.add_argument("--output-dir", default="data/canonical")
    parser.add_argument(
        "--subtasks",
        default="",
        help="Comma-separated subtasks, e.g. bbbp+drd2+plogp,bbbp+drd2+qed",
    )
    parser.add_argument(
        "--no-by-subtask",
        action="store_true",
        help="Disable writing per-subtask split files under output_dir/by_subtask",
    )
    args = parser.parse_args()

    subtasks = [s.strip() for s in args.subtasks.split(",") if s.strip()] or None
    outputs = build_canonical(
        repo_data_root=args.repo_data_root,
        output_dir=args.output_dir,
        allowed_subtasks=subtasks,
        write_by_subtask=not args.no_by_subtask,
    )

    print("Canonical merged files written:")
    for split, path in outputs["all"].items():
        print(f"  - {split}: {os.path.abspath(path)}")
    if outputs["by_subtask"]:
        print("Per-subtask files written:")
        for subtask, split_paths in sorted(outputs["by_subtask"].items()):
            print(f"  - {subtask}")
            for split, path in split_paths.items():
                print(f"      {split}: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
