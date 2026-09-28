from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from marco.data.align_sft_think_data import align_sft_think_data
from marco.data.build_baseline_data import build_baseline_data
from marco.data.build_canonical_mumo_jsonl import OUTPUT_SPLITS, _ensure_val_alias_from_test, _test_partition
from marco.data.build_repo_source_datasets import _baseline_jobs, _rlhf_jobs, _sft_jobs
from marco.prompts.system_prompt import PROMPT_MODE_THINK_ANSWER
from marco.utils import read_jsonl, write_jsonl


def _iter_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def _existing_test_partition(row: dict[str, Any]) -> str | None:
    partition = _test_partition(row)
    if partition is not None:
        return partition
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        return None
    raw = metadata.get("instr_setting")
    if not isinstance(raw, str):
        return None
    value = raw.strip().lower()
    if value in {"seen", "unseen"}:
        return value
    return None


def _load_existing_split_rows(existing_canonical_dir: str, *, allowed_subtasks: list[str] | None) -> dict[str, list[dict[str, Any]]]:
    root = Path(existing_canonical_dir)
    allowed = set(allowed_subtasks or [])
    split_rows: dict[str, list[dict[str, Any]]] = {
        split: _iter_rows(root / f"canonical_mumo_{split}.jsonl")
        for split in OUTPUT_SPLITS
    }

    if allowed:
        for split, rows in split_rows.items():
            split_rows[split] = [row for row in rows if str(row.get("subtask", "")).strip() in allowed]

    for split_name in ("test", "test_seen", "test_unseen"):
        normalized_rows: list[dict[str, Any]] = []
        for row in split_rows[split_name]:
            partition = _existing_test_partition(row)
            if partition is None:
                normalized_rows.append(row)
                continue
            cloned = dict(row)
            metadata = dict(cloned.get("metadata", {})) if isinstance(cloned.get("metadata"), dict) else {}
            metadata.setdefault("test_partition", partition)
            cloned["metadata"] = metadata
            normalized_rows.append(cloned)
        split_rows[split_name] = normalized_rows

    if not split_rows["test_seen"] or not split_rows["test_unseen"]:
        derived_seen: list[dict[str, Any]] = []
        derived_unseen: list[dict[str, Any]] = []
        for row in split_rows["test"]:
            partition = _existing_test_partition(row)
            if partition == "seen":
                derived_seen.append(row)
            elif partition == "unseen":
                derived_unseen.append(row)
        if not split_rows["test_seen"]:
            split_rows["test_seen"] = derived_seen
        if not split_rows["test_unseen"]:
            split_rows["test_unseen"] = derived_unseen

    _ensure_val_alias_from_test(split_rows)
    return split_rows


def build_canonical_from_existing(
    *,
    existing_canonical_dir: str,
    output_dir: str,
    allowed_subtasks: list[str] | None = None,
    write_by_subtask: bool = True,
) -> dict[str, Any]:
    split_rows = _load_existing_split_rows(existing_canonical_dir, allowed_subtasks=allowed_subtasks)

    all_outputs = {split: str(Path(output_dir) / f"canonical_mumo_{split}.jsonl") for split in OUTPUT_SPLITS}
    for split, path in all_outputs.items():
        write_jsonl(path, split_rows.get(split, []))

    by_subtask_outputs: dict[str, dict[str, str]] = {}
    if write_by_subtask:
        by_subtask: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        for split, rows in split_rows.items():
            for row in rows:
                subtask = str(row.get("subtask", "")).strip()
                if not subtask:
                    continue
                by_subtask[subtask][split].append(row)

        for subtask, subtask_split_rows in by_subtask.items():
            subtask_dir = Path(output_dir) / "by_subtask" / subtask
            split_paths = {split: str(subtask_dir / f"canonical_mumo_{split}.jsonl") for split in OUTPUT_SPLITS}
            for split, path in split_paths.items():
                write_jsonl(path, subtask_split_rows.get(split, []))
            by_subtask_outputs[subtask] = split_paths

    return {"all": all_outputs, "by_subtask": by_subtask_outputs}


def build_recomputed_source_datasets(
    *,
    canonical_source_dir: str,
    sft_think_dir: str,
    canonical_output_dir: str,
    baseline_jsonl_output_dir: str,
    sft_aligned_output_dir: str,
    sft_parquet_output_dir: str,
    baseline_parquet_output_dir: str,
    rlhf_parquet_output_dir: str,
    baseline_prompt_mode: str = PROMPT_MODE_THINK_ANSWER,
    rlhf_prompt_mode: str = PROMPT_MODE_THINK_ANSWER,
    max_turns: int = 5,
) -> dict[str, Any]:
    canonical_outputs = build_canonical_from_existing(
        existing_canonical_dir=canonical_source_dir,
        output_dir=canonical_output_dir,
    )
    baseline_jsonl_outputs = build_baseline_data(canonical_dir=canonical_output_dir, output_dir=baseline_jsonl_output_dir)
    sft_aligned_outputs = align_sft_think_data(
        sft_think_dir=sft_think_dir,
        canonical_dir=canonical_output_dir,
        output_dir=sft_aligned_output_dir,
    )
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
    sft_parquet_outputs = _sft_jobs(Path(sft_aligned_output_dir), Path(sft_parquet_output_dir))
    return {
        "canonical": canonical_outputs,
        "baseline_jsonl": baseline_jsonl_outputs,
        "sft_aligned": sft_aligned_outputs,
        "baseline_parquet": baseline_parquet_outputs,
        "rlhf_parquet": rlhf_parquet_outputs,
        "sft_parquet": sft_parquet_outputs,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build canonical, baseline JSONL, aligned SFT, and verl parquet datasets from recomputed canonical MARCO sources."
    )
    parser.add_argument(
        "--canonical-source-dir",
        default="data/canonical_source_recomputed",
    )
    parser.add_argument("--sft-think-dir", default="data/sft_think")
    parser.add_argument("--canonical-output-dir", default="data/canonical_recomputed")
    parser.add_argument("--baseline-jsonl-output-dir", default="data/baselines_recomputed")
    parser.add_argument("--sft-aligned-output-dir", default="data/sft_think_aligned_recomputed")
    parser.add_argument("--sft-parquet-output-dir", default="data/sft_recomputed")
    parser.add_argument("--baseline-parquet-output-dir", default="data/grpo_single_turn_recomputed")
    parser.add_argument("--rlhf-parquet-output-dir", default="data/rlhf_recomputed")
    parser.add_argument("--baseline-prompt-mode", default=PROMPT_MODE_THINK_ANSWER)
    parser.add_argument("--rlhf-prompt-mode", default=PROMPT_MODE_THINK_ANSWER)
    parser.add_argument("--max-turns", type=int, default=5)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    outputs = build_recomputed_source_datasets(
        canonical_source_dir=args.canonical_source_dir,
        sft_think_dir=args.sft_think_dir,
        canonical_output_dir=args.canonical_output_dir,
        baseline_jsonl_output_dir=args.baseline_jsonl_output_dir,
        sft_aligned_output_dir=args.sft_aligned_output_dir,
        sft_parquet_output_dir=args.sft_parquet_output_dir,
        baseline_parquet_output_dir=args.baseline_parquet_output_dir,
        rlhf_parquet_output_dir=args.rlhf_parquet_output_dir,
        baseline_prompt_mode=args.baseline_prompt_mode,
        rlhf_prompt_mode=args.rlhf_prompt_mode,
        max_turns=args.max_turns,
    )
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
