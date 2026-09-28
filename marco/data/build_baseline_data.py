from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Any

from marco.prompts.system_prompt import (
    PROMPT_MODE_DIRECT_SMILES,
    PROMPT_MODE_THINK_ANSWER,
    build_system_user_messages,
)
from marco.utils import read_jsonl, write_jsonl


SPLITS = ("train", "val", "test", "test_seen", "test_unseen")


def _canonical_file(canonical_dir: str, split: str) -> str:
    return os.path.join(canonical_dir, f"canonical_mumo_{split}.jsonl")


def _safe_instruction(record: dict[str, Any]) -> str:
    instruction = record.get("instruction")
    if isinstance(instruction, str) and instruction.strip():
        return instruction.strip()
    x0 = str(record.get("x0_smiles", ""))
    subtask = str(record.get("subtask", "molecular optimization"))
    return f"Optimize molecule {x0} for task {subtask}. Return one <SMILES>SMILES</SMILES>."


def _baseline_metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(record.get("metadata") or {})
    raw_instruction = metadata.get("raw_instruction")
    if not isinstance(raw_instruction, str) or not raw_instruction.strip():
        instruction = record.get("instruction")
        raw_instruction = instruction.strip() if isinstance(instruction, str) and instruction.strip() else None
    metadata.update(
        {
            "sample_id": record.get("sample_id"),
            "subtask": record.get("subtask"),
            "split": record.get("split"),
            "x0_smiles": record.get("x0_smiles"),
            "reference_smiles": record.get("reference_smiles"),
            "properties": record.get("properties", []),
            "raw_instruction": raw_instruction,
        }
    )
    return metadata


def _sft_row(record: dict[str, Any]) -> dict[str, Any] | None:
    answer = record.get("reference_smiles")
    if not isinstance(answer, str) or not answer.strip():
        return None

    instruction = _safe_instruction(record)
    metadata = _baseline_metadata(record)
    metadata["reference_smiles"] = answer.strip()
    return {
        "messages": build_system_user_messages(instruction, prompt_mode=PROMPT_MODE_DIRECT_SMILES)
        + [{"role": "assistant", "content": f"<SMILES>{answer.strip()}</SMILES>"}],
        "metadata": metadata,
    }


def _grpo_row(record: dict[str, Any]) -> dict[str, Any]:
    instruction = _safe_instruction(record)
    input_messages_direct = build_system_user_messages(instruction, prompt_mode=PROMPT_MODE_DIRECT_SMILES)
    input_messages_think = build_system_user_messages(instruction, prompt_mode=PROMPT_MODE_THINK_ANSWER)
    return {
        "input": instruction,
        "input_messages": input_messages_direct,
        "input_messages_direct_smiles": input_messages_direct,
        "input_messages_think_answer": input_messages_think,
        "metadata": _baseline_metadata(record),
    }


def _write_rows(rows: list[dict[str, Any]], path: str) -> None:
    write_jsonl(path, rows)


def _build_for_split(
    rows: list[dict[str, Any]],
    output_dir: str,
    split: str,
) -> dict[str, Any]:
    sft_rows: list[dict[str, Any]] = []
    grpo_rows: list[dict[str, Any]] = []

    sft_by_subtask: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grpo_by_subtask: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in rows:
        subtask = str(record.get("subtask", "unknown"))
        sft = _sft_row(record)
        grpo = _grpo_row(record)

        if sft is not None:
            sft_rows.append(sft)
            sft_by_subtask[subtask].append(sft)
        grpo_rows.append(grpo)
        grpo_by_subtask[subtask].append(grpo)

    outputs: dict[str, Any] = {
        "sft_merged": os.path.join(output_dir, "sft", f"sft_{split}.jsonl"),
        "grpo_merged": os.path.join(output_dir, "grpo_single_turn", f"grpo_{split}.jsonl"),
        "sft_by_subtask": {},
        "grpo_by_subtask": {},
    }
    _write_rows(sft_rows, outputs["sft_merged"])
    _write_rows(grpo_rows, outputs["grpo_merged"])

    for subtask, sub_rows in sft_by_subtask.items():
        path = os.path.join(output_dir, "sft", "by_subtask", subtask, f"sft_{split}.jsonl")
        _write_rows(sub_rows, path)
        outputs["sft_by_subtask"][subtask] = path

    for subtask, sub_rows in grpo_by_subtask.items():
        path = os.path.join(output_dir, "grpo_single_turn", "by_subtask", subtask, f"grpo_{split}.jsonl")
        _write_rows(sub_rows, path)
        outputs["grpo_by_subtask"][subtask] = path

    return outputs


def build_baseline_data(canonical_dir: str, output_dir: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in SPLITS:
        canonical_path = _canonical_file(canonical_dir, split)
        if not os.path.exists(canonical_path) and split in {"test_seen", "test_unseen"}:
            rows: list[dict[str, Any]] = []
        elif not os.path.exists(canonical_path):
            raise FileNotFoundError(f"Canonical split file not found: {canonical_path}")
        else:
            rows = read_jsonl(canonical_path)
        result[split] = _build_for_split(rows, output_dir, split)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SFT + single-turn GRPO baseline datasets from canonical MuMo JSONL.")
    parser.add_argument("--canonical-dir", default="data/canonical")
    parser.add_argument("--output-dir", default="data/baselines")
    args = parser.parse_args()

    outputs = build_baseline_data(canonical_dir=args.canonical_dir, output_dir=args.output_dir)
    print("Baseline dataset files written:")
    for split in SPLITS:
        split_outputs = outputs[split]
        print(f"- {split}")
        print(f"  sft_merged: {os.path.abspath(split_outputs['sft_merged'])}")
        print(f"  grpo_merged: {os.path.abspath(split_outputs['grpo_merged'])}")
        if split_outputs["sft_by_subtask"]:
            print("  sft_by_subtask:")
            for subtask, path in sorted(split_outputs["sft_by_subtask"].items()):
                print(f"    {subtask}: {os.path.abspath(path)}")
        if split_outputs["grpo_by_subtask"]:
            print("  grpo_by_subtask:")
            for subtask, path in sorted(split_outputs["grpo_by_subtask"].items()):
                print(f"    {subtask}: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
