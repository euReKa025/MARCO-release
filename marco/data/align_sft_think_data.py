from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from typing import Any

from marco.data.build_baseline_data import _safe_instruction
from marco.env.molecule_validation import validate_model_output
from marco.prompts.system_prompt import (
    PROMPT_MODE_THINK_ANSWER,
    build_system_user_messages,
    get_user_response_instruction,
)
from marco.utils import read_jsonl, write_jsonl


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
_SMILES_BLOCK_RE = re.compile(r"<(?P<tag>answer|SMILES)>(.*?)</(?P=tag)>", re.IGNORECASE | re.DOTALL)
_OLD_OUTPUT_PATTERNS = [
    re.compile(r"Output only the adjusted molecule in SMILES format, using <SMILES>\s*</SMILES> tags\.?", re.IGNORECASE),
    re.compile(r"Output only the SMILES .*?<SMILES>\s*</SMILES>\s*tags?\.?", re.IGNORECASE | re.DOTALL),
    re.compile(r"Your response should only contain .*?</SMILES> tag\.?", re.IGNORECASE | re.DOTALL),
    re.compile(r"Return only <answer>SMILES</answer>\.?", re.IGNORECASE),
    re.compile(r"Return only <SMILES>SMILES</SMILES>\.?", re.IGNORECASE),
    re.compile(r"Respond with only .*?<SMILES>\s*</SMILES>\s*tags?\.?", re.IGNORECASE | re.DOTALL),
    re.compile(r"Return the modified molecule .*?<SMILES>\s*</SMILES>\s*tags?\.?", re.IGNORECASE | re.DOTALL),
    re.compile(r"Return only the SMILES .*?<SMILES>\s*</SMILES>\s*tags?\.?", re.IGNORECASE | re.DOTALL),
]
_PLACEHOLDER_ANSWERS = {"", "smiles", "molecule", "valid molecule", "answer", "output"}
_THINK_PREFIX_RE = re.compile(r"(<think>\s*)think\s+", re.IGNORECASE)


def _canonical_file(canonical_dir: str, split: str) -> str:
    return os.path.join(canonical_dir, f"canonical_mumo_{split}.jsonl")


def _read_jsonl_if_exists(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    return read_jsonl(path)


def _looks_aligned_train_row(row: dict[str, Any]) -> bool:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        return False
    roles = [str(message.get("role", "")).strip().lower() for message in messages[:3] if isinstance(message, dict)]
    return roles[:3] == ["system", "user", "assistant"]


def _normalize_sample_id(sample_id: Any) -> str:
    return str(sample_id).strip()


def _canonical_raw_instruction(record: dict[str, Any]) -> str | None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return None
    raw_instruction = metadata.get("raw_instruction")
    if isinstance(raw_instruction, str) and raw_instruction.strip():
        return raw_instruction.strip()
    return None


def _canonical_user_prompt(record: dict[str, Any], *, fallback_prompt: str) -> str:
    raw_instruction = _canonical_raw_instruction(record)
    if raw_instruction is not None:
        return raw_instruction
    instruction = record.get("instruction")
    if isinstance(instruction, str) and instruction.strip():
        return instruction.strip()
    return fallback_prompt


def _canonical_semantic_payload(record: dict[str, Any]) -> dict[str, Any]:
    payload = dict(record)
    payload["sample_id"] = _normalize_sample_id(payload.get("sample_id"))
    payload.pop("split", None)
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        metadata_payload = dict(metadata)
        metadata_payload.pop("split", None)
        payload["metadata"] = metadata_payload
    return payload


def _is_allowed_val_test_alias_duplicate(existing: dict[str, Any], candidate: dict[str, Any]) -> bool:
    existing_split = str(existing.get("split", "")).strip()
    candidate_split = str(candidate.get("split", "")).strip()
    if {existing_split, candidate_split} != {"val", "test"}:
        return False
    return _canonical_semantic_payload(existing) == _canonical_semantic_payload(candidate)


def _load_canonical_index(canonical_dir: str) -> dict[str, dict[str, Any]]:
    canonical_by_sample_id: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        for record in _read_jsonl_if_exists(_canonical_file(canonical_dir, split)):
            sample_id = _normalize_sample_id(record.get("sample_id"))
            normalized_record = {**record, "sample_id": sample_id}
            existing = canonical_by_sample_id.get(sample_id)
            if existing is None:
                canonical_by_sample_id[sample_id] = normalized_record
                continue
            if _is_allowed_val_test_alias_duplicate(existing, normalized_record):
                continue
            raise ValueError(f"duplicate canonical sample_id={sample_id}")
    return canonical_by_sample_id


def _repaired_train_metadata(row_metadata: dict[str, Any], canonical_record: dict[str, Any]) -> dict[str, Any]:
    raw_instruction = _canonical_raw_instruction(canonical_record)
    if raw_instruction is None:
        canonical_instruction = canonical_record.get("instruction")
        if isinstance(canonical_instruction, str) and canonical_instruction.strip():
            raw_instruction = canonical_instruction.strip()
        else:
            existing_raw_instruction = row_metadata.get("raw_instruction")
            if isinstance(existing_raw_instruction, str) and existing_raw_instruction.strip():
                raw_instruction = existing_raw_instruction.strip()

    metadata = {
        "sample_id": canonical_record.get("sample_id", row_metadata.get("sample_id")),
        "subtask": canonical_record.get("subtask", row_metadata.get("subtask")),
        "split": canonical_record.get("split", row_metadata.get("split")),
        "x0_smiles": canonical_record.get("x0_smiles", row_metadata.get("x0_smiles")),
        "reference_smiles": canonical_record.get("reference_smiles", row_metadata.get("reference_smiles")),
        "properties": canonical_record.get("properties", row_metadata.get("properties", [])),
        "raw_instruction": raw_instruction,
    }
    for key in ("teacher", "think_quality"):
        if key in row_metadata:
            metadata[key] = row_metadata[key]
    return metadata


def _normalize_user_prompt(raw_prompt: str) -> str:
    text = str(raw_prompt).strip()
    for pattern in _OLD_OUTPUT_PATTERNS:
        text = pattern.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    instruction = get_user_response_instruction(PROMPT_MODE_THINK_ANSWER)
    if instruction.lower() in text.lower():
        return text
    if text.endswith("."):
        return f"{text}\n\n{instruction}"
    return f"{text}\n\n{instruction}"


def _turn1_messages_from_canonical(record: dict[str, Any], *, fallback_prompt: str) -> list[dict[str, str]]:
    canonical_prompt = _canonical_user_prompt(record, fallback_prompt=fallback_prompt)
    return build_system_user_messages(canonical_prompt, prompt_mode=PROMPT_MODE_THINK_ANSWER)


def _normalize_assistant_content(raw_content: str) -> str:
    text = str(raw_content).strip()
    text = _THINK_PREFIX_RE.sub(r"\1", text, count=1)
    text = re.sub(r"<\s*answer\s*>", "<SMILES>", text, flags=re.IGNORECASE)
    text = re.sub(r"<\s*/\s*answer\s*>", "</SMILES>", text, flags=re.IGNORECASE)
    return text


def _assistant_validation_error(assistant_content: str) -> tuple[str, str] | None:
    think_matches = list(_THINK_RE.finditer(assistant_content))
    smiles_matches = list(_SMILES_BLOCK_RE.finditer(assistant_content))

    if len(think_matches) != 1:
        return "invalid_think_block_count", "assistant target must contain exactly one outer <think> block"

    think_text = think_matches[0].group(1)
    if _SMILES_BLOCK_RE.search(think_text):
        return "answer_inside_think", "tagged molecule output must not appear inside <think>"
    if len(smiles_matches) != 1:
        return "invalid_answer_block_count", "assistant target must contain exactly one outer tagged molecule block"

    answer_text = smiles_matches[0].group(2).strip()
    if answer_text.lower() in _PLACEHOLDER_ANSWERS:
        return "invalid_answer_smiles", "assistant answer is placeholder-like"

    validation = validate_model_output(assistant_content)
    if validation.status != "ok" or not validation.smiles:
        return "invalid_answer_smiles", validation.message or validation.status
    return None


def _align_train_row(
    row: dict[str, Any],
    *,
    canonical_by_sample_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    messages = list(row.get("messages") or [])
    row_metadata = dict(row.get("metadata") or {})
    if len(messages) < 2:
        return None, {
            "sample_id": row_metadata.get("sample_id"),
            "subtask": row_metadata.get("subtask"),
            "split": row_metadata.get("split"),
            "error_type": "invalid_message_shape",
            "error": "expected at least [user, assistant]",
        }

    sample_id = _normalize_sample_id(row_metadata.get("sample_id"))
    canonical_record = canonical_by_sample_id.get(sample_id)
    if canonical_record is None:
        raise ValueError(f"missing canonical record for sample_id={sample_id}")

    user_prompt = str(messages[0].get("content", ""))
    assistant_content = _normalize_assistant_content(str(messages[1].get("content", "")))
    err = _assistant_validation_error(assistant_content)
    if err is not None:
        error_type, error_message = err
        return None, {
            "sample_id": row_metadata.get("sample_id"),
            "subtask": row_metadata.get("subtask"),
            "split": row_metadata.get("split"),
            "error_type": error_type,
            "error": error_message,
            "raw_output_response": assistant_content,
        }

    repaired_metadata = _repaired_train_metadata(row_metadata, canonical_record)
    aligned_messages = _turn1_messages_from_canonical(canonical_record, fallback_prompt=user_prompt) + [
        {"role": "assistant", "content": assistant_content}
    ]
    return {"messages": aligned_messages, "metadata": repaired_metadata}, None


def _build_eval_prompt_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": _turn1_messages_from_canonical(record, fallback_prompt=_safe_instruction(record)),
        "metadata": {
            "sample_id": record.get("sample_id"),
            "subtask": record.get("subtask"),
            "split": record.get("split"),
            "x0_smiles": record.get("x0_smiles"),
            "reference_smiles": record.get("reference_smiles"),
            "properties": record.get("properties", []),
        },
    }


def align_sft_think_data(*, sft_think_dir: str, canonical_dir: str, output_dir: str) -> dict[str, Any]:
    canonical_by_sample_id = _load_canonical_index(canonical_dir)
    train_rows = read_jsonl(os.path.join(sft_think_dir, "sft_train.jsonl"))
    for row in train_rows[:5]:
        if _looks_aligned_train_row(row):
            raise ValueError(
                "sft_think_dir appears to contain already aligned three-message rows; "
                "expected raw sft_think [user, assistant] records instead"
            )
    aligned_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    train_by_subtask: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        aligned, failure = _align_train_row(row, canonical_by_sample_id=canonical_by_sample_id)
        if aligned is not None:
            aligned_rows.append(aligned)
            subtask = str((aligned.get("metadata") or {}).get("subtask", "")).strip()
            if subtask:
                train_by_subtask[subtask].append(aligned)
        if failure is not None:
            failures.append(failure)

    test_rows = _read_jsonl_if_exists(_canonical_file(canonical_dir, "test"))
    eval_prompt_rows = [_build_eval_prompt_row(row) for row in test_rows]
    eval_prompt_by_subtask: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eval_prompt_rows:
        subtask = str((row.get("metadata") or {}).get("subtask", "")).strip()
        if subtask:
            eval_prompt_by_subtask[subtask].append(row)

    train_jsonl = os.path.join(output_dir, "sft_train.jsonl")
    failure_jsonl = os.path.join(output_dir, "sft_train_failures.jsonl")
    eval_prompt_jsonl = os.path.join(output_dir, "eval_prompts.jsonl")
    write_jsonl(train_jsonl, aligned_rows)
    write_jsonl(failure_jsonl, failures)
    write_jsonl(eval_prompt_jsonl, eval_prompt_rows)
    train_by_subtask_paths: dict[str, str] = {}
    for subtask, rows in train_by_subtask.items():
        path = os.path.join(output_dir, "by_subtask", subtask, "sft_train.jsonl")
        write_jsonl(path, rows)
        train_by_subtask_paths[subtask] = path
    eval_prompt_by_subtask_paths: dict[str, str] = {}
    for subtask, rows in eval_prompt_by_subtask.items():
        path = os.path.join(output_dir, "by_subtask", subtask, "eval_prompts.jsonl")
        write_jsonl(path, rows)
        eval_prompt_by_subtask_paths[subtask] = path
    return {
        "train_jsonl": train_jsonl,
        "failure_jsonl": failure_jsonl,
        "eval_prompt_jsonl": eval_prompt_jsonl,
        "train_by_subtask": train_by_subtask_paths,
        "eval_prompt_by_subtask": eval_prompt_by_subtask_paths,
    }


def _print_outputs(outputs: dict[str, Any]) -> None:
    for key, value in outputs.items():
        if isinstance(value, dict):
            for nested_key, nested_value in sorted(value.items()):
                print(f"{key}[{nested_key}]: {os.path.abspath(str(nested_value))}")
            continue
        print(f"{key}: {os.path.abspath(str(value))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Align existing sft_think data to the current MARCO think_answer contract.")
    parser.add_argument("--sft-think-dir", default="data/sft_think")
    parser.add_argument("--canonical-dir", default="data/canonical")
    parser.add_argument("--output-dir", default="data/sft_think_aligned")
    args = parser.parse_args()

    outputs = align_sft_think_data(
        sft_think_dir=args.sft_think_dir,
        canonical_dir=args.canonical_dir,
        output_dir=args.output_dir,
    )
    _print_outputs(outputs)


if __name__ == "__main__":
    main()
