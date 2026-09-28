from __future__ import annotations

from typing import Any

from marco.prompts.system_prompt import PROMPT_MODE_THINK_ANSWER, build_system_user_messages, normalize_prompt_mode


def dataset_instruction_from_record(record: dict[str, Any]) -> str:
    instruction = record.get("instruction")
    if isinstance(instruction, str) and instruction.strip():
        return instruction.strip()

    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        raw_instruction = metadata.get("raw_instruction")
        if isinstance(raw_instruction, str) and raw_instruction.strip():
            return raw_instruction.strip()

    sample_id = record.get("sample_id", "<unknown>")
    raise ValueError(f"record has missing non-empty instruction field; sample_id={sample_id}")


def canonical_record_to_verl_row(
    record: dict[str, Any],
    *,
    max_turns: int = 5,
    prompt_mode: str = PROMPT_MODE_THINK_ANSWER,
) -> dict[str, Any]:
    mode = normalize_prompt_mode(prompt_mode)
    properties = list(record.get("properties") or [])
    x0_smiles = str(record.get("x0_smiles", ""))
    sample_id = record.get("sample_id")
    reference_smiles = str(record.get("reference_smiles") or "")
    instruction = dataset_instruction_from_record(record)
    canonical_record = dict(record)
    canonical_record["instruction"] = instruction
    prompt = build_system_user_messages(instruction, prompt_mode=mode)

    return {
        "prompt": prompt,
        "ground_truth": reference_smiles,
        "reward_model": {"ground_truth": reference_smiles},
        "data_source": "marco_mumo",
        "agent_name": "marco_tool_agent",
        "extra_info": {
            "index": sample_id,
            "sample_id": sample_id,
            "subtask": record.get("subtask"),
            "x0_smiles": x0_smiles,
            "property_targets": properties,
            "max_turns": int(record.get("max_turns") or max_turns),
            "metadata": dict(record.get("metadata") or {}),
            "instruction": instruction,
            "prompt_mode": mode,
            "canonical_record": canonical_record,
            "interaction_kwargs": {
                "name": "marco",
                "record": canonical_record,
                "max_turns": int(record.get("max_turns") or max_turns),
                "prompt_mode": mode,
            },
        },
    }
