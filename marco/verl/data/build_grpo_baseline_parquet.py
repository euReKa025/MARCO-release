from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from marco.prompts.system_prompt import PROMPT_MODE_DIRECT_SMILES, PROMPT_MODE_THINK_ANSWER
from marco.utils import read_jsonl


PROMPT_MODE_TO_KEY = {
    PROMPT_MODE_DIRECT_SMILES: "input_messages_direct_smiles",
    PROMPT_MODE_THINK_ANSWER: "input_messages_think_answer",
}


def _validate_row(record: dict[str, Any], *, prompt_mode: str, index: int) -> dict[str, Any]:
    prompt_key = PROMPT_MODE_TO_KEY[prompt_mode]
    prompt = record.get(prompt_key)
    if not isinstance(prompt, list) or not prompt:
        raise ValueError(
            f"row[{index}] missing required prompt messages in '{prompt_key}' for prompt_mode={prompt_mode}"
        )

    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"row[{index}] metadata must be a dict")

    x0_smiles = metadata.get("x0_smiles")
    if not isinstance(x0_smiles, str) or not x0_smiles.strip():
        raise ValueError(f"row[{index}] metadata.x0_smiles must be a non-empty string")

    properties = metadata.get("properties")
    if not isinstance(properties, list) or not properties:
        raise ValueError(f"row[{index}] metadata.properties must be a non-empty list")
    valid_properties: list[dict[str, Any]] = []
    for item in properties:
        if isinstance(item, dict) and isinstance(item.get("name"), str) and item.get("name").strip():
            valid_properties.append(item)
    if not valid_properties:
        raise ValueError(
            f"row[{index}] metadata.properties must contain at least one dict with non-empty 'name'"
        )

    sample_id = metadata.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        sample_id = f"row-{index}"
    else:
        sample_id = sample_id.strip()

    extra_info = dict(metadata)
    extra_info["index"] = int(index)
    extra_info["sample_id"] = sample_id
    extra_info["prompt_mode"] = prompt_mode

    reward_model = {
        "ground_truth": {
            "x0_smiles": x0_smiles.strip(),
            "reference_smiles": metadata.get("reference_smiles"),
            "properties": valid_properties,
        }
    }

    return {
        "prompt": prompt,
        "reward_model": reward_model,
        "data_source": "repo_single_turn_baseline",
        "extra_info": extra_info,
    }


def build_grpo_baseline_parquet(
    *,
    input_jsonl: str,
    output_parquet: str,
    prompt_mode: str = PROMPT_MODE_THINK_ANSWER,
) -> None:
    if prompt_mode not in PROMPT_MODE_TO_KEY:
        allowed = ", ".join(sorted(PROMPT_MODE_TO_KEY))
        raise ValueError(f"unsupported prompt_mode '{prompt_mode}', expected one of: {allowed}")

    rows = read_jsonl(input_jsonl)
    if not rows:
        raise ValueError("input JSONL is empty")

    converted = [
        _validate_row(record, prompt_mode=prompt_mode, index=index)
        for index, record in enumerate(rows)
    ]
    table = pa.Table.from_pylist(converted)

    output_path = Path(output_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert baseline single-turn GRPO JSONL to verl parquet format."
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-parquet", required=True)
    parser.add_argument("--prompt-mode", default=PROMPT_MODE_THINK_ANSWER)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    build_grpo_baseline_parquet(
        input_jsonl=args.input_jsonl,
        output_parquet=args.output_parquet,
        prompt_mode=args.prompt_mode,
    )


if __name__ == "__main__":
    main()
