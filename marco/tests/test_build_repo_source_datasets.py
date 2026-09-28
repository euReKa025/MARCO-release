from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from marco.data.build_repo_source_datasets import _baseline_jobs, _rlhf_jobs, _sft_jobs, build_repo_source_datasets
from marco.prompts.system_prompt import DIRECT_SMILES_SYSTEM_PROMPT, PROMPT_MODE_DIRECT_SMILES
from marco.utils import write_json, write_jsonl


def _repo_row(*, split: str, instruction: str) -> dict:
    instr_setting = None
    if split == "test_seen":
        split = "test"
        instr_setting = "seen"
    elif split == "test_unseen":
        split = "test"
        instr_setting = "unseen"
    metadata = {
        "task": "bbbp+drd2+plogp",
        "split": split,
        "source_smiles": "CCO",
        "target_smiles": "CCN",
        "properties": {
            "bbbp": {"source": 0.20, "target": 0.35, "change": 0.15},
            "drd2": {"source": 0.10, "target": 0.30, "change": 0.20},
            "plogp": {"source": 1.00, "target": 1.25, "change": 0.25},
        },
    }
    if instr_setting is not None:
        metadata["instr_setting"] = instr_setting
    return {
        "instruction": instruction,
        "output": "CCN",
        "meta-data": metadata,
    }


def _baseline_jsonl_row(sample_id: str, *, raw_instruction: str) -> dict:
    return {
        "input": raw_instruction,
        "input_messages_direct_smiles": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": raw_instruction},
        ],
        "input_messages_think_answer": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": raw_instruction},
        ],
        "metadata": {
            "sample_id": sample_id,
            "subtask": "bbbp+drd2+plogp",
            "split": "train",
            "x0_smiles": "CCO",
            "reference_smiles": "CCN",
            "properties": [{"name": "bbbp", "direction": "increase"}],
            "raw_instruction": raw_instruction,
        },
    }


def _sft_train_row(sample_id: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
            {"role": "assistant", "content": "<think>brief</think><answer>CCN</answer>"},
        ],
        "metadata": {"sample_id": sample_id, "subtask": "bbbp+drd2+plogp"},
    }


def test_build_repo_source_datasets_writes_all_expected_trees(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo_data"
    train_instruction = "Train instruction from raw RePO source."
    test_instruction = "Test instruction from raw RePO source."
    write_json(
        repo_root / "TRAIN_multi_prop" / "train_rows.json",
        [_repo_row(split="train", instruction=train_instruction)],
    )
    write_json(
        repo_root / "TEST_multi_prop" / "test_rows.json",
        [
            _repo_row(split="test_seen", instruction=test_instruction),
            _repo_row(split="test_unseen", instruction=test_instruction),
        ],
    )

    sft_think_dir = tmp_path / "sft_think"
    write_jsonl(
        sft_think_dir / "sft_train.jsonl",
        [
            {
                "messages": [
                    {"role": "user", "content": "stale prompt"},
                    {"role": "assistant", "content": "<think>Increase bbbp.</think>\n<answer>CCN</answer>"},
                ],
                "metadata": {
                    "sample_id": "train_rows:0",
                    "subtask": "stale_subtask",
                    "split": "train",
                    "x0_smiles": "STALE",
                    "reference_smiles": "STALE",
                    "properties": [{"name": "old_prop", "direction": "increase"}],
                },
            }
        ],
    )
    write_jsonl(sft_think_dir / "sft_train_failures.jsonl", [])

    outputs = build_repo_source_datasets(
        repo_data_root=str(repo_root),
        sft_think_dir=str(sft_think_dir),
        canonical_output_dir=str(tmp_path / "data" / "canonical"),
        baseline_jsonl_output_dir=str(tmp_path / "data" / "baselines"),
        sft_aligned_output_dir=str(tmp_path / "data" / "sft_think_aligned"),
        sft_parquet_output_dir=str(tmp_path / "data" / "sft"),
        baseline_parquet_output_dir=str(tmp_path / "data" / "grpo_single_turn"),
        rlhf_parquet_output_dir=str(tmp_path / "data" / "rlhf"),
        baseline_prompt_mode="think_answer",
        max_turns=5,
    )

    subtask = "bbbp+drd2+plogp"

    assert Path(outputs["canonical"]["all"]["train"]).exists()
    assert Path(outputs["canonical"]["all"]["val"]).exists()
    assert Path(outputs["canonical"]["all"]["test"]).exists()
    assert Path(outputs["canonical"]["all"]["test_seen"]).exists()
    assert Path(outputs["canonical"]["all"]["test_unseen"]).exists()
    assert Path(outputs["canonical"]["by_subtask"][subtask]["train"]).exists()
    assert Path(outputs["canonical"]["by_subtask"][subtask]["val"]).exists()
    assert Path(outputs["canonical"]["by_subtask"][subtask]["test"]).exists()
    assert Path(outputs["canonical"]["by_subtask"][subtask]["test_seen"]).exists()
    assert Path(outputs["canonical"]["by_subtask"][subtask]["test_unseen"]).exists()

    assert Path(outputs["baseline_jsonl"]["train"]["grpo_merged"]).exists()
    assert Path(outputs["baseline_jsonl"]["val"]["grpo_merged"]).exists()
    assert Path(outputs["baseline_jsonl"]["test"]["grpo_merged"]).exists()
    assert Path(outputs["baseline_jsonl"]["test_seen"]["grpo_merged"]).exists()
    assert Path(outputs["baseline_jsonl"]["test_unseen"]["grpo_merged"]).exists()

    assert Path(outputs["baseline_parquet"]["merged"]["train"]).exists()
    assert Path(outputs["baseline_parquet"]["merged"]["val"]).exists()
    assert Path(outputs["baseline_parquet"]["merged"]["test"]).exists()
    assert Path(outputs["baseline_parquet"]["merged"]["test_seen"]).exists()
    assert Path(outputs["baseline_parquet"]["merged"]["test_unseen"]).exists()
    assert Path(outputs["baseline_parquet"]["by_subtask"][subtask]["train"]).exists()
    assert Path(outputs["baseline_parquet"]["by_subtask"][subtask]["val"]).exists()
    assert Path(outputs["baseline_parquet"]["by_subtask"][subtask]["test"]).exists()
    assert Path(outputs["baseline_parquet"]["by_subtask"][subtask]["test_seen"]).exists()
    assert Path(outputs["baseline_parquet"]["by_subtask"][subtask]["test_unseen"]).exists()

    assert Path(outputs["rlhf_parquet"]["merged"]["train"]).exists()
    assert Path(outputs["rlhf_parquet"]["merged"]["val"]).exists()
    assert Path(outputs["rlhf_parquet"]["merged"]["test"]).exists()
    assert Path(outputs["rlhf_parquet"]["merged"]["test_seen"]).exists()
    assert Path(outputs["rlhf_parquet"]["merged"]["test_unseen"]).exists()
    assert Path(outputs["rlhf_parquet"]["by_subtask"][subtask]["train"]).exists()
    assert Path(outputs["rlhf_parquet"]["by_subtask"][subtask]["val"]).exists()
    assert Path(outputs["rlhf_parquet"]["by_subtask"][subtask]["test"]).exists()
    assert Path(outputs["rlhf_parquet"]["by_subtask"][subtask]["test_seen"]).exists()
    assert Path(outputs["rlhf_parquet"]["by_subtask"][subtask]["test_unseen"]).exists()

    assert Path(outputs["sft_parquet"]["merged"]["train"]).exists()
    assert Path(outputs["sft_parquet"]["by_subtask"][subtask]["train"]).exists()

    sft_row = pq.read_table(outputs["sft_parquet"]["merged"]["train"]).to_pylist()[0]
    assert sft_row["metadata"]["raw_instruction"] == train_instruction

    baseline_row = pq.read_table(outputs["baseline_parquet"]["merged"]["train"]).to_pylist()[0]
    assert baseline_row["extra_info"]["sample_id"] == "train_rows:0"
    assert baseline_row["extra_info"]["raw_instruction"] == train_instruction
    baseline_seen_row = pq.read_table(outputs["baseline_parquet"]["merged"]["val"]).to_pylist()[0]
    baseline_unseen_row = pq.read_table(outputs["baseline_parquet"]["merged"]["test_unseen"]).to_pylist()[0]
    assert baseline_seen_row["extra_info"]["test_partition"] == "seen"
    assert baseline_unseen_row["extra_info"]["test_partition"] == "unseen"

    rlhf_row = pq.read_table(outputs["rlhf_parquet"]["merged"]["train"]).to_pylist()[0]
    assert rlhf_row["extra_info"]["metadata"]["raw_instruction"] == train_instruction


def test_baseline_jobs_skips_missing_subtask_split_files(tmp_path: Path) -> None:
    baseline_root = tmp_path / "baselines"
    parquet_root = tmp_path / "grpo_single_turn"
    raw_instruction = "Instruction that must be preserved."
    row = _baseline_jsonl_row("s-merged", raw_instruction=raw_instruction)
    for split in ("train", "val", "test"):
        write_jsonl(baseline_root / "grpo_single_turn" / f"grpo_{split}.jsonl", [row])

    write_jsonl(
        baseline_root / "grpo_single_turn" / "by_subtask" / "bbbp+drd2+plogp" / "grpo_train.jsonl",
        [_baseline_jsonl_row("s-subtask", raw_instruction=raw_instruction)],
    )

    outputs = _baseline_jobs(baseline_root, parquet_root, prompt_mode="think_answer")

    assert set(outputs["merged"]) == {"train", "val", "test"}
    assert Path(outputs["by_subtask"]["bbbp+drd2+plogp"]["train"]).exists()
    assert set(outputs["by_subtask"]["bbbp+drd2+plogp"]) == {"train"}


def test_rlhf_jobs_threads_prompt_mode_to_parquet_rows(tmp_path: Path) -> None:
    canonical_root = tmp_path / "canonical"
    parquet_root = tmp_path / "rlhf"
    raw_instruction = "Instruction that must become the RL user prompt."
    write_jsonl(
        canonical_root / "canonical_mumo_train.jsonl",
        [
            {
                "sample_id": "s-direct",
                "subtask": "bbbp+drd2+plogp",
                "instruction": raw_instruction,
                "reference_smiles": "CCN",
                "x0_smiles": "CCO",
                "properties": [{"name": "bbbp", "direction": "increase"}],
                "metadata": {"split": "train", "raw_instruction": raw_instruction},
            }
        ],
    )

    outputs = _rlhf_jobs(canonical_root, parquet_root, max_turns=5, prompt_mode=PROMPT_MODE_DIRECT_SMILES)

    row = pq.read_table(outputs["merged"]["train"]).to_pylist()[0]
    assert row["prompt"][0]["content"] == DIRECT_SMILES_SYSTEM_PROMPT
    assert row["prompt"][1]["content"] == raw_instruction
    assert row["extra_info"]["prompt_mode"] == PROMPT_MODE_DIRECT_SMILES
    assert row["extra_info"]["interaction_kwargs"]["prompt_mode"] == PROMPT_MODE_DIRECT_SMILES


def test_sft_jobs_skips_subtasks_without_sft_train_jsonl(tmp_path: Path) -> None:
    aligned_root = tmp_path / "sft_think_aligned"
    parquet_root = tmp_path / "sft"

    write_jsonl(aligned_root / "sft_train.jsonl", [_sft_train_row("merged-0")])
    write_jsonl(
        aligned_root / "by_subtask" / "bbbp+drd2+plogp" / "sft_train.jsonl",
        [_sft_train_row("subtask-0")],
    )
    write_jsonl(
        aligned_root / "by_subtask" / "bbbp+drd2+qed" / "eval_prompts.jsonl",
        [{"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], "metadata": {}}],
    )

    outputs = _sft_jobs(aligned_root, parquet_root)

    assert Path(outputs["merged"]["train"]).exists()
    assert "bbbp+drd2+plogp" in outputs["by_subtask"]
    assert "bbbp+drd2+qed" not in outputs["by_subtask"]
