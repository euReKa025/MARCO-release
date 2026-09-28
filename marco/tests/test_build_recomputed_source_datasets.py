from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from marco.data.build_recomputed_source_datasets import (
    build_canonical_from_existing,
    build_recomputed_source_datasets,
)
from marco.utils import read_jsonl, write_jsonl


def _canonical_row(
    sample_id: str,
    *,
    split: str,
    subtask: str = "bbbp+drd2+plogp",
    test_partition: str | None = None,
    instr_setting: str | None = None,
    bbbp_source: float = 0.2696981430053711,
) -> dict:
    metadata = {
        "raw_instruction": f"Optimize sample {sample_id}",
        "raw_output": "CCN",
    }
    if test_partition is not None:
        metadata["test_partition"] = test_partition
    if instr_setting is not None:
        metadata["instr_setting"] = instr_setting
    return {
        "sample_id": sample_id,
        "split": split,
        "source_repo": "MARCO",
        "benchmark": "MuMoInstruction",
        "subtask": subtask,
        "instruction": f"Optimize sample {sample_id}",
        "x0_smiles": "CCO",
        "reference_smiles": "CCN",
        "properties": [
            {
                "name": "bbbp",
                "direction": "increase",
                "delta": 0.15,
                "source": bbbp_source,
                "target": 0.35,
            },
            {
                "name": "drd2",
                "direction": "increase",
                "delta": 0.20,
                "source": 0.011265689230248259,
                "target": 0.30,
            },
            {
                "name": "plogp",
                "direction": "increase",
                "delta": 0.25,
                "source": -0.3184194343415801,
                "target": 1.25,
            },
        ],
        "metadata": metadata,
    }


def _sft_train_row(sample_id: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": "stale prompt"},
            {"role": "assistant", "content": "<think>brief</think><SMILES>CCN</SMILES>"},
        ],
        "metadata": {
            "sample_id": sample_id,
            "subtask": "stale_subtask",
            "split": "train",
            "x0_smiles": "STALE",
            "reference_smiles": "STALE",
            "properties": [{"name": "old_prop", "direction": "increase"}],
        },
    }


def test_build_canonical_from_existing_splits_seen_unseen_and_aliases_val(tmp_path: Path) -> None:
    existing_root = tmp_path / "canonical_source_recomputed"
    output_root = tmp_path / "canonical_recomputed"
    write_jsonl(existing_root / "canonical_mumo_train.jsonl", [_canonical_row("train:0", split="train")])
    write_jsonl(existing_root / "canonical_mumo_val.jsonl", [])
    write_jsonl(
        existing_root / "canonical_mumo_test.jsonl",
        [
            _canonical_row("test:seen:0", split="test", instr_setting="seen"),
            _canonical_row("test:unseen:0", split="test", instr_setting="unseen"),
        ],
    )

    outputs = build_canonical_from_existing(existing_canonical_dir=str(existing_root), output_dir=str(output_root))

    val_rows = read_jsonl(outputs["all"]["val"])
    test_seen_rows = read_jsonl(outputs["all"]["test_seen"])
    test_unseen_rows = read_jsonl(outputs["all"]["test_unseen"])

    assert len(val_rows) == 1
    assert val_rows[0]["split"] == "val"
    assert val_rows[0]["metadata"]["test_partition"] == "seen"
    assert len(test_seen_rows) == 1
    assert test_seen_rows[0]["metadata"]["test_partition"] == "seen"
    assert len(test_unseen_rows) == 1
    assert test_unseen_rows[0]["metadata"]["test_partition"] == "unseen"


def test_build_recomputed_source_datasets_writes_expected_parallel_trees(tmp_path: Path) -> None:
    canonical_source_root = tmp_path / "canonical_source_recomputed"
    write_jsonl(canonical_source_root / "canonical_mumo_train.jsonl", [_canonical_row("train:0", split="train")])
    write_jsonl(canonical_source_root / "canonical_mumo_val.jsonl", [])
    write_jsonl(
        canonical_source_root / "canonical_mumo_test.jsonl",
        [
            _canonical_row("test:seen:0", split="test", instr_setting="seen"),
            _canonical_row("test:unseen:0", split="test", instr_setting="unseen"),
        ],
    )

    sft_think_dir = tmp_path / "sft_think"
    write_jsonl(sft_think_dir / "sft_train.jsonl", [_sft_train_row("train:0")])
    write_jsonl(sft_think_dir / "sft_train_failures.jsonl", [])

    outputs = build_recomputed_source_datasets(
        canonical_source_dir=str(canonical_source_root),
        sft_think_dir=str(sft_think_dir),
        canonical_output_dir=str(tmp_path / "data" / "canonical_recomputed"),
        baseline_jsonl_output_dir=str(tmp_path / "data" / "baselines_recomputed"),
        sft_aligned_output_dir=str(tmp_path / "data" / "sft_think_aligned_recomputed"),
        sft_parquet_output_dir=str(tmp_path / "data" / "sft_recomputed"),
        baseline_parquet_output_dir=str(tmp_path / "data" / "grpo_single_turn_recomputed"),
        rlhf_parquet_output_dir=str(tmp_path / "data" / "rlhf_recomputed"),
        baseline_prompt_mode="think_answer",
        max_turns=5,
    )

    subtask = "bbbp+drd2+plogp"
    assert Path(outputs["canonical"]["all"]["train"]).exists()
    assert Path(outputs["canonical"]["all"]["val"]).exists()
    assert Path(outputs["canonical"]["all"]["test"]).exists()
    assert Path(outputs["canonical"]["all"]["test_seen"]).exists()
    assert Path(outputs["canonical"]["all"]["test_unseen"]).exists()
    assert Path(outputs["baseline_parquet"]["by_subtask"][subtask]["train"]).exists()
    assert Path(outputs["baseline_parquet"]["by_subtask"][subtask]["val"]).exists()
    assert Path(outputs["baseline_parquet"]["by_subtask"][subtask]["test_seen"]).exists()
    assert Path(outputs["baseline_parquet"]["by_subtask"][subtask]["test_unseen"]).exists()
    assert Path(outputs["rlhf_parquet"]["by_subtask"][subtask]["train"]).exists()
    assert Path(outputs["rlhf_parquet"]["by_subtask"][subtask]["val"]).exists()
    assert Path(outputs["rlhf_parquet"]["by_subtask"][subtask]["test_seen"]).exists()
    assert Path(outputs["rlhf_parquet"]["by_subtask"][subtask]["test_unseen"]).exists()
    assert Path(outputs["sft_parquet"]["merged"]["train"]).exists()
    assert Path(outputs["sft_parquet"]["by_subtask"][subtask]["train"]).exists()

    baseline_train_row = pq.read_table(outputs["baseline_parquet"]["merged"]["train"]).to_pylist()[0]
    baseline_seen_row = pq.read_table(outputs["baseline_parquet"]["merged"]["val"]).to_pylist()[0]
    sft_row = pq.read_table(outputs["sft_parquet"]["merged"]["train"]).to_pylist()[0]

    assert baseline_train_row["extra_info"]["properties"][0]["source"] == 0.2696981430053711
    assert baseline_seen_row["extra_info"]["test_partition"] == "seen"
    assert sft_row["metadata"]["properties"][0]["source"] == 0.2696981430053711
