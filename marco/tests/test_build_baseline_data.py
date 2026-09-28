from __future__ import annotations

from pathlib import Path

from marco.data.build_baseline_data import build_baseline_data
from marco.utils import read_jsonl, write_jsonl


def _row(sample_id: str, split: str, *, answer: str | None) -> dict:
    return {
        "sample_id": sample_id,
        "split": split,
        "subtask": "bbbp+drd2+qed",
        "instruction": f"instruction-{sample_id}",
        "x0_smiles": "CCO",
        "reference_smiles": answer,
        "properties": [
            {"name": "bbbp", "direction": "increase", "source": 0.2},
            {"name": "drd2", "direction": "increase", "source": 0.1},
            {"name": "qed", "direction": "increase", "source": 0.3},
        ],
        "metadata": {},
    }


def test_build_baseline_data_outputs(tmp_path: Path):
    canonical_dir = tmp_path / "canonical"
    write_jsonl(canonical_dir / "canonical_mumo_train.jsonl", [_row("t1", "train", answer="CCC"), _row("t2", "train", answer=None)])
    seen_row = _row("v1", "val", answer="CCN")
    seen_row["metadata"] = {"test_partition": "seen", "instr_setting": "seen"}
    unseen_row = _row("e1", "test", answer="CCCl")
    unseen_row["metadata"] = {"test_partition": "unseen", "instr_setting": "unseen"}
    write_jsonl(canonical_dir / "canonical_mumo_val.jsonl", [seen_row])
    write_jsonl(canonical_dir / "canonical_mumo_test.jsonl", [seen_row, unseen_row])
    write_jsonl(canonical_dir / "canonical_mumo_test_seen.jsonl", [seen_row])
    write_jsonl(canonical_dir / "canonical_mumo_test_unseen.jsonl", [unseen_row])

    output_dir = tmp_path / "baselines"
    outputs = build_baseline_data(str(canonical_dir), str(output_dir))

    sft_train = read_jsonl(outputs["train"]["sft_merged"])
    grpo_train = read_jsonl(outputs["train"]["grpo_merged"])
    assert len(sft_train) == 1  # one row is skipped due to missing reference_smiles
    assert len(grpo_train) == 2

    assert sft_train[0]["messages"][0]["role"] == "system"
    assert sft_train[0]["messages"][1]["role"] == "user"
    assert sft_train[0]["messages"][2]["role"] == "assistant"
    assert sft_train[0]["messages"][2]["content"] == "<SMILES>CCC</SMILES>"
    assert "metadata" in grpo_train[0]
    assert "x0_smiles" in grpo_train[0]["metadata"]
    assert grpo_train[0]["input_messages_direct_smiles"][0]["role"] == "system"
    assert grpo_train[0]["input_messages_direct_smiles"][1]["role"] == "user"
    assert grpo_train[0]["input_messages_think_answer"][0]["role"] == "system"
    assert grpo_train[0]["input_messages_think_answer"][1]["role"] == "user"

    assert "test_seen" in outputs
    assert "test_unseen" in outputs
    grpo_val = read_jsonl(outputs["val"]["grpo_merged"])
    grpo_test_seen = read_jsonl(outputs["test_seen"]["grpo_merged"])
    grpo_test_unseen = read_jsonl(outputs["test_unseen"]["grpo_merged"])
    assert grpo_val[0]["metadata"]["test_partition"] == "seen"
    assert grpo_test_seen[0]["metadata"]["test_partition"] == "seen"
    assert grpo_test_unseen[0]["metadata"]["test_partition"] == "unseen"
