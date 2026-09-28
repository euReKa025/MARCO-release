from __future__ import annotations

from pathlib import Path

from marco.data.build_canonical_mumo_jsonl import build_canonical
from marco.utils import read_jsonl, write_json


def _repo_row(split: str, *, instr_setting: str | None = None) -> dict:
    metadata = {
        "task": "bbbp+drd2+plogp",
        "split": split,
        "source_smiles": "CCCO",
        "target_smiles": "CCCN",
        "properties": {
            "bbbp": {"source": 0.20, "target": 0.40, "change": 0.20},
            "drd2": {"source": 0.10, "target": 0.30, "change": 0.20},
            "plogp": {"source": 1.10, "target": 1.40, "change": 0.30},
        },
    }
    if instr_setting is not None:
        metadata["instr_setting"] = instr_setting
    return {
        "instruction": f"optimize for split={split}",
        "output": "CCCN",
        "meta-data": metadata,
    }


def test_build_canonical_preserves_seen_unseen_test_partitions_and_uses_seen_for_val(tmp_path: Path):
    repo_root = tmp_path / "repo_data"
    write_json(repo_root / "TRAIN_multi_prop" / "train_rows.json", [_repo_row("train")])
    write_json(
        repo_root / "TEST_multi_prop" / "seen_rows.json",
        [_repo_row("test", instr_setting="seen")],
    )
    write_json(
        repo_root / "TEST_multi_prop" / "unseen_rows.json",
        [_repo_row("test", instr_setting="unseen")],
    )

    outputs = build_canonical(repo_data_root=str(repo_root), output_dir=str(tmp_path / "canonical"))

    all_val = read_jsonl(outputs["all"]["val"])
    all_test = read_jsonl(outputs["all"]["test"])
    all_test_seen = read_jsonl(outputs["all"]["test_seen"])
    all_test_unseen = read_jsonl(outputs["all"]["test_unseen"])
    assert len(all_val) == 1
    assert len(all_test) == 2
    assert len(all_test_seen) == 1
    assert len(all_test_unseen) == 1
    assert all_val[0]["split"] == "val"
    assert all_val[0]["metadata"]["instr_setting"] == "seen"
    assert all_val[0]["metadata"]["test_partition"] == "seen"
    assert all_test_seen[0]["metadata"]["test_partition"] == "seen"
    assert all_test_unseen[0]["metadata"]["test_partition"] == "unseen"
    assert {row["metadata"]["test_partition"] for row in all_test} == {"seen", "unseen"}

    subtask_val = read_jsonl(outputs["by_subtask"]["bbbp+drd2+plogp"]["val"])
    subtask_test_seen = read_jsonl(outputs["by_subtask"]["bbbp+drd2+plogp"]["test_seen"])
    subtask_test_unseen = read_jsonl(outputs["by_subtask"]["bbbp+drd2+plogp"]["test_unseen"])
    assert len(subtask_val) == 1
    assert subtask_val[0]["split"] == "val"
    assert subtask_val[0]["metadata"]["test_partition"] == "seen"
    assert len(subtask_test_seen) == 1
    assert len(subtask_test_unseen) == 1
