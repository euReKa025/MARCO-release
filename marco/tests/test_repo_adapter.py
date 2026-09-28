from pathlib import Path

from marco.data.repo_mumo_adapter import (
    canonical_record_from_dict,
    canonical_record_to_dict,
    load_repo_mumo_records,
)
from marco.utils import write_json


def test_canonical_roundtrip():
    payload = {
        "sample_id": "x:0",
        "split": "train",
        "source_repo": "RePO",
        "benchmark": "MuMoInstruction",
        "subtask": "bbbp+drd2+plogp",
        "instruction": "do something",
        "x0_smiles": "CCO",
        "reference_smiles": "CCC",
        "properties": [
            {"name": "bbbp", "direction": "increase", "delta": 0.2, "source": 0.3, "target": 0.5}
        ],
        "metadata": {"a": 1},
    }
    record = canonical_record_from_dict(payload)
    out = canonical_record_to_dict(record)

    assert out["sample_id"] == payload["sample_id"]
    assert out["properties"][0]["name"] == "bbbp"


def test_load_repo_mumo_records_keeps_raw_instruction_and_source_values(tmp_path: Path):
    repo_root = tmp_path / "repo_data"
    instruction = "Increase BBBP while keeping activity constraints."
    write_json(
        repo_root / "TRAIN_multi_prop" / "bbbp.json",
        [
            {
                "instruction": instruction,
                "output": "CCCN",
                "meta-data": {
                    "task": "bbbp",
                    "split": "train",
                    "source_smiles": "CCCO",
                    "target_smiles": "CCCN",
                    "properties": {
                        "bbbp": {
                            "source": 0.57,
                            "target": 0.93,
                            "change": 0.36,
                        }
                    },
                },
            }
        ],
    )

    records = load_repo_mumo_records(str(repo_root))

    assert len(records) == 1
    record = records[0]
    prop = record.properties[0]
    assert record.metadata["raw_instruction"] == instruction
    assert prop.source == 0.57
    assert prop.target == 0.93
    assert prop.delta == 0.36
