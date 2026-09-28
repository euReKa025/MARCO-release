from __future__ import annotations

from pathlib import Path

from marco.data.recompute_source_properties import build_recomputed_dataset
from marco.utils import read_jsonl, write_jsonl


class _DummyPredictor:
    def predict(self, smiles: str, required_properties):
        required = list(required_properties)
        if smiles == "CCO":
            base = {"bbbp": 0.11, "drd2": 0.22, "plogp": -0.33}
        else:
            base = {"bbbp": 0.41, "drd2": 0.52, "plogp": -1.23}
        return {k: base[k] for k in required}


def _row(x0_smiles: str) -> dict:
    return {
        "input": "optimize this",
        "metadata": {
            "x0_smiles": x0_smiles,
            "properties": [
                {"name": "bbbp", "direction": "increase", "source": 0.0},
                {"name": "drd2", "direction": "increase", "source": 0.0},
                {"name": "plogp", "direction": "increase", "source": 0.0},
            ],
        },
    }


def test_build_recomputed_dataset_mirrors_tree_and_updates_sources(tmp_path: Path):
    input_root = tmp_path / "in_data"
    output_root = tmp_path / "out_data"

    write_jsonl(input_root / "grpo_train.jsonl", [_row("CCO")])
    write_jsonl(input_root / "by_subtask" / "bbbp+drd2+plogp" / "grpo_test.jsonl", [_row("CCN")])

    stats = build_recomputed_dataset(
        input_root=input_root,
        output_root=output_root,
        predictor=_DummyPredictor(),
    )

    assert stats["files_processed"] == 2
    assert stats["rows_processed"] == 2
    assert stats["rows_updated"] == 2
    assert stats["properties_updated"] == 6

    out_train = read_jsonl(output_root / "grpo_train.jsonl")
    out_test = read_jsonl(output_root / "by_subtask" / "bbbp+drd2+plogp" / "grpo_test.jsonl")
    assert out_train[0]["metadata"]["properties"][0]["source"] == 0.11
    assert out_train[0]["metadata"]["properties"][1]["source"] == 0.22
    assert out_train[0]["metadata"]["properties"][2]["source"] == -0.33
    assert out_test[0]["metadata"]["properties"][0]["source"] == 0.41
    assert out_test[0]["metadata"]["properties"][1]["source"] == 0.52
    assert out_test[0]["metadata"]["properties"][2]["source"] == -1.23

    # Input data must remain unchanged.
    original_train = read_jsonl(input_root / "grpo_train.jsonl")
    assert original_train[0]["metadata"]["properties"][0]["source"] == 0.0
