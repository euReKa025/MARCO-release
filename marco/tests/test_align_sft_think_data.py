from __future__ import annotations

from pathlib import Path

import pytest

from marco.utils import read_jsonl, write_jsonl


def _sft_row(user_text: str, assistant_text: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
        "metadata": {
            "sample_id": "s1",
            "subtask": "bbbp+drd2+plogp",
            "split": "train",
            "x0_smiles": "CCO",
            "reference_smiles": "CCN",
            "properties": [
                {"name": "bbbp", "direction": "increase"},
                {"name": "drd2", "direction": "increase"},
                {"name": "plogp", "direction": "increase"},
            ],
        },
    }


def _canonical_row(
    sample_id: str,
    *,
    subtask: str = "bbbp+drd2+plogp",
    split: str = "train",
    x0_smiles: str = "CCO",
    reference_smiles: str = "CCN",
    properties: list[dict] | None = None,
) -> dict:
    if properties is None:
        properties = [
            {"name": "bbbp", "direction": "increase"},
            {"name": "drd2", "direction": "increase"},
            {"name": "plogp", "direction": "increase"},
        ]
    return {
        "sample_id": sample_id,
        "subtask": subtask,
        "split": split,
        "instruction": f"Dataset instruction for {str(sample_id).strip()}.",
        "x0_smiles": x0_smiles,
        "reference_smiles": reference_smiles,
        "properties": properties,
    }


def test_align_sft_think_data_inserts_system_and_think_answer_prompt(tmp_path: Path):
    from marco.data.align_sft_think_data import align_sft_think_data

    src = tmp_path / "src"
    out = tmp_path / "out"
    write_jsonl(
        src / "sft_train.jsonl",
        [
            _sft_row(
                "Alter the given molecule to meet the desired property changes with the least structural alteration possible. "
                "Output only the adjusted molecule in SMILES format, using <SMILES> </SMILES> tags.",
                "<think>Increase bbbp. Increase drd2. Increase plogp.</think>\n<answer>CCN</answer>",
            )
        ],
    )
    write_jsonl(src / "sft_train_failures.jsonl", [])
    write_jsonl(tmp_path / "canonical" / "canonical_mumo_train.jsonl", [_canonical_row("s1")])
    write_jsonl(tmp_path / "canonical" / "canonical_mumo_val.jsonl", [])
    write_jsonl(tmp_path / "canonical" / "canonical_mumo_test.jsonl", [])

    outputs = align_sft_think_data(
        sft_think_dir=str(src),
        canonical_dir=str(tmp_path / "canonical"),
        output_dir=str(out),
    )

    rows = read_jsonl(outputs["train_jsonl"])
    assert [m["role"] for m in rows[0]["messages"]] == ["system", "user", "assistant"]
    user_text = rows[0]["messages"][1]["content"]
    assert user_text == "Dataset instruction for s1."
    assert "Turn: 1" not in user_text
    assert "Source molecule (x0): CCO" not in user_text
    assert "Similarity target:" not in user_text
    assert "using <SMILES> </SMILES> tags" not in user_text


def test_align_sft_think_data_removes_other_legacy_smiles_output_contracts(tmp_path: Path):
    from marco.data.align_sft_think_data import align_sft_think_data

    src = tmp_path / "src"
    out = tmp_path / "out"
    write_jsonl(
        src / "sft_train.jsonl",
        [
            _sft_row(
                "Adjust the structure of the given molecule to target the specified adjustments in molecular properties. "
                "Retain the core structure as much as possible. Respond with only the SMILES of the modified molecule "
                "enclosed in <SMILES> </SMILES> tags. Input : <SMILES> CCO </SMILES>",
                "<think>Increase bbbp. Increase drd2. Increase plogp.</think>\n<answer>CCN</answer>",
            )
        ],
    )
    write_jsonl(src / "sft_train_failures.jsonl", [])
    write_jsonl(tmp_path / "canonical" / "canonical_mumo_train.jsonl", [_canonical_row("s1")])
    write_jsonl(tmp_path / "canonical" / "canonical_mumo_val.jsonl", [])
    write_jsonl(tmp_path / "canonical" / "canonical_mumo_test.jsonl", [])

    outputs = align_sft_think_data(
        sft_think_dir=str(src),
        canonical_dir=str(tmp_path / "canonical"),
        output_dir=str(out),
    )

    rows = read_jsonl(outputs["train_jsonl"])
    user_text = rows[0]["messages"][1]["content"]
    assert user_text == "Dataset instruction for s1."
    assert "enclosed in <SMILES> </SMILES> tags" not in user_text
    assert "Keep the <think> block to one short sentence." not in user_text


def test_align_sft_think_data_cleans_leading_think_prefix_inside_assistant(tmp_path: Path):
    from marco.data.align_sft_think_data import align_sft_think_data

    src = tmp_path / "src"
    out = tmp_path / "out"
    write_jsonl(
        src / "sft_train.jsonl",
        [
            _sft_row(
                "prompt",
                "<think>think Increase bbbp. Increase drd2. Increase plogp.</think>\n<answer>CCN</answer>",
            )
        ],
    )
    write_jsonl(src / "sft_train_failures.jsonl", [])
    write_jsonl(tmp_path / "canonical" / "canonical_mumo_train.jsonl", [_canonical_row("s1")])
    write_jsonl(tmp_path / "canonical" / "canonical_mumo_val.jsonl", [])
    write_jsonl(tmp_path / "canonical" / "canonical_mumo_test.jsonl", [])

    outputs = align_sft_think_data(
        sft_think_dir=str(src),
        canonical_dir=str(tmp_path / "canonical"),
        output_dir=str(out),
    )

    rows = read_jsonl(outputs["train_jsonl"])
    assert rows[0]["messages"][2]["content"].startswith("<think>Increase bbbp.")


def test_align_sft_think_data_filters_nested_answer_and_invalid_smiles(tmp_path: Path):
    from marco.data.align_sft_think_data import align_sft_think_data

    src = tmp_path / "src"
    out = tmp_path / "out"
    write_jsonl(
        src / "sft_train.jsonl",
        [
            _sft_row("prompt", "<think>bad <answer>CCN</answer></think>\n<answer>CCN</answer>"),
            _sft_row("prompt", "<think>ok think. second. third.</think>\n<answer>SMILES</answer>"),
        ],
    )
    write_jsonl(src / "sft_train_failures.jsonl", [])
    write_jsonl(tmp_path / "canonical" / "canonical_mumo_train.jsonl", [_canonical_row("s1")])
    write_jsonl(tmp_path / "canonical" / "canonical_mumo_val.jsonl", [])
    write_jsonl(tmp_path / "canonical" / "canonical_mumo_test.jsonl", [])

    outputs = align_sft_think_data(
        sft_think_dir=str(src),
        canonical_dir=str(tmp_path / "canonical"),
        output_dir=str(out),
    )

    assert read_jsonl(outputs["train_jsonl"]) == []
    failures = read_jsonl(outputs["failure_jsonl"])
    assert {row["error_type"] for row in failures} == {"answer_inside_think", "invalid_answer_smiles"}


def test_align_sft_think_data_builds_behavior_only_eval_prompts(tmp_path: Path):
    from marco.data.align_sft_think_data import align_sft_think_data

    src = tmp_path / "src"
    out = tmp_path / "out"
    write_jsonl(src / "sft_train.jsonl", [])
    write_jsonl(src / "sft_train_failures.jsonl", [])
    write_jsonl(
        tmp_path / "canonical" / "canonical_mumo_test.jsonl",
        [
            {
                "sample_id": "t1",
                "split": "test",
                "subtask": "bbbp+drd2+plogp",
                "instruction": "Modify the molecule. Output only <SMILES>.",
                "x0_smiles": "CCO",
                "reference_smiles": "CCN",
                "properties": [{"name": "bbbp", "direction": "increase"}],
            }
        ],
    )

    outputs = align_sft_think_data(
        sft_think_dir=str(src),
        canonical_dir=str(tmp_path / "canonical"),
        output_dir=str(out),
    )

    eval_rows = read_jsonl(outputs["eval_prompt_jsonl"])
    assert [m["role"] for m in eval_rows[0]["messages"]] == ["system", "user"]
    eval_user_text = eval_rows[0]["messages"][1]["content"]
    assert eval_user_text == "Modify the molecule. Output only <SMILES>."
    assert "Turn: 1" not in eval_user_text
    assert "Source molecule (x0): CCO" not in eval_user_text
    assert "Similarity target:" not in eval_user_text
    assert "metadata" in eval_rows[0]


def test_align_sft_think_data_writes_by_subtask_train_and_eval(tmp_path: Path):
    from marco.data.align_sft_think_data import align_sft_think_data

    src = tmp_path / "src"
    out = tmp_path / "out"
    row_two = _sft_row(
        "prompt two",
        "<think>alpha. beta. gamma.</think>\n<answer>CCC</answer>",
    )
    row_two["metadata"] = {
        **row_two["metadata"],
        "sample_id": "s2",
        "subtask": "bbbp+plogp+qed",
    }
    write_jsonl(
        src / "sft_train.jsonl",
        [
            _sft_row(
                "prompt one",
                "<think>one. two. three.</think>\n<answer>CCN</answer>",
            ),
            row_two,
        ],
    )
    write_jsonl(src / "sft_train_failures.jsonl", [])
    write_jsonl(
        tmp_path / "canonical" / "canonical_mumo_train.jsonl",
        [
            _canonical_row("s1", subtask="bbbp+drd2+plogp"),
            _canonical_row("s2", subtask="bbbp+plogp+qed", x0_smiles="CCC", reference_smiles="CCC"),
        ],
    )
    write_jsonl(tmp_path / "canonical" / "canonical_mumo_val.jsonl", [])
    write_jsonl(
        tmp_path / "canonical" / "canonical_mumo_test.jsonl",
        [
            {
                "sample_id": "t1",
                "split": "test",
                "subtask": "bbbp+drd2+plogp",
                "instruction": "Modify one.",
                "x0_smiles": "CCO",
                "reference_smiles": "CCN",
                "properties": [{"name": "bbbp", "direction": "increase"}],
            },
            {
                "sample_id": "t2",
                "split": "test",
                "subtask": "bbbp+plogp+qed",
                "instruction": "Modify two.",
                "x0_smiles": "CCC",
                "reference_smiles": "CCCl",
                "properties": [{"name": "qed", "direction": "increase"}],
            },
        ],
    )

    outputs = align_sft_think_data(
        sft_think_dir=str(src),
        canonical_dir=str(tmp_path / "canonical"),
        output_dir=str(out),
    )

    assert sorted(outputs["train_by_subtask"]) == ["bbbp+drd2+plogp", "bbbp+plogp+qed"]
    assert len(read_jsonl(outputs["train_by_subtask"]["bbbp+drd2+plogp"])) == 1
    assert len(read_jsonl(outputs["train_by_subtask"]["bbbp+plogp+qed"])) == 1

    assert sorted(outputs["eval_prompt_by_subtask"]) == ["bbbp+drd2+plogp", "bbbp+plogp+qed"]
    assert len(read_jsonl(outputs["eval_prompt_by_subtask"]["bbbp+drd2+plogp"])) == 1
    assert len(read_jsonl(outputs["eval_prompt_by_subtask"]["bbbp+plogp+qed"])) == 1


def test_align_sft_think_data_overwrites_metadata_from_canonical_by_sample_id(tmp_path: Path):
    from marco.data.align_sft_think_data import align_sft_think_data

    src = tmp_path / "src"
    out = tmp_path / "out"
    row = _sft_row(
        "stale prompt that should be replaced",
        "<think>Increase bbbp. Increase drd2. Increase plogp.</think>\n<answer>CCN</answer>",
    )
    row["metadata"] = {
        **row["metadata"],
        "sample_id": "canonical:0",
        "subtask": "stale+subtask",
        "split": "train",
        "x0_smiles": "STALEx0",
        "reference_smiles": "STALEref",
        "properties": [{"name": "old_prop", "direction": "decrease"}],
        "teacher": "teacher-a",
        "think_quality": "strong",
    }
    write_jsonl(src / "sft_train.jsonl", [row])
    write_jsonl(src / "sft_train_failures.jsonl", [])
    write_jsonl(
        tmp_path / "canonical" / "canonical_mumo_train.jsonl",
        [
            {
                "sample_id": "canonical:0",
                "split": "train",
                "subtask": "bbbp+drd2+plogp",
                "instruction": "Fallback instruction should not be used.",
                "x0_smiles": "CCO",
                "reference_smiles": "CCN",
                "properties": [{"name": "bbbp", "direction": "increase"}],
                "metadata": {"raw_instruction": "Canonical raw instruction from metadata."},
            }
        ],
    )
    write_jsonl(tmp_path / "canonical" / "canonical_mumo_val.jsonl", [])
    write_jsonl(tmp_path / "canonical" / "canonical_mumo_test.jsonl", [])

    outputs = align_sft_think_data(
        sft_think_dir=str(src),
        canonical_dir=str(tmp_path / "canonical"),
        output_dir=str(out),
    )

    rows = read_jsonl(outputs["train_jsonl"])
    metadata = rows[0]["metadata"]
    assert metadata["sample_id"] == "canonical:0"
    assert metadata["subtask"] == "bbbp+drd2+plogp"
    assert metadata["split"] == "train"
    assert metadata["x0_smiles"] == "CCO"
    assert metadata["reference_smiles"] == "CCN"
    assert metadata["properties"] == [{"name": "bbbp", "direction": "increase"}]
    assert metadata["raw_instruction"] == "Canonical raw instruction from metadata."
    assert metadata["teacher"] == "teacher-a"
    assert metadata["think_quality"] == "strong"
    user_text = rows[0]["messages"][1]["content"]
    assert user_text == "Canonical raw instruction from metadata."
    assert "Turn: 1" not in user_text
    assert "Source molecule (x0): CCO" not in user_text
    assert "stale prompt that should be replaced" not in rows[0]["messages"][1]["content"]


def test_align_sft_think_data_raises_on_missing_canonical_sample_id(tmp_path: Path):
    from marco.data.align_sft_think_data import align_sft_think_data

    src = tmp_path / "src"
    out = tmp_path / "out"
    row = _sft_row(
        "prompt",
        "<think>bad <answer>CCN</answer></think>\n<answer>SMILES</answer>",
    )
    row["metadata"] = {**row["metadata"], "sample_id": "missing:0"}
    write_jsonl(src / "sft_train.jsonl", [row])
    write_jsonl(src / "sft_train_failures.jsonl", [])
    write_jsonl(tmp_path / "canonical" / "canonical_mumo_train.jsonl", [])
    write_jsonl(tmp_path / "canonical" / "canonical_mumo_val.jsonl", [])
    write_jsonl(tmp_path / "canonical" / "canonical_mumo_test.jsonl", [])

    with pytest.raises(ValueError, match="missing canonical record for sample_id=missing:0"):
        align_sft_think_data(
            sft_think_dir=str(src),
            canonical_dir=str(tmp_path / "canonical"),
            output_dir=str(out),
        )


def test_align_sft_think_data_rejects_already_aligned_three_message_rows(tmp_path: Path):
    from marco.data.align_sft_think_data import align_sft_think_data

    src = tmp_path / "src"
    out = tmp_path / "out"
    write_jsonl(
        src / "sft_train.jsonl",
        [
            {
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "prompt"},
                    {"role": "assistant", "content": "<think>Increase bbbp. Increase drd2. Increase plogp.</think>\n<answer>CCN</answer>"},
                ],
                "metadata": {
                    "sample_id": "s1",
                    "subtask": "bbbp+drd2+plogp",
                    "split": "train",
                },
            }
        ],
    )
    write_jsonl(src / "sft_train_failures.jsonl", [])
    write_jsonl(tmp_path / "canonical" / "canonical_mumo_train.jsonl", [_canonical_row("s1")])
    write_jsonl(tmp_path / "canonical" / "canonical_mumo_val.jsonl", [])
    write_jsonl(tmp_path / "canonical" / "canonical_mumo_test.jsonl", [])

    with pytest.raises(ValueError, match="already aligned"):
        align_sft_think_data(
            sft_think_dir=str(src),
            canonical_dir=str(tmp_path / "canonical"),
            output_dir=str(out),
        )


def test_align_sft_think_data_allows_val_test_alias_duplicate_sample_id(tmp_path: Path):
    from marco.data.align_sft_think_data import align_sft_think_data

    src = tmp_path / "src"
    out = tmp_path / "out"
    row = _sft_row(
        "prompt",
        "<think>Increase bbbp. Increase drd2. Increase plogp.</think>\n<answer>CCN</answer>",
    )
    row["metadata"] = {**row["metadata"], "sample_id": " alias:0 "}
    write_jsonl(src / "sft_train.jsonl", [row])
    write_jsonl(src / "sft_train_failures.jsonl", [])
    write_jsonl(tmp_path / "canonical" / "canonical_mumo_train.jsonl", [])
    write_jsonl(
        tmp_path / "canonical" / "canonical_mumo_val.jsonl",
        [_canonical_row("alias:0", split="val", x0_smiles="CCO", reference_smiles="CCN")],
    )
    write_jsonl(
        tmp_path / "canonical" / "canonical_mumo_test.jsonl",
        [_canonical_row(" alias:0 ", split="test", x0_smiles="CCO", reference_smiles="CCN")],
    )

    outputs = align_sft_think_data(
        sft_think_dir=str(src),
        canonical_dir=str(tmp_path / "canonical"),
        output_dir=str(out),
    )

    rows = read_jsonl(outputs["train_jsonl"])
    assert len(rows) == 1
    assert rows[0]["metadata"]["sample_id"] == "alias:0"


def test_align_sft_think_data_raises_on_conflicting_duplicate_canonical_sample_id(tmp_path: Path):
    from marco.data.align_sft_think_data import align_sft_think_data

    src = tmp_path / "src"
    out = tmp_path / "out"
    row = _sft_row(
        "prompt",
        "<think>Increase bbbp. Increase drd2. Increase plogp.</think>\n<answer>CCN</answer>",
    )
    row["metadata"] = {**row["metadata"], "sample_id": "dup:0"}
    write_jsonl(src / "sft_train.jsonl", [row])
    write_jsonl(src / "sft_train_failures.jsonl", [])
    write_jsonl(tmp_path / "canonical" / "canonical_mumo_train.jsonl", [])
    write_jsonl(
        tmp_path / "canonical" / "canonical_mumo_val.jsonl",
        [_canonical_row("dup:0", split="val", x0_smiles="CCO")],
    )
    write_jsonl(
        tmp_path / "canonical" / "canonical_mumo_test.jsonl",
        [_canonical_row("dup:0", split="test", x0_smiles="CCC")],
    )

    with pytest.raises(ValueError, match="duplicate canonical sample_id=dup:0"):
        align_sft_think_data(
            sft_think_dir=str(src),
            canonical_dir=str(tmp_path / "canonical"),
            output_dir=str(out),
        )


def test_align_sft_think_data_print_outputs_handles_nested_by_subtask_maps(capsys):
    from marco.data.align_sft_think_data import _print_outputs

    _print_outputs(
        {
            "train_jsonl": "data/sft_think_aligned/sft_train.jsonl",
            "train_by_subtask": {"bbbp+drd2+plogp": "data/sft_think_aligned/by_subtask/bbbp+drd2+plogp/sft_train.jsonl"},
        }
    )

    output = capsys.readouterr().out
    assert "train_jsonl:" in output
    assert "train_by_subtask[bbbp+drd2+plogp]:" in output
