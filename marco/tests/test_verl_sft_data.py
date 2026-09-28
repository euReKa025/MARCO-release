from pathlib import Path

import pyarrow.parquet as pq
import pytest

from marco.utils import write_jsonl
from marco.verl.data.build_sft_parquet import build_sft_parquet


def test_build_sft_parquet_writes_messages_and_metadata(tmp_path: Path) -> None:
    src = tmp_path / "sft_train.jsonl"
    out = tmp_path / "sft_train.parquet"
    assistant = "<think>brief reasoning</think><answer>CCO</answer>"
    write_jsonl(
        str(src),
        [
            {
                "messages": [
                    {"role": "system", "content": "system prompt"},
                    {"role": "user", "content": "user prompt"},
                    {"role": "assistant", "content": assistant},
                ],
                "metadata": {"sample_id": "s1", "subtask": "bbbp+drd2+qed"},
            }
        ],
    )

    build_sft_parquet(input_jsonl=str(src), output_parquet=str(out))

    table = pq.read_table(out)
    assert set(table.column_names) == {"messages", "metadata"}
    assert table.num_rows == 1

    row = table.to_pylist()[0]
    assert row["messages"][2]["content"] == assistant
    assert row["metadata"]["sample_id"] == "s1"
    assert row["metadata"]["subtask"] == "bbbp+drd2+qed"


def test_verl_sft_config_and_launcher_exist() -> None:
    root = Path(__file__).resolve().parents[2]
    assert (root / "marco" / "verl" / "config" / "sft_marco_fsdp.yaml").exists()
    assert (root / "scripts" / "verl" / "run_marco_sft_qwen.sh").exists()


def test_build_sft_parquet_fails_on_empty_input(tmp_path: Path) -> None:
    src = tmp_path / "empty.jsonl"
    out = tmp_path / "out.parquet"
    write_jsonl(str(src), [])

    with pytest.raises(ValueError, match="input JSONL is empty"):
        build_sft_parquet(input_jsonl=str(src), output_parquet=str(out))


def test_build_sft_parquet_fails_when_messages_not_list(tmp_path: Path) -> None:
    src = tmp_path / "bad_messages.jsonl"
    out = tmp_path / "out.parquet"
    write_jsonl(
        str(src),
        [
            {
                "messages": "not-a-list",
                "metadata": {"sample_id": "s1"},
            }
        ],
    )

    with pytest.raises(ValueError, match="messages must be a list"):
        build_sft_parquet(input_jsonl=str(src), output_parquet=str(out))


def test_build_sft_parquet_fails_when_metadata_not_dict(tmp_path: Path) -> None:
    src = tmp_path / "bad_metadata.jsonl"
    out = tmp_path / "out.parquet"
    write_jsonl(
        str(src),
        [
            {
                "messages": [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": "<think>x</think><answer>CCO</answer>"},
                ],
                "metadata": "not-a-dict",
            }
        ],
    )

    with pytest.raises(ValueError, match="metadata must be a dict"):
        build_sft_parquet(input_jsonl=str(src), output_parquet=str(out))


def test_sft_config_and_launcher_use_dynamic_verl_config_path() -> None:
    root = Path(__file__).resolve().parents[2]
    config_text = (root / "marco" / "verl" / "config" / "sft_marco_fsdp.yaml").read_text(encoding="utf-8")
    launcher_text = (root / "scripts" / "verl" / "run_marco_sft_qwen.sh").read_text(encoding="utf-8")
    assert "oc.env:VERL_CONFIG_PATH" in config_text
    assert "VERL_CONFIG_PATH" in launcher_text
    assert "import verl" in launcher_text


def test_sft_config_uses_fsdp_engine_strategy_field() -> None:
    root = Path(__file__).resolve().parents[2]
    config_text = (root / "marco" / "verl" / "config" / "sft_marco_fsdp.yaml").read_text(encoding="utf-8")
    assert "engine:" in config_text
    assert "strategy: fsdp" in config_text
    assert "distributed_strategy:" not in config_text
