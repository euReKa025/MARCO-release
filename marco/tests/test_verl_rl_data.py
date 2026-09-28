import os
import subprocess
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from marco.core_types import EpisodeState, PropertyTarget
from marco.env.molecule_env import MoleculeEnv
from marco.utils import write_jsonl
from marco.prompts.system_prompt import DIRECT_SMILES_SYSTEM_PROMPT, PROMPT_MODE_DIRECT_SMILES, PROMPT_MODE_THINK_ANSWER
from marco.verl.data.build_rlhf_parquet import build_rlhf_parquet
from marco.verl.data.schema import canonical_record_to_verl_row
from marco.verl.interaction.trajectory import bootstrap_trajectory_state


class _FakeBootstrapEnv:
    def init_episode(self, *, record, max_turns):
        targets = [
            PropertyTarget(name=item["name"], direction=item["direction"], delta=1.0)
            for item in record.get("properties", [])
        ]
        return EpisodeState(
            episode_id=record["sample_id"],
            subtask=record["subtask"],
            x0_smiles=record["x0_smiles"],
            current_smiles=record["x0_smiles"],
            property_targets=targets,
            x0_predictions={"bbbp": 0.0},
            current_predictions={"bbbp": 0.0},
            current_total_gap=0.0,
            turn_id=0,
            max_turns=max_turns,
            history=[],
        )

    @staticmethod
    def state_to_payload(state):
        return MoleculeEnv.state_to_payload(state)


def _capture_python_bin(tmp_path: Path) -> tuple[Path, Path, Path]:
    python_bin = tmp_path / "capture_python_bin.sh"
    rollout_file = tmp_path / "captured_rollout_max_response_len.txt"
    args_file = tmp_path / "captured_args.txt"
    python_bin.write_text(
        (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "printf '%s' \"${ROLLOUT_MAX_RESPONSE_LEN:-}\" > \"${CAPTURE_ROLLOUT_FILE:?}\"\n"
            "if [[ -n \"${CAPTURE_ADMET_PORT_FILE:-}\" ]]; then\n"
            "  printf '%s' \"${ADMET_PORT:-}\" > \"${CAPTURE_ADMET_PORT_FILE}\"\n"
            "fi\n"
            "if [[ -n \"${CAPTURE_DRD2_PORT_FILE:-}\" ]]; then\n"
            "  printf '%s' \"${DRD2_PORT:-}\" > \"${CAPTURE_DRD2_PORT_FILE}\"\n"
            "fi\n"
            "if [[ -n \"${CAPTURE_PREDICTOR_HOST_FILE:-}\" ]]; then\n"
            "  printf '%s' \"${PREDICTOR_HOST:-}\" > \"${CAPTURE_PREDICTOR_HOST_FILE}\"\n"
            "fi\n"
            "printf '%s\\n' \"$@\" > \"${CAPTURE_ARGS_FILE:?}\"\n"
            "exit 0\n"
        ),
        encoding="utf-8",
    )
    python_bin.chmod(0o755)
    return python_bin, rollout_file, args_file


def test_canonical_record_to_verl_row_keeps_prompt_and_extra_info():
    record = {
        "sample_id": "s1",
        "subtask": "bbbp+drd2+qed",
        "instruction": "Optimize this molecule.",
        "reference_smiles": "CCO",
        "x0_smiles": "CCC",
        "properties": [{"name": "bbbp", "direction": "increase"}],
        "metadata": {"split": "train"},
    }

    row = canonical_record_to_verl_row(record)

    assert isinstance(row["prompt"], list)
    assert len(row["prompt"]) == 2
    assert row["prompt"][0]["role"] == "system"
    assert row["prompt"][1]["role"] == "user"
    assert row["prompt"][1]["content"] == "Optimize this molecule."
    assert "Source molecule (x0): CCC" not in row["prompt"][1]["content"]
    assert row["ground_truth"] == "CCO"
    assert row["data_source"] == "marco_mumo"
    assert row["agent_name"] == "marco_tool_agent"
    assert row["reward_model"]["ground_truth"] == "CCO"
    assert row["extra_info"]["sample_id"] == "s1"
    assert row["extra_info"]["subtask"] == "bbbp+drd2+qed"
    assert row["extra_info"]["x0_smiles"] == "CCC"
    assert row["extra_info"]["property_targets"] == [{"name": "bbbp", "direction": "increase"}]
    assert row["extra_info"]["interaction_kwargs"]["name"] == "marco"
    assert row["extra_info"]["interaction_kwargs"]["max_turns"] == 5
    assert row["extra_info"]["prompt_mode"] == PROMPT_MODE_THINK_ANSWER
    assert row["extra_info"]["interaction_kwargs"]["prompt_mode"] == PROMPT_MODE_THINK_ANSWER


def test_canonical_record_to_verl_row_accepts_direct_smiles_prompt_mode():
    record = {
        "sample_id": "s-direct",
        "subtask": "bbbp+drd2+qed",
        "instruction": "Optimize this molecule.",
        "reference_smiles": "CCO",
        "x0_smiles": "CCC",
        "properties": [{"name": "bbbp", "direction": "increase"}],
        "metadata": {"split": "train"},
    }

    row = canonical_record_to_verl_row(record, prompt_mode=PROMPT_MODE_DIRECT_SMILES)

    assert row["prompt"][0]["content"] == DIRECT_SMILES_SYSTEM_PROMPT
    assert row["prompt"][1]["content"] == "Optimize this molecule."
    assert row["extra_info"]["prompt_mode"] == PROMPT_MODE_DIRECT_SMILES
    assert row["extra_info"]["interaction_kwargs"]["prompt_mode"] == PROMPT_MODE_DIRECT_SMILES


def test_canonical_record_to_verl_row_uses_empty_ground_truth_for_source_only_records():
    record = {
        "sample_id": "source-only:0",
        "subtask": "bbbp+hia+mutagenicity+qed",
        "instruction": "Optimize this molecule.",
        "reference_smiles": None,
        "x0_smiles": "CCC",
        "properties": [
            {"name": "bbbp", "direction": "increase", "source": 0.2},
            {"name": "hia", "direction": "increase", "source": 0.7},
            {"name": "mutagenicity", "direction": "decrease", "source": 0.4},
            {"name": "qed", "direction": "increase", "source": 0.5},
        ],
        "metadata": {"split": "train"},
    }

    row = canonical_record_to_verl_row(record)

    assert row["ground_truth"] == ""
    assert row["reward_model"]["ground_truth"] == ""


def test_canonical_record_to_verl_row_uses_raw_instruction_fallback():
    record = {
        "sample_id": "source-only:raw-instruction",
        "subtask": "bbbp+hia+mutagenicity+qed",
        "reference_smiles": None,
        "x0_smiles": "CCC",
        "properties": [{"name": "bbbp", "direction": "increase", "source": 0.2}],
        "metadata": {"split": "train", "raw_instruction": "Canonical raw instruction from metadata."},
    }

    row = canonical_record_to_verl_row(record)

    assert row["prompt"][1]["content"] == "Canonical raw instruction from metadata."
    assert row["extra_info"]["instruction"] == "Canonical raw instruction from metadata."


def test_canonical_record_to_verl_row_requires_dataset_instruction():
    record = {
        "sample_id": "missing-instruction",
        "subtask": "bbbp+drd2+qed",
        "reference_smiles": "CCO",
        "x0_smiles": "CCC",
        "properties": [{"name": "bbbp", "direction": "increase"}],
        "metadata": {"split": "train"},
    }

    with pytest.raises(ValueError, match="missing non-empty instruction"):
        canonical_record_to_verl_row(record)


def test_build_rlhf_parquet_writes_expected_columns(tmp_path: Path):
    src = tmp_path / "train.jsonl"
    out = tmp_path / "train.parquet"
    write_jsonl(
        str(src),
        [
            {
                "sample_id": "s1",
                "subtask": "bbbp+drd2+qed",
                "instruction": "Optimize this molecule.",
                "reference_smiles": "CCO",
                "x0_smiles": "CCC",
                "properties": [{"name": "bbbp", "direction": "increase"}],
                "metadata": {"split": "train"},
            }
        ],
    )

    build_rlhf_parquet(input_jsonl=str(src), output_parquet=str(out))

    table = pq.read_table(out)
    assert set(table.column_names) == {
        "prompt",
        "ground_truth",
        "reward_model",
        "data_source",
        "agent_name",
        "extra_info",
    }
    assert table.num_rows == 1


def test_build_rlhf_parquet_accepts_direct_smiles_prompt_mode(tmp_path: Path):
    src = tmp_path / "train.jsonl"
    out = tmp_path / "train.parquet"
    write_jsonl(
        str(src),
        [
            {
                "sample_id": "s-direct",
                "subtask": "bbbp+drd2+qed",
                "instruction": "Optimize this molecule.",
                "reference_smiles": "CCO",
                "x0_smiles": "CCC",
                "properties": [{"name": "bbbp", "direction": "increase"}],
                "metadata": {"split": "train"},
            }
        ],
    )

    build_rlhf_parquet(input_jsonl=str(src), output_parquet=str(out), prompt_mode=PROMPT_MODE_DIRECT_SMILES)

    row = pq.read_table(out).to_pylist()[0]
    assert row["prompt"][0]["content"] == DIRECT_SMILES_SYSTEM_PROMPT
    assert row["extra_info"]["prompt_mode"] == PROMPT_MODE_DIRECT_SMILES
    assert row["extra_info"]["interaction_kwargs"]["prompt_mode"] == PROMPT_MODE_DIRECT_SMILES


def test_bootstrap_trajectory_state_starts_with_initial_messages():
    row = {
        "prompt": [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}],
        "extra_info": {
            "sample_id": "s1",
            "subtask": "bbbp+drd2+qed",
            "x0_smiles": "CCC",
            "property_targets": [{"name": "bbbp", "direction": "increase"}],
            "canonical_record": {
                "sample_id": "s1",
                "subtask": "bbbp+drd2+qed",
                "instruction": "Dataset instruction for MARCO RL.",
                "x0_smiles": "CCC",
                "properties": [{"name": "bbbp", "direction": "increase"}],
            },
        },
    }

    state = bootstrap_trajectory_state(row, max_turns=5, env=_FakeBootstrapEnv())

    assert state["episode_id"] == "s1"
    assert state["turn_id"] == 0
    assert state["max_turns"] == 5
    assert isinstance(state["messages"], list)
    assert len(state["messages"]) == 2
    assert state["messages"][1]["content"] == "Dataset instruction for MARCO RL."
    assert "Source molecule (x0)" not in state["messages"][1]["content"]
    assert state["instruction"] == "Dataset instruction for MARCO RL."


def test_verl_grpo_config_exists():
    path = Path(__file__).resolve().parents[1] / "verl" / "config" / "grpo_marco_fsdp_sglang.yaml"
    assert path.exists()


def test_verl_multiturn_config_files_exist():
    config_root = Path(__file__).resolve().parents[1] / "verl" / "config"
    assert (config_root / "agent_loop" / "marco_agent_loop.yaml").exists()
    assert (config_root / "interaction_config" / "marco_interaction.yaml").exists()


def test_marco_ordinary_rl_launcher_skips_predictor_startup_when_start_predictors_zero(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "verl" / "run_marco_rl_ord_qwen.sh"
    if not (root / ".venv" / "bin" / "activate").exists():
        pytest.skip("missing .venv activation script")

    env = os.environ.copy()
    env["START_PREDICTORS"] = "0"
    env["PYTHON_BIN"] = "true"
    env["VERL_CONFIG_PATH"] = str(tmp_path)
    env["MODEL_PATH"] = "Qwen/Qwen2.5-3B-Instruct"
    env["TRAIN_PARQUET"] = str(tmp_path / "train.parquet")
    env["VAL_PARQUET"] = str(tmp_path / "val.parquet")
    env["MARCO_ADMET_API"] = "http://127.0.0.1:18086/predict/"
    env["MARCO_DRD2_API"] = "http://127.0.0.1:18087/predict/"
    env["MARCO_PREDICTOR_CACHE_PATH"] = str(tmp_path / "predictor_cache.sqlite3")

    result = subprocess.run(
        ["bash", str(script)],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_marco_ordinary_rl_config_and_launcher_static_assertions() -> None:
    root = Path(__file__).resolve().parents[2]
    config_text = (root / "marco" / "verl" / "config" / "grpo_marco_fsdp_sglang.yaml").read_text(encoding="utf-8")
    launcher_text = (root / "scripts" / "verl" / "run_marco_rl_ord_qwen.sh").read_text(encoding="utf-8")

    assert "oc.env:VERL_CONFIG_PATH" in config_text
    assert "adv_estimator: grpo" in config_text
    assert "prompt_key: prompt" in config_text
    assert "return_raw_chat: true" in config_text
    assert "multi_turn:" in config_text
    assert "interaction_config_path: marco/verl/config/interaction_config/marco_interaction.yaml" in config_text

    assert "VERL_CONFIG_PATH" in launcher_text
    assert "start_predictors.sh" in launcher_text
    assert 'if [[ "$START_PREDICTORS" == "1" ]]; then' in launcher_text
    assert "MARCO_ADMET_API" in launcher_text
    assert "MARCO_DRD2_API" in launcher_text
    assert "MARCO_PREDICTOR_CACHE_PATH" in launcher_text
    assert "ROLLOUT_MAX_RESPONSE_LEN" in launcher_text
    assert "extract_endpoint_field" in launcher_text
    assert "PYTHON_BIN" in launcher_text
    assert "OUTPUT_DIR" in launcher_text
    assert "PROJECT_NAME" in launcher_text
    assert "EXPERIMENT_NAME" in launcher_text
    assert "main_ppo" in launcher_text


def test_marco_ordinary_rl_launcher_aligns_rollout_response_len_env(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "verl" / "run_marco_rl_ord_qwen.sh"
    if not (root / ".venv" / "bin" / "activate").exists():
        pytest.skip("missing .venv activation script")

    python_bin, rollout_file, args_file = _capture_python_bin(tmp_path)

    env = os.environ.copy()
    env["START_PREDICTORS"] = "0"
    env["PYTHON_BIN"] = str(python_bin)
    env["VERL_CONFIG_PATH"] = str(tmp_path)
    env["MODEL_PATH"] = "Qwen/Qwen2.5-3B-Instruct"
    env["TRAIN_PARQUET"] = str(tmp_path / "train.parquet")
    env["VAL_PARQUET"] = str(tmp_path / "val.parquet")
    env["MAX_RESPONSE_LENGTH"] = "1536"
    env["ROLLOUT_MAX_RESPONSE_LEN"] = "777"
    env["CAPTURE_ROLLOUT_FILE"] = str(rollout_file)
    env["CAPTURE_ARGS_FILE"] = str(args_file)

    result = subprocess.run(
        ["bash", str(script)],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert rollout_file.read_text(encoding="utf-8") == "1536"
    args_text = args_file.read_text(encoding="utf-8")
    assert "data.max_response_length=1536" in args_text


def test_marco_ordinary_rl_launcher_derives_predictor_ports_from_endpoint_overrides(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "verl" / "run_marco_rl_ord_qwen.sh"
    if not (root / ".venv" / "bin" / "activate").exists():
        pytest.skip("missing .venv activation script")

    python_bin, rollout_file, args_file = _capture_python_bin(tmp_path)
    admet_port_file = tmp_path / "captured_admet_port.txt"
    drd2_port_file = tmp_path / "captured_drd2_port.txt"
    predictor_host_file = tmp_path / "captured_predictor_host.txt"

    env = os.environ.copy()
    env["START_PREDICTORS"] = "0"
    env["PYTHON_BIN"] = str(python_bin)
    env["VERL_CONFIG_PATH"] = str(tmp_path)
    env["MODEL_PATH"] = "Qwen/Qwen2.5-3B-Instruct"
    env["TRAIN_PARQUET"] = str(tmp_path / "train.parquet")
    env["VAL_PARQUET"] = str(tmp_path / "val.parquet")
    env["MARCO_ADMET_API"] = "http://127.0.0.1:18086/predict/"
    env["MARCO_DRD2_API"] = "http://127.0.0.1:18087/predict/"
    env["CAPTURE_ROLLOUT_FILE"] = str(rollout_file)
    env["CAPTURE_ARGS_FILE"] = str(args_file)
    env["CAPTURE_ADMET_PORT_FILE"] = str(admet_port_file)
    env["CAPTURE_DRD2_PORT_FILE"] = str(drd2_port_file)
    env["CAPTURE_PREDICTOR_HOST_FILE"] = str(predictor_host_file)

    result = subprocess.run(
        ["bash", str(script)],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert admet_port_file.read_text(encoding="utf-8") == "18086"
    assert drd2_port_file.read_text(encoding="utf-8") == "18087"
    assert predictor_host_file.read_text(encoding="utf-8") == "127.0.0.1"
