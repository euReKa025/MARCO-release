from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from marco.eval.multi_turn_ckpt_evaluate import summarize_trajectories, write_evaluation_outputs


def test_summarize_trajectories_uses_success_turn_for_selected_similarity_and_horizon_for_turn_usage():
    trajectories = [
        {
            "trajectory_id": "t1",
            "subtask": "bbbp+drd2+plogp",
            "x0_smiles": "CCO",
            "turns": [
                {"turn_id": 1, "invalid_type": "ok", "met_all_targets": False, "similarity": 0.8, "directional_improvements": {"bbbp": 0.1}, "active_property_names": ["bbbp"], "candidate_smiles": "CCN"},
                {"turn_id": 2, "invalid_type": "ok", "met_all_targets": True, "similarity": 0.7, "directional_improvements": {"bbbp": 0.2}, "active_property_names": ["bbbp"], "candidate_smiles": "CCC"},
                {"turn_id": 3, "invalid_type": "ok", "met_all_targets": False, "similarity": 0.1, "directional_improvements": {"bbbp": -1.0}, "active_property_names": ["bbbp"], "candidate_smiles": "BAD"},
            ],
        }
    ]

    payload = summarize_trajectories(trajectories, similarity_threshold=0.5)

    assert payload["metrics"]["success_rate"] == 1.0
    assert payload["metrics"]["avg_turns_used"] == 3.0
    assert "final_similarity_mean" not in payload["metrics"]
    assert payload["metrics"]["valid_selected_similarity_mean"] == 0.7


def test_write_evaluation_outputs_writes_repo_style_files(tmp_path: Path):
    trajectories = [
        {
            "trajectory_id": "t1",
            "subtask": "bbbp+drd2+plogp",
            "x0_smiles": "CCO",
            "turns": [
                {"turn_id": 1, "invalid_type": "ok", "met_all_targets": True, "similarity": 0.8, "directional_improvements": {"bbbp": 0.1}, "active_property_names": ["bbbp"], "candidate_smiles": "CCN"},
            ],
        }
    ]
    summary = summarize_trajectories(trajectories, similarity_threshold=0.5)
    experiment_dir = tmp_path / "exp"
    output_folder = tmp_path / "MuMo_performance"

    paths = write_evaluation_outputs(
        summary,
        experiment_dir=experiment_dir,
        output_folder=output_folder,
        IND_setting="IND",
        seen_setting="seen",
        property_setting="bbbp+drd2+plogp",
        method_name="marco",
    )

    detailed = pd.read_csv(paths["detailed_csv"])
    avg_df = pd.read_csv(paths["avg_csv"])
    summary_json = json.loads(Path(paths["summary_json"]).read_text(encoding="utf-8"))

    assert detailed.loc[0, "trajectory_id"] == "t1"
    assert float(avg_df.loc[0, "success_rate"]) == 1.0
    assert summary_json["success_rate"] == 1.0


def test_write_evaluation_outputs_uses_selected_success_turn_similarity(tmp_path: Path):
    trajectories = [
        {
            "trajectory_id": "t1",
            "subtask": "bbbp+drd2+plogp",
            "x0_smiles": "CCO",
            "turns": [
                {"turn_id": 1, "invalid_type": "ok", "met_all_targets": False, "similarity": 0.80, "directional_improvements": {"bbbp": 0.1}, "active_property_names": ["bbbp"], "candidate_smiles": "CCN"},
                {"turn_id": 2, "invalid_type": "ok", "met_all_targets": True, "similarity": 0.72, "directional_improvements": {"bbbp": 0.2}, "active_property_names": ["bbbp"], "candidate_smiles": "CCC"},
                {"turn_id": 3, "invalid_type": "ok", "met_all_targets": False, "similarity": 0.10, "directional_improvements": {"bbbp": -1.0}, "active_property_names": ["bbbp"], "candidate_smiles": "BAD"},
            ],
        }
    ]
    summary = summarize_trajectories(trajectories, similarity_threshold=0.5)
    paths = write_evaluation_outputs(
        summary,
        experiment_dir=tmp_path / "exp",
        output_folder=tmp_path / "MuMo_performance",
        IND_setting="IND",
        seen_setting="seen",
        property_setting="bbbp+drd2+plogp",
        method_name="marco",
    )

    detailed = pd.read_csv(paths["detailed_csv"])
    assert float(detailed.loc[0, "final_similarity"]) == pytest.approx(0.72)
    assert detailed.loc[0, "final_candidate_smiles"] == "CCC"


def test_write_evaluation_outputs_selected_turn_fields_beat_horizon_fields(tmp_path: Path):
    trajectories = [
        {
            "trajectory_id": "t1",
            "subtask": "bbbp+drd2+plogp",
            "x0_smiles": "CCO",
            "turns": [
                {"turn_id": 1, "invalid_type": "ok", "met_all_targets": False, "similarity": 0.81, "directional_improvements": {"bbbp": 0.1}, "active_property_names": ["bbbp"], "candidate_smiles": "CCN"},
                {"turn_id": 2, "invalid_type": "ok", "met_all_targets": True, "similarity": 0.71, "directional_improvements": {"bbbp": 0.3}, "active_property_names": ["bbbp"], "candidate_smiles": "CCC"},
                {"turn_id": 3, "invalid_type": "ok", "met_all_targets": False, "similarity": 0.10, "directional_improvements": {"bbbp": -1.0}, "active_property_names": ["bbbp"], "candidate_smiles": "BAD", "stop_reason": "max_turns"},
            ],
        }
    ]
    summary = summarize_trajectories(trajectories, similarity_threshold=0.5)
    paths = write_evaluation_outputs(
        summary,
        experiment_dir=tmp_path / "exp",
        output_folder=tmp_path / "MuMo_performance",
        IND_setting="IND",
        seen_setting="seen",
        property_setting="bbbp+drd2+plogp",
        method_name="marco",
    )

    detailed = pd.read_csv(paths["detailed_csv"])
    avg_df = pd.read_csv(paths["avg_csv"])
    summary_json = json.loads(Path(paths["summary_json"]).read_text(encoding="utf-8"))
    assert detailed.loc[0, "final_candidate_smiles"] == "CCC"
    assert float(detailed.loc[0, "final_similarity"]) == pytest.approx(0.71)
    assert float(detailed.loc[0, "meets_similarity_threshold"]) == pytest.approx(1.0)
    assert float(detailed.loc[0, "improvement_bbbp"]) == pytest.approx(0.3)
    assert "final_similarity_mean" not in avg_df.columns
    assert float(avg_df.loc[0, "valid_selected_similarity_mean"]) == pytest.approx(
        float(detailed.loc[0, "final_similarity"])
    )
    assert float(avg_df.loc[0, "final_directional_improvement/bbbp"]) == pytest.approx(
        float(detailed.loc[0, "improvement_bbbp"])
    )
    assert "final_similarity_mean" not in summary_json
    assert float(summary_json["valid_selected_similarity_mean"]) == pytest.approx(
        float(detailed.loc[0, "final_similarity"])
    )
    assert float(summary_json["final_directional_improvement/bbbp"]) == pytest.approx(
        float(detailed.loc[0, "improvement_bbbp"])
    )
