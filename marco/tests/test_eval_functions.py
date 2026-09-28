import pytest

from marco.eval.eval_multi_turn import evaluate_multi_turn_trajectories
from marco.eval.eval_single_turn import evaluate_single_turn_trajectories


def test_single_turn_evaluator_directional_success_with_similarity_gate():
    trajectories = [
        {
            "trajectory_id": "t1",
            "turns": [
                {
                    "invalid_type": "ok",
                    "met_all_targets": True,
                    "similarity": 0.8,
                    "directional_improvements": {"bbbp": 0.1, "drd2": 0.2},
                    "active_property_names": ["bbbp", "drd2"],
                }
            ],
        },
        {
            "trajectory_id": "t2",
            "turns": [
                {
                    "invalid_type": "ok",
                    "met_all_targets": True,
                    "similarity": 0.4,
                    "directional_improvements": {"bbbp": 0.1, "drd2": 0.2},
                    "active_property_names": ["bbbp", "drd2"],
                }
            ],
        },
        {
            "trajectory_id": "t3",
            "turns": [
                {
                    "invalid_type": "invalid_parse",
                    "met_all_targets": False,
                    "similarity": 0.9,
                    "directional_improvements": {"bbbp": 0.0, "drd2": 0.0},
                    "active_property_names": ["bbbp", "drd2"],
                }
            ],
        },
    ]

    out = evaluate_single_turn_trajectories(trajectories, similarity_threshold=0.5)

    assert out["metrics"]["num_trajectories"] == pytest.approx(3.0)
    assert out["metrics"]["success_rate"] == pytest.approx(2.0 / 3.0)
    assert out["metrics"]["constraint_success_rate"] == pytest.approx(1.0 / 3.0)
    assert out["metrics"]["invalid_rate"] == pytest.approx(1.0 / 3.0)


def test_multi_turn_evaluator_tracks_turns_oracle_and_invalids():
    trajectories = [
        {
            "trajectory_id": "t1",
            "subtask": "bbbp+drd2+qed",
            "x0_smiles": "CCO",
            "turns": [
                {
                    "turn_id": 1,
                    "invalid_type": "ok",
                    "met_all_targets": False,
                    "similarity": 0.75,
                    "directional_improvements": {"bbbp": 0.1, "drd2": -0.1, "qed": 0.05},
                    "active_property_names": ["bbbp", "drd2", "qed"],
                    "candidate_smiles": "CCN",
                },
                {
                    "turn_id": 2,
                    "invalid_type": "ok",
                    "met_all_targets": True,
                    "similarity": 0.72,
                    "directional_improvements": {"bbbp": 0.2, "drd2": 0.1, "qed": 0.08},
                    "active_property_names": ["bbbp", "drd2", "qed"],
                    "candidate_smiles": "CCC",
                },
            ],
        },
        {
            "trajectory_id": "t2",
            "subtask": "bbbp+drd2+qed",
            "x0_smiles": "CCO",
            "turns": [
                {
                    "turn_id": 1,
                    "invalid_type": "invalid_parse",
                    "met_all_targets": False,
                    "similarity": 0.80,
                    "directional_improvements": {"bbbp": 0.0, "drd2": 0.0, "qed": 0.0},
                    "active_property_names": ["bbbp", "drd2", "qed"],
                    "candidate_smiles": None,
                },
                {
                    "turn_id": 2,
                    "invalid_type": "ok",
                    "met_all_targets": False,
                    "similarity": 0.78,
                    "directional_improvements": {"bbbp": 0.05, "drd2": -0.1, "qed": 0.03},
                    "active_property_names": ["bbbp", "drd2", "qed"],
                    "candidate_smiles": "CCCl",
                },
                {
                    "turn_id": 3,
                    "invalid_type": "ok",
                    "met_all_targets": False,
                    "similarity": 0.76,
                    "directional_improvements": {"bbbp": 0.08, "drd2": -0.05, "qed": 0.04},
                    "active_property_names": ["bbbp", "drd2", "qed"],
                    "candidate_smiles": "CCBr",
                },
            ],
        },
    ]

    out = evaluate_multi_turn_trajectories(trajectories, similarity_threshold=0.5)

    assert out["metrics"]["num_trajectories"] == pytest.approx(2.0)
    assert out["metrics"]["success_rate"] == pytest.approx(0.5)
    assert out["metrics"]["constraint_success_rate"] == pytest.approx(0.5)
    assert out["metrics"]["avg_turns_used"] == pytest.approx(2.5)
    assert out["metrics"]["invalid_turns_per_trajectory"] == pytest.approx(0.5)
    assert out["metrics"]["avg_oracle_calls"] == pytest.approx(2.0)
    assert len(out["success_examples"]) == 1


def test_multi_turn_evaluator_separates_success_from_in_range_success():
    trajectories = [
        {
            "trajectory_id": "t1",
            "turns": [
                {
                    "turn_id": 1,
                    "invalid_type": "ok",
                    "met_all_targets": True,
                    "similarity": 0.2,
                    "directional_improvements": {"bbbp": 0.1},
                    "active_property_names": ["bbbp"],
                    "candidate_smiles": "CCN",
                }
            ],
        },
        {
            "trajectory_id": "t2",
            "turns": [
                {
                    "turn_id": 1,
                    "invalid_type": "ok",
                    "met_all_targets": True,
                    "similarity": 0.7,
                    "directional_improvements": {"bbbp": 0.1},
                    "active_property_names": ["bbbp"],
                    "candidate_smiles": "CCC",
                }
            ],
        },
    ]

    out = evaluate_multi_turn_trajectories(trajectories, similarity_threshold=0.6)

    assert out["metrics"]["success_rate"] == pytest.approx(1.0)
    assert out["metrics"]["constraint_success_rate"] == pytest.approx(0.5)


def test_multi_turn_evaluator_uses_first_property_success_turn_for_selected_fields_with_later_horizon():
    trajectories = [
        {
            "trajectory_id": "t1",
            "subtask": "bbbp",
            "x0_smiles": "CCO",
            "turns": [
                {
                    "turn_id": 1,
                    "invalid_type": "ok",
                    "met_all_targets": False,
                    "similarity": 0.82,
                    "directional_improvements": {"bbbp": 0.05},
                    "active_property_names": ["bbbp"],
                    "candidate_smiles": "CCN",
                },
                {
                    "turn_id": 2,
                    "invalid_type": "ok",
                    "met_all_targets": True,
                    "similarity": 0.72,
                    "directional_improvements": {"bbbp": 0.30},
                    "active_property_names": ["bbbp"],
                    "candidate_smiles": "CCC",
                },
                {
                    "turn_id": 3,
                    "invalid_type": "ok",
                    "met_all_targets": False,
                    "similarity": 0.11,
                    "directional_improvements": {"bbbp": -1.0},
                    "active_property_names": ["bbbp"],
                    "candidate_smiles": "BAD",
                    "stop_reason": "max_turns",
                },
            ],
        }
    ]

    out = evaluate_multi_turn_trajectories(trajectories, similarity_threshold=0.5)

    assert out["metrics"]["success_rate"] == pytest.approx(1.0)
    assert out["metrics"]["avg_turns_used"] == pytest.approx(3.0)
    assert "final_similarity_mean" not in out["metrics"]
    assert out["metrics"]["valid_selected_similarity_mean"] == pytest.approx(0.72)
    assert out["metrics"]["final_directional_improvement/bbbp"] == pytest.approx(0.30)
    assert out["success_examples"][0]["final_candidate_smiles"] == "CCC"
    assert out["success_examples"][0]["final_similarity"] == pytest.approx(0.72)


def test_multi_turn_evaluator_reports_only_valid_selected_similarity_mean():
    trajectories = [
        {
            "trajectory_id": "valid-success",
            "turns": [
                {
                    "turn_id": 1,
                    "invalid_type": "ok",
                    "met_all_targets": True,
                    "similarity": 0.70,
                    "directional_improvements": {"bbbp": 0.1},
                    "active_property_names": ["bbbp"],
                    "candidate_smiles": "CCN",
                }
            ],
        },
        {
            "trajectory_id": "invalid-horizon",
            "turns": [
                {
                    "turn_id": 1,
                    "invalid_type": "invalid_parse",
                    "met_all_targets": False,
                    "similarity": 0.0,
                    "directional_improvements": {"bbbp": 0.0},
                    "active_property_names": ["bbbp"],
                    "candidate_smiles": None,
                    "stop_reason": "max_turns",
                }
            ],
        },
        {
            "trajectory_id": "valid-failure",
            "turns": [
                {
                    "turn_id": 1,
                    "invalid_type": "ok",
                    "met_all_targets": False,
                    "similarity": 0.50,
                    "directional_improvements": {"bbbp": 0.0},
                    "active_property_names": ["bbbp"],
                    "candidate_smiles": "CCC",
                    "stop_reason": "max_turns",
                }
            ],
        },
    ]

    out = evaluate_multi_turn_trajectories(trajectories, similarity_threshold=0.4)

    assert "final_similarity_mean" not in out["metrics"]
    assert out["metrics"]["valid_selected_similarity_count"] == pytest.approx(2.0)
    assert out["metrics"]["valid_selected_similarity_rate"] == pytest.approx(2.0 / 3.0)
    assert out["metrics"]["valid_selected_similarity_mean"] == pytest.approx(0.60)


def test_multi_turn_evaluator_constraint_success_uses_similarity_acceptable_rule():
    trajectories = [
        {
            "trajectory_id": "t_accept",
            "turns": [
                {
                    "turn_id": 1,
                    "invalid_type": "ok",
                    "met_all_targets": True,
                    "similarity": 0.90,
                    "directional_improvements": {"bbbp": 0.1},
                    "active_property_names": ["bbbp"],
                }
            ],
        },
        {
            "trajectory_id": "t_copy_like",
            "turns": [
                {
                    "turn_id": 1,
                    "invalid_type": "ok",
                    "met_all_targets": True,
                    "similarity": 0.99,
                    "directional_improvements": {"bbbp": 0.1},
                    "active_property_names": ["bbbp"],
                }
            ],
        },
    ]

    out = evaluate_multi_turn_trajectories(trajectories, similarity_threshold=0.6)

    assert out["metrics"]["success_rate"] == pytest.approx(1.0)
    assert out["metrics"]["constraint_success_rate"] == pytest.approx(0.5)
