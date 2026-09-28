import asyncio

import pytest

pytest.importorskip("verl", reason="Requires the optional Linux training environment")

from marco.env.stop_rules import should_stop_episode
from marco.rollout.sample_conversion import _settle_posthoc_rewards
from marco.trainer_types import Sample
from marco.verl.interaction import marco_interaction
from marco.verl.interaction.marco_interaction import MarcoTrajectoryInteraction
from marco.verl.reward.score import build_score_payload
from marco.verl.reward.settlement import (
    settle_trajectory_rewards_from_samples,
    settle_trajectory_rewards_from_turn_payloads,
)


def _sample(turn_id: int, traj_id: str, similarity: float, progress_score: float, met: bool) -> Sample:
    s = Sample(response="<answer>CCO</answer>", response_length=8, tokens=[1, 2, 3], metadata={})
    s.train_metadata = {
        "trajectory_id": traj_id,
        "group_id": "g1",
        "turn_id": turn_id,
        "max_turns": 5,
        "invalid_type": "ok",
        "similarity": similarity,
        "progress_score": progress_score,
        "met_all_targets": met,
    }
    s.status = Sample.Status.COMPLETED
    return s


def test_verl_settlement_matches_existing_posthoc_settlement():
    samples = [
        _sample(1, "t1", 0.62, 0.30, False),
        _sample(2, "t1", 0.70, 0.80, True),
    ]

    old_step_rewards, old_breakdowns, old_returns, _, _ = _settle_posthoc_rewards(samples)
    new_step_rewards, new_breakdowns, new_returns = settle_trajectory_rewards_from_samples(samples)

    assert new_step_rewards == old_step_rewards
    assert new_returns == old_returns
    assert new_breakdowns[0]["reward_total"] == old_breakdowns[0]["reward_total"]
    assert new_breakdowns[1]["reward_total"] == old_breakdowns[1]["reward_total"]


def test_build_score_payload_keeps_required_fields():
    payload = build_score_payload(
        prompt="Optimize this molecule.",
        response="<answer>CCO</answer>",
        ground_truth="CCO",
        extra_info={"sample_id": "s1"},
    )

    assert payload["prompt"] == "Optimize this molecule."
    assert payload["response"] == "<answer>CCO</answer>"
    assert payload["ground_truth"] == "CCO"
    assert payload["extra_info"]["sample_id"] == "s1"


def test_turn_payload_settlement_matches_sample_settlement():
    samples = [
        _sample(1, "t1", 0.62, 0.30, False),
        _sample(2, "t1", 0.70, 0.80, True),
    ]
    old_step_rewards, old_breakdowns, old_returns, _, _ = _settle_posthoc_rewards(samples)

    result = settle_trajectory_rewards_from_turn_payloads(
        [
            {
                "turn_id": 1,
                "max_turns": 5,
                "invalid_type": "ok",
                "similarity": 0.62,
                "progress_score": 0.30,
                "met_all_targets": False,
                "error_detail": "",
                "response_length": 8,
                "response_text": "<answer>CCO</answer>",
            },
            {
                "turn_id": 2,
                "max_turns": 5,
                "invalid_type": "ok",
                "similarity": 0.70,
                "progress_score": 0.80,
                "met_all_targets": True,
                "error_detail": "",
                "response_length": 8,
                "response_text": "<answer>CCO</answer>",
            },
        ],
        trajectory_id="t1",
        group_id="g1",
    )

    assert result["step_rewards"] == old_step_rewards
    assert result["trajectory_reward"] == old_returns["t1"]
    assert result["breakdowns"][0]["reward_total"] == old_breakdowns[0]["reward_total"]
    assert result["breakdowns"][1]["reward_total"] == old_breakdowns[1]["reward_total"]


def _run(coro):
    return asyncio.run(coro)


def test_stop_rule_requires_similarity_acceptance_when_targets_met():
    below_min = should_stop_episode(
        turn_id=2,
        max_turns=5,
        invalid_type="ok",
        met_all_targets=True,
        similarity_acceptable=False,
    )
    acceptable = should_stop_episode(
        turn_id=2,
        max_turns=5,
        invalid_type="ok",
        met_all_targets=True,
        similarity_acceptable=True,
    )

    assert below_min.should_stop is False
    assert below_min.reason == "continue"
    assert acceptable.should_stop is True
    assert acceptable.reason == "success"


def test_marco_interaction_propagates_trajectory_reward_on_terminal_turn(monkeypatch):
    interaction = MarcoTrajectoryInteraction(config={})
    initial_state = {
        "episode_id": "ep1",
        "subtask": "bbbp",
        "x0_smiles": "CCO",
        "current_smiles": "CCO",
        "property_targets": [{"name": "bbbp", "direction": "increase", "delta": 1.0, "role": "active"}],
        "x0_predictions": {"bbbp": 0.0},
        "current_predictions": {"bbbp": 0.0},
        "current_total_gap": 0.0,
        "current_directional_improvements": {"bbbp": 0.0},
        "current_progress_score": 0.0,
        "success_turn": None,
        "turn_id": 0,
        "max_turns": 4,
        "history": [],
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "turn1"},
        ],
    }

    first_turn_payload = {
        "turn_id": 1,
        "max_turns": 4,
        "invalid_type": "ok",
        "similarity": 0.58,
        "progress_score": 1.0,
        "met_all_targets": True,
        "error_detail": "",
        "response_length": 12,
        "response_text": "<answer>CCN</answer>",
    }
    second_turn_payload = {
        "turn_id": 2,
        "max_turns": 4,
        "invalid_type": "ok",
        "similarity": 0.66,
        "progress_score": 1.0,
        "met_all_targets": True,
        "error_detail": "",
        "response_length": 13,
        "response_text": "<answer>CCCl</answer>",
    }
    expected_settlement = settle_trajectory_rewards_from_turn_payloads(
        [first_turn_payload, second_turn_payload],
        trajectory_id="traj-ord",
    )
    scripted_results = iter(
        [
            {
                "turn_result": {
                    "status": "ok",
                    "should_stop": False,
                    "met_all_targets": True,
                    "similarity": 0.58,
                    "progress_score": 1.0,
                    "message": "continue",
                    "error_detail": "",
                },
                "feedback_message": "feedback-1",
                "next_state": {**initial_state, "turn_id": 1},
                "next_user_content": "turn2",
                "invalid_streak": 0,
                "turn_payload": first_turn_payload,
            },
            {
                "turn_result": {
                    "status": "ok",
                    "should_stop": True,
                    "met_all_targets": True,
                    "similarity": 0.66,
                    "progress_score": 1.0,
                    "message": "success",
                    "error_detail": "",
                },
                "feedback_message": "feedback-2",
                "next_state": {**initial_state, "turn_id": 2},
                "next_user_content": "should_not_be_returned",
                "invalid_streak": 0,
                "turn_payload": second_turn_payload,
            },
        ]
    )

    def _fake_evaluate(**kwargs):
        del kwargs
        return next(scripted_results)

    monkeypatch.setattr(marco_interaction, "evaluate_marco_response_turn", _fake_evaluate)

    instance_id = _run(
        interaction.start_interaction(
            instance_id="traj-ord",
            initial_state_payload=initial_state,
        )
    )

    stop_1, next_user_1, reward_1, additional_1 = _run(
        interaction.generate_response(
            instance_id,
            [
                *initial_state["messages"],
                {"role": "assistant", "content": "<answer>CCN</answer>"},
            ],
            response_length=12,
            rollout_max_response_len=1024,
        )
    )
    assert stop_1 is False
    assert next_user_1 == "turn2"
    assert reward_1 == pytest.approx(0.0)
    assert "trajectory_reward" not in additional_1

    stop_2, next_user_2, reward_2, additional_2 = _run(
        interaction.generate_response(
            instance_id,
            [
                *initial_state["messages"],
                {"role": "assistant", "content": "<answer>CCN</answer>"},
                {"role": "user", "content": "turn2"},
                {"role": "assistant", "content": "<answer>CCCl</answer>"},
            ],
            response_length=13,
            rollout_max_response_len=1024,
        )
    )
    assert stop_2 is True
    assert next_user_2 == ""
    assert reward_2 == pytest.approx(0.0)
    assert additional_2["trajectory_reward"] == pytest.approx(expected_settlement["trajectory_reward"])
    assert additional_2["trajectory_return"] == pytest.approx(expected_settlement["trajectory_reward"])
    assert additional_2["step_rewards"] == expected_settlement["step_rewards"]
    assert additional_2["reward_breakdowns"] == expected_settlement["breakdowns"]

    _run(interaction.finalize_interaction(instance_id))


def test_marco_interaction_force_terminate_sets_loop_limit_reason(monkeypatch):
    interaction = MarcoTrajectoryInteraction(config={})
    initial_state = {
        "episode_id": "ep-force",
        "subtask": "bbbp",
        "x0_smiles": "CCO",
        "current_smiles": "CCO",
        "property_targets": [{"name": "bbbp", "direction": "increase", "delta": 1.0, "role": "active"}],
        "x0_predictions": {"bbbp": 0.0},
        "current_predictions": {"bbbp": 0.0},
        "current_total_gap": 0.0,
        "current_directional_improvements": {"bbbp": 0.0},
        "current_progress_score": 0.0,
        "success_turn": None,
        "turn_id": 0,
        "max_turns": 4,
        "history": [],
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "turn1"},
        ],
    }
    scripted_result = {
        "turn_result": {
            "status": "ok",
            "should_stop": False,
            "met_all_targets": False,
            "similarity": 0.62,
            "progress_score": 0.2,
            "message": "continue",
            "error_detail": "",
        },
        "feedback_message": "feedback-force",
        "next_state": {**initial_state, "turn_id": 1},
        "next_user_content": "turn2",
        "invalid_streak": 0,
        "turn_payload": {
            "turn_id": 1,
            "max_turns": 4,
            "invalid_type": "ok",
            "similarity": 0.62,
            "progress_score": 0.2,
            "met_all_targets": False,
            "error_detail": "",
            "stop_reason": "continue",
            "response_length": 12,
            "response_text": "<answer>CCN</answer>",
        },
    }

    monkeypatch.setattr(marco_interaction, "evaluate_marco_response_turn", lambda **_: dict(scripted_result))

    instance_id = _run(interaction.start_interaction(instance_id="traj-force", initial_state_payload=initial_state))
    should_stop, next_user, reward, additional_data = _run(
        interaction.generate_response(
            instance_id,
            [
                *initial_state["messages"],
                {"role": "assistant", "content": "<answer>CCN</answer>"},
            ],
            response_length=12,
            rollout_max_response_len=1024,
            force_terminate=True,
        )
    )

    assert should_stop is True
    assert next_user == ""
    assert reward == pytest.approx(0.0)
    assert additional_data["turn_result"]["message"] == "agent_loop_limit"
    assert additional_data["turn_payload"]["stop_reason"] == "agent_loop_limit"

    _run(interaction.finalize_interaction(instance_id))
