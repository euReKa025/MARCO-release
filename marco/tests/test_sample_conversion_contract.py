import pytest

from marco.rollout.sample_conversion import convert_samples_to_train_data
from marco.trainer_types import Sample


class Args:
    hf_checkpoint = None


def test_convert_samples_contract(monkeypatch):
    monkeypatch.setenv("MARCO_SIMILARITY_THRESHOLD", "0.4")
    monkeypatch.setenv("MARCO_SIMILARITY_CAP", "0.9")
    monkeypatch.setenv("MARCO_REWARD_OVER_SIM_PENALTY", "1.0")
    monkeypatch.setenv("MARCO_REWARD_ALPHA", "1.0")
    monkeypatch.setenv("MARCO_REWARD_LAMBDA_SIM", "0.5")

    s1 = Sample(
        index=1,
        tokens=[101, 102, 103],
        response="<answer>CCO</answer>",
        response_length=3,
        reward=0.0,
        status=Sample.Status.COMPLETED,
        train_metadata={
            "episode_id": "ep1",
            "trajectory_id": "traj1",
            "group_id": "g1",
            "turn_id": 1,
            "max_turns": 2,
            "invalid_type": "ok",
            "met_all_targets": False,
            "progress_score": 0.3,
            "similarity": 1.0,
        },
    )
    s2 = Sample(
        index=2,
        tokens=[101, 202, 203],
        response="<answer>CCC</answer>",
        response_length=3,
        reward=0.0,
        status=Sample.Status.TRUNCATED,
        train_metadata={
            "episode_id": "ep1",
            "trajectory_id": "traj1",
            "group_id": "g1",
            "turn_id": 2,
            "max_turns": 2,
            "invalid_type": "ok",
            "met_all_targets": False,
            "progress_score": 0.3,
            "similarity": 1.0,
        },
    )

    data = convert_samples_to_train_data(Args(), [s1, s2])

    required = {
        "tokens",
        "response_lengths",
        "rewards",
        "raw_reward",
        "truncated",
        "sample_indices",
        "loss_masks",
        "round_number",
        "metadata",
    }
    assert required.issubset(set(data.keys()))
    assert len(data["tokens"]) == 2
    assert data["truncated"] == [0, 1]
    assert data["raw_reward"] == [pytest.approx(-0.7), pytest.approx(-0.7)]
    assert data["rewards"] == [0.0, 0.0]
    assert data["metadata"][0]["trajectory_return"] == pytest.approx(-1.4)
    assert data["metadata"][1]["trajectory_return"] == pytest.approx(-1.4)
