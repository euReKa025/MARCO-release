import pytest

from marco.core_types import PropertyTarget
from marco.env.property_utils import compute_directional_progress_score, parse_property_targets
from marco.rollout.sample_conversion import convert_samples_to_train_data
from marco.trainer_types import Sample


class Args:
    hf_checkpoint = None


def _sample(
    idx: int,
    turn: int,
    status: str,
    progress_score: float,
    *,
    met_all: bool = False,
    similarity: float = 1.0,
    trajectory_id: str = "traj1",
) -> Sample:
    return Sample(
        index=idx,
        tokens=[10 + idx, 20 + idx],
        response="<answer>CCO</answer>",
        response_length=2,
        reward=0.0,
        status=Sample.Status.COMPLETED,
        train_metadata={
            "episode_id": "ep1",
            "trajectory_id": trajectory_id,
            "group_id": "group1",
            "turn_id": turn,
            "max_turns": 5,
            "invalid_type": status,
            "met_all_targets": met_all,
            "progress_score": progress_score,
            "similarity": similarity,
        },
    )


def _set_quality_trend_defaults(monkeypatch) -> None:
    monkeypatch.setenv("MARCO_REWARD_INVALID_PENALTY", "1.0")
    monkeypatch.setenv("MARCO_SIMILARITY_TARGET_LOW", "0.6")
    monkeypatch.setenv("MARCO_SIMILARITY_TARGET_HIGH", "0.75")
    monkeypatch.setenv("MARCO_PROGRESS_MIN_SIMILARITY", "0.3")
    monkeypatch.setenv("MARCO_SIMILARITY_COPY_THRESHOLD", "0.99")
    monkeypatch.setenv("MARCO_SIMILARITY_LOW_WEIGHT", "1.0")
    monkeypatch.setenv("MARCO_SIMILARITY_COPY_PENALTY_WEIGHT", "2.0")
    monkeypatch.setenv("MARCO_STATE_PROP_WEIGHT", "1.0")
    monkeypatch.setenv("MARCO_STATE_SIM_WEIGHT", "1.0")
    monkeypatch.setenv("MARCO_QUALITY_SUCCESS_BONUS", "0.15")
    monkeypatch.setenv("MARCO_TREND_IMPROVE_WEIGHT", "0.5")
    monkeypatch.setenv("MARCO_TREND_REGRESS_WEIGHT", "0.75")
    monkeypatch.setenv("MARCO_TREND_IMPROVE_MARGIN", "0.02")
    monkeypatch.setenv("MARCO_TREND_REGRESS_MARGIN", "0.02")


def test_similarity_quality_rises_then_drops_near_copy_threshold(monkeypatch):
    _set_quality_trend_defaults(monkeypatch)
    monkeypatch.setenv("MARCO_STATE_PROP_WEIGHT", "0.0")
    monkeypatch.setenv("MARCO_STATE_SIM_WEIGHT", "1.0")
    monkeypatch.setenv("MARCO_QUALITY_SUCCESS_BONUS", "0.0")
    monkeypatch.setenv("MARCO_TREND_IMPROVE_WEIGHT", "0.0")
    monkeypatch.setenv("MARCO_TREND_REGRESS_WEIGHT", "0.0")

    samples = [
        _sample(1, 1, "ok", 0.0, similarity=0.65),
        _sample(2, 2, "ok", 0.0, similarity=0.9),
        _sample(3, 3, "ok", 0.0, similarity=0.995),
    ]

    data = convert_samples_to_train_data(Args(), samples)

    assert data["raw_reward"] == pytest.approx([0.1282051282, 0.7692307692, 0.0])
    assert data["metadata"][0]["reward_similarity"] == pytest.approx(0.1282051282)
    assert data["metadata"][1]["reward_similarity"] == pytest.approx(0.7692307692)
    assert data["metadata"][2]["reward_similarity"] == pytest.approx(0.0)
    assert data["raw_reward"][1] > data["raw_reward"][0]
    assert data["raw_reward"][2] < data["raw_reward"][1]


def test_invalid_and_env_error_turn_rewards_are_fixed(monkeypatch):
    _set_quality_trend_defaults(monkeypatch)
    monkeypatch.setenv("MARCO_REWARD_INVALID_PENALTY", "1.0")

    samples = [
        _sample(1, 1, "invalid_parse", 1.0, met_all=True, similarity=0.9),
        _sample(2, 2, "env_error", 1.0, met_all=True, similarity=0.9),
    ]

    data = convert_samples_to_train_data(Args(), samples)

    invalid_md = data["metadata"][0]
    env_error_md = data["metadata"][1]

    assert data["raw_reward"][0] == pytest.approx(-1.0)
    assert invalid_md["reward_total"] == pytest.approx(-1.0)
    assert invalid_md["reward_invalid"] == pytest.approx(-1.0)
    assert invalid_md["reward_state_quality"] == pytest.approx(0.0)
    assert invalid_md["reward_improve"] == pytest.approx(0.0)
    assert invalid_md["reward_regress"] == pytest.approx(0.0)

    assert data["raw_reward"][1] == pytest.approx(0.0)
    assert env_error_md["reward_total"] == pytest.approx(0.0)
    assert env_error_md["reward_invalid"] == pytest.approx(0.0)
    assert env_error_md["reward_state_quality"] == pytest.approx(0.0)
    assert env_error_md["reward_improve"] == pytest.approx(0.0)
    assert env_error_md["reward_regress"] == pytest.approx(0.0)
    assert env_error_md["skip_training"] is True


def test_terminal_failure_and_old_success_contributions_are_inactive(monkeypatch):
    _set_quality_trend_defaults(monkeypatch)
    monkeypatch.setenv("MARCO_TERMINAL_Q_WEIGHT", "9.0")
    monkeypatch.setenv("MARCO_REWARD_FAILURE_PENALTY", "9.0")
    monkeypatch.setenv("MARCO_LENGTH_PENALTY_THRESHOLD", "1")
    monkeypatch.setenv("MARCO_LENGTH_PENALTY_VALUE", "3.0")
    monkeypatch.setenv("ROLLOUT_MAX_RESPONSE_LEN", "2")
    monkeypatch.setenv("MARCO_REWARD_SUCCESS_BONUS", "7.0")

    s1 = _sample(1, 1, "ok", 0.1, met_all=False, similarity=0.8)
    s2 = _sample(2, 2, "ok", 0.1, met_all=False, similarity=0.8)
    s1.response_length = 10
    s2.response_length = 10

    data = convert_samples_to_train_data(Args(), [s1, s2])

    for md in data["metadata"]:
        assert md["reward_terminal_q"] == pytest.approx(0.0)
        assert md["reward_failure"] == pytest.approx(0.0)
        assert md["reward_length"] == pytest.approx(-3.0)
        assert md["reward_success"] == pytest.approx(0.0)


def test_length_penalty_reduces_qt_and_total_reward(monkeypatch):
    _set_quality_trend_defaults(monkeypatch)
    monkeypatch.setenv("MARCO_STATE_PROP_WEIGHT", "1.0")
    monkeypatch.setenv("MARCO_STATE_SIM_WEIGHT", "0.0")
    monkeypatch.setenv("MARCO_QUALITY_SUCCESS_BONUS", "0.0")
    monkeypatch.setenv("MARCO_TREND_IMPROVE_WEIGHT", "0.0")
    monkeypatch.setenv("MARCO_TREND_REGRESS_WEIGHT", "0.0")
    monkeypatch.setenv("MARCO_LENGTH_PENALTY_THRESHOLD", "4")
    monkeypatch.setenv("MARCO_LENGTH_PENALTY_VALUE", "0.3")

    sample = _sample(1, 1, "ok", 0.5, met_all=False, similarity=0.7)
    sample.response_length = 10

    data = convert_samples_to_train_data(Args(), [sample])
    md = data["metadata"][0]

    assert md["reward_length"] == pytest.approx(-0.3)
    assert md["state_quality_score"] == pytest.approx(0.2)
    assert md["reward_state_quality"] == pytest.approx(0.2)
    assert data["raw_reward"][0] == pytest.approx(0.2)


def test_copy_threshold_reward_uses_near_copy_penalty_ramp(monkeypatch):
    _set_quality_trend_defaults(monkeypatch)
    monkeypatch.setenv("MARCO_STATE_PROP_WEIGHT", "0.0")
    monkeypatch.setenv("MARCO_STATE_SIM_WEIGHT", "1.0")
    monkeypatch.setenv("MARCO_QUALITY_SUCCESS_BONUS", "0.0")
    monkeypatch.setenv("MARCO_TREND_IMPROVE_WEIGHT", "0.0")
    monkeypatch.setenv("MARCO_TREND_REGRESS_WEIGHT", "0.0")

    samples = [
        _sample(1, 1, "ok", 0.0, similarity=0.995),
        _sample(2, 2, "ok", 0.0, similarity=1.0),
    ]

    data = convert_samples_to_train_data(Args(), samples)

    assert data["metadata"][0]["reward_similarity"] == pytest.approx(0.0)
    assert data["metadata"][1]["reward_similarity"] == pytest.approx(-1.0)
    assert data["raw_reward"] == pytest.approx([0.0, -1.0])


def test_quality_success_bonus_gates_on_targets_and_similarity_acceptance(monkeypatch):
    _set_quality_trend_defaults(monkeypatch)
    monkeypatch.setenv("MARCO_STATE_PROP_WEIGHT", "0.0")
    monkeypatch.setenv("MARCO_STATE_SIM_WEIGHT", "0.0")
    monkeypatch.setenv("MARCO_QUALITY_SUCCESS_BONUS", "0.25")
    monkeypatch.setenv("MARCO_TREND_IMPROVE_WEIGHT", "0.0")
    monkeypatch.setenv("MARCO_TREND_REGRESS_WEIGHT", "0.0")
    monkeypatch.setenv("MARCO_SIMILARITY_TARGET_LOW", "0.6")
    monkeypatch.setenv("MARCO_SIMILARITY_COPY_THRESHOLD", "0.99")

    samples = [
        _sample(1, 1, "ok", 0.0, met_all=True, similarity=0.59),
        _sample(2, 2, "ok", 0.0, met_all=True, similarity=0.8),
        _sample(3, 3, "ok", 0.0, met_all=True, similarity=0.99),
    ]

    data = convert_samples_to_train_data(Args(), samples)

    assert data["metadata"][0]["reward_success_state"] == pytest.approx(0.0)
    assert data["metadata"][1]["reward_success_state"] == pytest.approx(0.25)
    assert data["metadata"][2]["reward_success_state"] == pytest.approx(0.0)
    assert data["metadata"][0]["state_quality_score"] == pytest.approx(0.0)
    assert data["metadata"][1]["state_quality_score"] == pytest.approx(0.25)
    assert data["metadata"][2]["state_quality_score"] == pytest.approx(0.0)
    assert data["raw_reward"] == pytest.approx([0.0, 0.25, 0.0])


def test_posthoc_reward_uses_quality_plus_trend_terms(monkeypatch):
    _set_quality_trend_defaults(monkeypatch)
    monkeypatch.setenv("MARCO_STATE_PROP_WEIGHT", "1.0")
    monkeypatch.setenv("MARCO_STATE_SIM_WEIGHT", "0.0")
    monkeypatch.setenv("MARCO_QUALITY_SUCCESS_BONUS", "0.0")
    monkeypatch.setenv("MARCO_TREND_IMPROVE_WEIGHT", "1.0")
    monkeypatch.setenv("MARCO_TREND_REGRESS_WEIGHT", "1.0")
    monkeypatch.setenv("MARCO_TREND_IMPROVE_MARGIN", "0.02")
    monkeypatch.setenv("MARCO_TREND_REGRESS_MARGIN", "0.02")

    samples = [
        _sample(1, 1, "ok", 0.2, met_all=False, similarity=0.7),
        _sample(2, 2, "ok", 0.5, met_all=False, similarity=0.7),
        _sample(3, 3, "ok", 0.4, met_all=False, similarity=0.7),
    ]

    data = convert_samples_to_train_data(Args(), samples)

    assert data["raw_reward"] == pytest.approx([0.38, 0.78, 0.32])
    assert data["metadata"][0]["reward_state_quality"] == pytest.approx(0.2)
    assert data["metadata"][0]["reward_improve"] == pytest.approx(0.18)
    assert data["metadata"][1]["reward_state_quality"] == pytest.approx(0.5)
    assert data["metadata"][1]["reward_improve"] == pytest.approx(0.28)
    assert data["metadata"][2]["reward_regress"] == pytest.approx(-0.08)
    assert data["metadata"][2]["state_quality_score"] == pytest.approx(0.4)
    assert data["metadata"][0]["trajectory_reward"] == pytest.approx(1.48)


def test_best_quality_turn_filter_masks_non_best_turns(monkeypatch):
    _set_quality_trend_defaults(monkeypatch)
    monkeypatch.setenv("MARCO_STATE_PROP_WEIGHT", "1.0")
    monkeypatch.setenv("MARCO_STATE_SIM_WEIGHT", "0.0")
    monkeypatch.setenv("MARCO_QUALITY_SUCCESS_BONUS", "0.0")
    monkeypatch.setenv("MARCO_TREND_IMPROVE_WEIGHT", "0.0")
    monkeypatch.setenv("MARCO_TREND_REGRESS_WEIGHT", "0.0")
    monkeypatch.setenv("MARCO_TRAIN_TURN_FILTER", "best_quality")

    samples = [
        _sample(1, 1, "ok", 0.2, met_all=True, similarity=0.4),
        _sample(2, 2, "ok", 0.8, met_all=True, similarity=0.7),
        _sample(3, 3, "ok", 0.3, met_all=False, similarity=0.8),
    ]

    data = convert_samples_to_train_data(Args(), samples)

    assert data["metadata"][0]["is_best_quality_turn"] is False
    assert data["metadata"][1]["is_best_quality_turn"] is True
    assert data["metadata"][2]["is_best_quality_turn"] is False
    assert data["metadata"][0]["turn_filter_masked"] is True
    assert data["metadata"][1]["turn_filter_masked"] is False
    assert data["metadata"][2]["turn_filter_masked"] is True
    assert data["loss_masks"][0] == [0, 0]
    assert data["loss_masks"][1] == [1, 1]
    assert data["loss_masks"][2] == [0, 0]
    assert data["rewards"][0] == pytest.approx(0.0)
    assert data["rewards"][2] == pytest.approx(0.0)


def test_directional_progress_score_normalizes_by_target_delta():
    targets = [
        PropertyTarget(name="bbbp", direction="increase", delta=0.5),
        PropertyTarget(name="drd2", direction="increase", delta=0.2),
    ]

    score, components, success = compute_directional_progress_score(
        directional_improvements={"bbbp": 0.25, "drd2": 0.2},
        targets=targets,
        default_clip=1.0,
    )

    assert score == pytest.approx(0.75)
    assert components["bbbp"] == pytest.approx(0.5)
    assert components["drd2"] == pytest.approx(1.0)
    assert success is True


def test_property_weight_env_overrides_progress_weights(monkeypatch):
    monkeypatch.setenv("MARCO_PROPERTY_WEIGHTS", "drd2=0.6")
    targets = parse_property_targets(
        {
            "subtask": "bbbp+drd2+plogp",
            "properties": [
                {"name": "bbbp", "direction": "increase"},
                {"name": "drd2", "direction": "increase"},
                {"name": "plogp", "direction": "increase"},
            ],
        }
    )

    score, components, success = compute_directional_progress_score(
        directional_improvements={"bbbp": 1.0, "drd2": 0.0, "plogp": 0.0},
        targets=targets,
        default_clip=1.0,
    )

    assert score == pytest.approx(0.2)
    assert components == {"bbbp": 1.0, "drd2": 0.0, "plogp": 0.0}
    assert {target.name: target.weight for target in targets} == {
        "bbbp": None,
        "drd2": 0.6,
        "plogp": None,
    }
    assert success is False


def test_progress_bottleneck_weight_pulls_score_toward_weakest_property(monkeypatch):
    targets = [
        PropertyTarget(name="bbbp", direction="increase", delta=1.0),
        PropertyTarget(name="drd2", direction="increase", delta=1.0),
        PropertyTarget(name="plogp", direction="increase", delta=1.0),
    ]

    default_score, default_components, default_success = compute_directional_progress_score(
        directional_improvements={"bbbp": 1.0, "drd2": 0.0, "plogp": 1.0},
        targets=targets,
        default_clip=1.0,
    )

    monkeypatch.setenv("MARCO_PROGRESS_BOTTLENECK_WEIGHT", "0.5")
    bottleneck_score, bottleneck_components, bottleneck_success = compute_directional_progress_score(
        directional_improvements={"bbbp": 1.0, "drd2": 0.0, "plogp": 1.0},
        targets=targets,
        default_clip=1.0,
    )

    assert default_score == pytest.approx(2.0 / 3.0)
    assert default_components == {"bbbp": 1.0, "drd2": 0.0, "plogp": 1.0}
    assert default_success is False
    assert bottleneck_components == default_components
    assert bottleneck_score == pytest.approx(1.0 / 3.0)
    assert bottleneck_success is False
