from types import SimpleNamespace

from marco.core_types import RewardConfig
from marco.env.molecule_validation import ValidationResult
from marco.reward import reward_fn
from marco.reward.config import build_reward_config_from_env
from marco.reward.reward_fn import score_single_sample
from marco.trainer_types import Sample


def test_build_reward_config_from_env_uses_env_without_yaml(monkeypatch):
    monkeypatch.delenv("MARCO_REWARD_CONFIG", raising=False)
    monkeypatch.setenv("MARCO_REWARD_LAMBDA_SIM", "0.33")
    monkeypatch.setenv("MARCO_REWARD_IN_RANGE_SIM_BONUS", "0.44")

    cfg = build_reward_config_from_env()

    assert cfg.lambda_sim == 0.33
    assert cfg.in_range_sim_bonus == 0.44


def test_build_reward_config_from_env_applies_yaml_overrides(monkeypatch, tmp_path):
    cfg_path = tmp_path / "reward_config.yaml"
    cfg_path.write_text(
        "success_bonus: 9.0\nlambda_sim: 0.25\nturn_weight_temperature: 0.8\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("MARCO_REWARD_SUCCESS_BONUS", "4.0")
    monkeypatch.setenv("MARCO_REWARD_LAMBDA_SIM", "0.5")
    monkeypatch.setenv("MARCO_REWARD_CONFIG", str(cfg_path))

    cfg = build_reward_config_from_env()

    assert cfg.success_bonus == 9.0
    assert cfg.lambda_sim == 0.25
    assert cfg.turn_weight_temperature == 0.8


def test_reward_fn_get_env_uses_shared_reward_config_loader(monkeypatch):
    sentinel_cfg = RewardConfig(success_bonus=7.5)
    calls = {"count": 0}

    def _fake_build_reward_config():
        calls["count"] += 1
        return sentinel_cfg

    class FakeEnv:
        def __init__(self, client, cfg):
            self.client = client
            self.cfg = cfg

    monkeypatch.setattr(reward_fn, "_SHARED_ENV", None)
    monkeypatch.setattr(reward_fn, "build_reward_config_from_env", _fake_build_reward_config, raising=False)
    monkeypatch.setattr(reward_fn, "PredictorClient", lambda: object())
    monkeypatch.setattr(reward_fn, "MoleculeEnv", FakeEnv)

    env = reward_fn._get_env()

    assert calls["count"] == 1
    assert env.cfg is sentinel_cfg


def test_reward_missing_state_becomes_invalid_eval_penalty(monkeypatch):
    monkeypatch.setenv("MARCO_REWARD_INVALID_PENALTY", "1.0")

    sample = Sample(response="<answer>CCO</answer>", response_length=1, tokens=[1], metadata={})
    reward = score_single_sample(sample)

    assert reward == 0.0
    assert sample.metadata["turn_result"]["status"] == "invalid_eval"
    assert sample.metadata["turn_result"]["error_message"] != ""


def test_truncated_without_closed_answer_becomes_invalid_format():
    sample = Sample(
        response="<answer>CCO",
        response_length=4,
        tokens=[1, 2, 3, 4],
        status=Sample.Status.TRUNCATED,
        metadata={
            "turn_id": 1,
            "rl_state": {
                "episode_id": "ep1",
                "subtask": "bbbp+qed",
                "x0_smiles": "CCO",
                "current_smiles": "CCO",
                "property_targets": [{"name": "bbbp", "direction": "increase", "delta": 0.1, "role": "active"}],
                "x0_predictions": {"bbbp": 0.0},
                "current_predictions": {"bbbp": 0.0},
                "current_total_gap": 0.0,
                "current_directional_improvements": {"bbbp": 0.0},
                "current_progress_score": 0.0,
                "success_turn": None,
                "turn_id": 0,
                "max_turns": 1,
                "history": [],
            },
        },
    )
    reward = score_single_sample(sample)

    assert reward == 0.0
    assert sample.metadata["turn_result"]["status"] == "invalid_format"
    assert sample.metadata["turn_result"]["error_detail"] == "truncated_without_closed_answer_tag"
    assert sample.metadata["turn_result"]["similarity"] == 0.0
    assert sample.metadata["turn_result"]["progress_score"] == 0.0


def test_reward_appends_assistant_and_feedback_messages_for_next_turn(monkeypatch):
    class FakeEnv:
        def step(self, *, state, candidate_smiles: str, turn_id: int):
            return SimpleNamespace(
                status="ok",
                turn_id=turn_id,
                candidate_smiles=candidate_smiles,
                predictions={"bbbp": 0.2},
                property_gaps={},
                total_gap=0.0,
                similarity=0.85,
                met_all_targets=False,
                reward_total=0.0,
                reward_gap=0.0,
                reward_similarity=0.0,
                reward_success=0.0,
                should_stop=False,
                message="continue",
                error_message="",
                error_detail="",
                recovery_hint="",
                directional_improvements={"bbbp": 0.2},
                progress_components={"bbbp": 0.2},
                progress_score=0.2,
                active_property_names=["bbbp"],
                success_this_turn=False,
            )

        def apply_step(self, state, result):
            state.current_smiles = result.candidate_smiles
            state.current_predictions = dict(result.predictions)
            state.current_total_gap = float(result.total_gap)
            state.current_directional_improvements = dict(result.directional_improvements)
            state.current_progress_score = float(result.progress_score)
            state.turn_id = int(result.turn_id)
            return state

        def state_to_payload(self, state):
            return {
                "episode_id": state.episode_id,
                "subtask": state.subtask,
                "x0_smiles": state.x0_smiles,
                "current_smiles": state.current_smiles,
                "property_targets": [
                    {
                        "name": target.name,
                        "direction": target.direction,
                        "delta": target.delta,
                        "role": target.role,
                    }
                    for target in state.property_targets
                ],
                "x0_predictions": dict(state.x0_predictions),
                "current_predictions": dict(state.current_predictions),
                "current_total_gap": float(state.current_total_gap),
                "current_directional_improvements": dict(state.current_directional_improvements),
                "current_progress_score": float(state.current_progress_score),
                "success_turn": state.success_turn,
                "turn_id": state.turn_id,
                "max_turns": state.max_turns,
                "history": list(state.history),
            }

    monkeypatch.setattr("marco.reward.reward_fn._get_env", lambda: FakeEnv())
    monkeypatch.setattr(
        "marco.reward.reward_fn.validate_model_output",
        lambda text: ValidationResult(status="ok", smiles="CCN"),
    )

    sample = Sample(
        response="<answer>CCN</answer>",
        response_length=4,
        tokens=[1, 2, 3, 4],
        status=Sample.Status.COMPLETED,
        metadata={
            "turn_id": 1,
            "rl_state": {
                "episode_id": "ep1",
                "subtask": "bbbp",
                "x0_smiles": "CCO",
                "current_smiles": "CCO",
                "property_targets": [{"name": "bbbp", "direction": "increase", "delta": 0.1, "role": "active"}],
                "x0_predictions": {"bbbp": 0.0},
                "current_predictions": {"bbbp": 0.0},
                "current_total_gap": 0.0,
                "current_directional_improvements": {"bbbp": 0.0},
                "current_progress_score": 0.0,
                "success_turn": None,
                "turn_id": 0,
                "max_turns": 3,
                "history": [],
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "turn1"},
                ],
            },
        },
    )

    reward = score_single_sample(sample)

    assert reward == 0.0
    next_messages = sample.metadata["next_state"]["messages"]
    assert next_messages[0] == {"role": "system", "content": "sys"}
    assert next_messages[1] == {"role": "user", "content": "turn1"}
    assert next_messages[2] == {"role": "assistant", "content": "<answer>CCN</answer>"}
    assert next_messages[3]["role"] == "user"
    assert "Environment feedback:" in next_messages[3]["content"]
    assert sample.metadata["feedback_message"].startswith("Environment feedback:")
