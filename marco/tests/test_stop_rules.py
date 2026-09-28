import marco.env.molecule_env as molecule_env_module
from marco.core_types import RewardConfig
from marco.env.molecule_env import MoleculeEnv
from marco.env.stop_rules import should_stop_episode


class _FakePredictor:
    def predict(self, smiles: str, required_properties):
        value = 0.0 if smiles == "CCO" else 2.0
        return {name: value for name in required_properties}


def _build_env_and_state():
    env = MoleculeEnv(
        predictor_client=_FakePredictor(),
        reward_config=RewardConfig(
            similarity_target_low=0.6,
            similarity_target_high=1.0,
        ),
    )
    record = {
        "sample_id": "s1",
        "subtask": "bbbp",
        "x0_smiles": "CCO",
        "properties": [{"name": "bbbp", "direction": "increase", "delta": 1.0, "role": "active"}],
    }
    state = env.init_episode(record=record, max_turns=4)
    return env, state


def test_invalid_does_not_stop_before_max_turns():
    result = should_stop_episode(
        turn_id=1,
        max_turns=4,
        invalid_type="invalid_format",
        met_all_targets=False,
        similarity_acceptable=False,
    )
    assert result.should_stop is False
    assert result.reason == "continue"


def test_success_stops_only_when_similarity_is_acceptable():
    result = should_stop_episode(
        turn_id=2,
        max_turns=4,
        invalid_type="ok",
        met_all_targets=True,
        similarity_acceptable=True,
    )
    assert result.should_stop is True
    assert result.reason == "success"


def test_target_hit_without_similarity_acceptance_does_not_stop():
    result = should_stop_episode(
        turn_id=2,
        max_turns=4,
        invalid_type="ok",
        met_all_targets=True,
        similarity_acceptable=False,
    )
    assert result.should_stop is False
    assert result.reason == "continue"


def test_step_stops_when_similarity_equals_target_low(monkeypatch):
    env, state = _build_env_and_state()
    monkeypatch.setattr(
        molecule_env_module,
        "tanimoto_similarity",
        lambda *_args, **_kwargs: float(env.cfg.similarity_target_low),
    )

    result = env.step(state=state, candidate_smiles="CCN", turn_id=1)

    assert result.met_all_targets is True
    assert result.should_stop is True
    assert result.message == "success"


def test_step_does_not_stop_when_similarity_equals_copy_threshold(monkeypatch):
    env, state = _build_env_and_state()
    monkeypatch.setattr(
        molecule_env_module,
        "tanimoto_similarity",
        lambda *_args, **_kwargs: 0.99,
    )

    result = env.step(state=state, candidate_smiles="CCN", turn_id=1)

    assert result.met_all_targets is True
    assert result.should_stop is False
    assert result.message == "continue"


def test_step_does_not_stop_when_similarity_is_above_copy_threshold(monkeypatch):
    env, state = _build_env_and_state()
    monkeypatch.setattr(
        molecule_env_module,
        "tanimoto_similarity",
        lambda *_args, **_kwargs: 1.0,
    )

    result = env.step(state=state, candidate_smiles="CCN", turn_id=1)

    assert result.met_all_targets is True
    assert result.should_stop is False
    assert result.message == "continue"
