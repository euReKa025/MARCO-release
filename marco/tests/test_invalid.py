from marco.reward.invalid import is_model_invalid


def test_invalid_types():
    assert is_model_invalid("invalid_format")
    assert is_model_invalid("invalid_parse")
    assert is_model_invalid("invalid_eval")
    assert not is_model_invalid("env_error")
    assert not is_model_invalid("ok")
