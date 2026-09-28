from __future__ import annotations

import pytest

from marco.prompts.system_prompt import (
    DIRECT_SMILES_SYSTEM_PROMPT,
    PROMPT_MODE_DIRECT_SMILES,
    PROMPT_MODE_THINK_ANSWER,
    THINK_ANSWER_SYSTEM_PROMPT,
    build_system_user_messages,
    get_input_messages_key,
    get_system_prompt,
)


def test_prompt_registry_builds_system_user_messages_for_both_modes():
    direct = build_system_user_messages("Optimize CCO.", prompt_mode=PROMPT_MODE_DIRECT_SMILES)
    think = build_system_user_messages("Optimize CCO.", prompt_mode=PROMPT_MODE_THINK_ANSWER)

    assert direct == [
        {"role": "system", "content": DIRECT_SMILES_SYSTEM_PROMPT},
        {"role": "user", "content": "Optimize CCO."},
    ]
    assert think == [
        {"role": "system", "content": THINK_ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": "Optimize CCO."},
    ]
    assert get_system_prompt(PROMPT_MODE_DIRECT_SMILES) == DIRECT_SMILES_SYSTEM_PROMPT
    assert get_system_prompt(PROMPT_MODE_THINK_ANSWER) == THINK_ANSWER_SYSTEM_PROMPT
    assert get_input_messages_key(PROMPT_MODE_DIRECT_SMILES) == "input_messages_direct_smiles"
    assert get_input_messages_key(PROMPT_MODE_THINK_ANSWER) == "input_messages_think_answer"
    assert "<SMILES>" in THINK_ANSWER_SYSTEM_PROMPT
    assert "<answer>" not in THINK_ANSWER_SYSTEM_PROMPT.lower()


def test_prompt_registry_rejects_unknown_mode():
    with pytest.raises(ValueError):
        get_system_prompt("unknown")
