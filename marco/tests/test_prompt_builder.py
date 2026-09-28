from __future__ import annotations

from marco.prompts.prompt_builder import build_turn_messages, build_turn_prompt
from marco.prompts.system_prompt import PROMPT_MODE_THINK_ANSWER, THINK_ANSWER_SYSTEM_PROMPT


def test_build_turn_messages_uses_system_role_for_think_answer_mode():
    prompt = build_turn_prompt(
        x0_smiles="CCO",
        current_smiles="CCN",
        property_targets=[{"name": "bbbp", "direction": "increase", "delta": 0.1}],
        turn_id=2,
        history=[],
        previous_feedback=None,
        prompt_mode=PROMPT_MODE_THINK_ANSWER,
    )
    messages = build_turn_messages(
        x0_smiles="CCO",
        current_smiles="CCN",
        property_targets=[{"name": "bbbp", "direction": "increase", "delta": 0.1}],
        turn_id=2,
        history=[],
        previous_feedback=None,
        prompt_mode=PROMPT_MODE_THINK_ANSWER,
    )

    assert messages[0] == {"role": "system", "content": THINK_ANSWER_SYSTEM_PROMPT}
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == prompt
    assert "Turn: 2" in prompt
    assert "Source molecule (x0): CCO" in prompt
    assert "bbbp: increase" in prompt
    assert "by at least" not in prompt
    assert "target range [0.60, 0.95]" in prompt
    assert "exactly one brief <think>...</think> block" in prompt
    assert "exactly one <SMILES>...</SMILES> block" in prompt
    assert "Return only <SMILES>SMILES</SMILES>." not in prompt


def test_build_turn_prompt_with_feedback_keeps_single_contract_and_brief_think():
    prompt = build_turn_prompt(
        x0_smiles="CCO",
        current_smiles="CCN",
        property_targets=[{"name": "bbbp", "direction": "increase", "delta": 0.1}],
        turn_id=3,
        history=[],
        previous_feedback={
            "predictions": {"bbbp": 0.4},
            "property_gaps": {},
            "directional_improvements": {"bbbp": 0.1},
            "progress_score": 0.1,
            "similarity": 0.8,
            "met_all_targets": False,
            "invalid_type": "ok",
            "error_message": "",
            "error_detail": "",
            "recovery_hint": "",
            "message": "",
        },
        prompt_mode=PROMPT_MODE_THINK_ANSWER,
    )

    assert prompt.count("exactly one brief <think>...</think> block") == 1
    assert prompt.count("exactly one <SMILES>...</SMILES> block") == 1
    assert "Keep the <think> block to one short sentence." in prompt
