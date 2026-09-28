from __future__ import annotations

from marco.prompts.feedback_formatter import format_feedback


def test_format_feedback_invalid_uses_think_answer_contract(monkeypatch):
    monkeypatch.setenv("MARCO_PROMPT_MODE", "think_answer")
    monkeypatch.setenv("MARCO_SIMILARITY_TARGET_LOW", "0.6")
    monkeypatch.setenv("MARCO_SIMILARITY_TARGET_HIGH", "0.75")

    message = format_feedback(
        predictions={},
        property_gaps={},
        directional_improvements={},
        progress_score=0.0,
        similarity=0.0,
        met_all_targets=False,
        invalid_type="invalid_format",
        error_message="no_tagged_candidate",
        error_detail="no_tagged_candidate",
        recovery_hint="",
        message="",
    )

    assert message.count("exactly one brief <think>...</think> block") == 1
    assert "exactly one valid molecule in exactly one <SMILES>...</SMILES> block" in message
    assert "Keep the <think> block to one short sentence." in message
    assert "multiple <SMILES> tags" in message
    assert "target_similarity_range: [0.60, 0.75]" in message


def test_format_feedback_ok_reports_similarity_without_appending_contract(monkeypatch):
    monkeypatch.setenv("MARCO_PROMPT_MODE", "think_answer")
    monkeypatch.setenv("MARCO_SIMILARITY_TARGET_LOW", "0.6")
    monkeypatch.setenv("MARCO_SIMILARITY_TARGET_HIGH", "0.75")

    message = format_feedback(
        predictions={"bbbp": 0.4},
        property_gaps={},
        directional_improvements={"bbbp": 0.1},
        progress_score=0.1,
        similarity=0.8,
        met_all_targets=False,
        invalid_type="ok",
    )

    assert "exactly one <think>...</think> block" not in message
    assert "exactly one <SMILES>...</SMILES> block" not in message
    assert "Please output exactly one molecule as <SMILES>SMILES</SMILES>." not in message
    assert "target_similarity_range: [0.60, 0.75]" in message
    assert "similarity_status: above" in message
    assert "similarity_distance_to_range: 0.050000" in message


def test_format_feedback_invalid_parse_multi_fragment_adds_specific_guidance(monkeypatch):
    monkeypatch.setenv("MARCO_PROMPT_MODE", "think_answer")
    monkeypatch.setenv("MARCO_SIMILARITY_TARGET_LOW", "0.6")
    monkeypatch.setenv("MARCO_SIMILARITY_TARGET_HIGH", "0.75")

    message = format_feedback(
        predictions={},
        property_gaps={},
        directional_improvements={},
        progress_score=0.0,
        similarity=0.0,
        met_all_targets=False,
        invalid_type="invalid_parse",
        error_message="multi_fragment_smiles",
        error_detail="multi_fragment_smiles",
        recovery_hint="",
        message="",
    )

    assert 'disconnected fragments separated by "."' in message
    assert "Return exactly one connected molecule." in message
    assert "Do not output salts, solvents, or extra fragments." in message
    assert "exactly one valid molecule in exactly one <SMILES>...</SMILES> block" in message
