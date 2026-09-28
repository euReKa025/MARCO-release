from marco.prompts.answer_parser import parse_answer_smiles


def test_parse_answer_tag_ok():
    result = parse_answer_smiles("<think>x</think><answer>CCO</answer>")
    assert result.status == "ok"
    assert result.smiles == "CCO"


def test_parse_smiles_tag_ok():
    result = parse_answer_smiles("<SMILES>CCN</SMILES>")
    assert result.status == "ok"
    assert result.smiles == "CCN"


def test_parse_invalid_multiple_candidates():
    result = parse_answer_smiles("CCO CCC")
    assert result.status == "invalid_format"


def test_parse_without_tags_is_invalid():
    result = parse_answer_smiles("CCO")
    assert result.status == "invalid_format"


def test_parse_multiple_tagged_candidates_uses_last_closed_tag():
    result = parse_answer_smiles(
        "<answer>CCO</answer>\n"
        "<think>revise</think>\n"
        "<SMILES>CCN</SMILES>"
    )
    assert result.status == "ok"
    assert result.smiles == "CCN"
