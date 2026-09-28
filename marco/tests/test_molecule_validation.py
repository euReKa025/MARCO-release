from marco.env.molecule_validation import validate_model_output


def test_validate_model_output_accepts_single_fragment_smiles() -> None:
    result = validate_model_output("<SMILES>CCN</SMILES>")

    assert result.status == "ok"
    assert result.smiles == "CCN"


def test_validate_model_output_rejects_multi_fragment_smiles() -> None:
    result = validate_model_output("<SMILES>CCN.C</SMILES>")

    assert result.status == "invalid_parse"
    assert result.smiles == "CCN.C"
    assert "multi_fragment" in result.message
