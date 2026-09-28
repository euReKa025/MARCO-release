import pandas as pd
import pytest
from rdkit import DataStructs

from marco.eval.sws import fingerprint, score


def test_sws_excludes_canonical_copies_and_counts_failures(tmp_path):
    path = tmp_path / "details.csv"
    pd.DataFrame([
        ["CCO", "OCC", 1, 1, 1.0],
        ["CCO", "CCN", 1, 1, 0.3],
        ["CCO", "", 0, 0, 0.0],
    ], columns=["x0_smiles", "final_candidate_smiles", "success",
                "valid_selected_candidate", "final_similarity"]).to_csv(path, index=False)
    result = score(path, 3)
    similarity = DataStructs.TanimotoSimilarity(fingerprint("CCO"), fingerprint("CCN"))
    assert result["sws"] == pytest.approx(similarity / 3)
    assert result["sr"] == pytest.approx(2 / 3)
    assert result["sim"] == pytest.approx(0.65)
    assert result["successful_exact_copies"] == 1
    with pytest.raises(RuntimeError, match="expected 500 rows"):
        score(path)


def test_sws_requires_per_example_columns(tmp_path):
    path = tmp_path / "aggregate.csv"
    pd.DataFrame({"sr": [0.5]}).to_csv(path, index=False)
    with pytest.raises(RuntimeError, match="missing columns"):
        score(path, 1)
