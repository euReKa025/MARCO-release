from __future__ import annotations

from marco.errors import InvalidEvaluationError


try:
    from rdkit import Chem, DataStructs  # type: ignore
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator  # type: ignore
except Exception:  # pragma: no cover
    Chem = None
    DataStructs = None
    GetMorganGenerator = None


def tanimoto_similarity(smiles_a: str, smiles_b: str) -> float:
    if Chem is None or DataStructs is None or GetMorganGenerator is None:
        raise InvalidEvaluationError("RDKit is required for similarity computation")

    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    if mol_a is None or mol_b is None:
        raise InvalidEvaluationError("Cannot parse molecules for similarity")

    fp_gen = GetMorganGenerator(radius=2, fpSize=2048)
    fp_a = fp_gen.GetFingerprint(mol_a)
    fp_b = fp_gen.GetFingerprint(mol_b)
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))
