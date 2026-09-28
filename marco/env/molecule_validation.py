from __future__ import annotations

from dataclasses import dataclass

from marco.prompts.answer_parser import parse_answer_smiles


try:
    from rdkit import Chem  # type: ignore
except Exception:  # pragma: no cover
    Chem = None


@dataclass
class ValidationResult:
    status: str
    smiles: str | None
    message: str = ""


def validate_candidate_smiles(smiles: str | None) -> ValidationResult:
    if not isinstance(smiles, str) or not smiles.strip():
        return ValidationResult(status="invalid_format", smiles=None, message="empty_smiles")

    candidate = smiles.strip()
    if "." in candidate:
        return ValidationResult(status="invalid_parse", smiles=candidate, message="multi_fragment_smiles")

    if Chem is None:
        # In CPU dev env without RDKit we still reject obvious disconnected outputs.
        return ValidationResult(status="ok", smiles=candidate)

    mol = Chem.MolFromSmiles(candidate)
    if mol is None:
        return ValidationResult(status="invalid_parse", smiles=candidate, message="MolFromSmiles failed")

    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        return ValidationResult(status="invalid_parse", smiles=candidate, message=str(exc))

    if len(Chem.GetMolFrags(mol)) != 1:
        return ValidationResult(status="invalid_parse", smiles=candidate, message="multi_fragment_smiles")

    return ValidationResult(status="ok", smiles=candidate)


def is_valid_candidate_smiles(smiles: str | None) -> bool:
    return validate_candidate_smiles(smiles).status == "ok"


def validate_model_output(output_text: str | None) -> ValidationResult:
    parsed = parse_answer_smiles(output_text)
    if parsed.status != "ok":
        return ValidationResult(status="invalid_format", smiles=None, message=parsed.reason)
    return validate_candidate_smiles(parsed.smiles)
