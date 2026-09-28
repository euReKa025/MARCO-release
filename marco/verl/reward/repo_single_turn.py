from __future__ import annotations

import json
from typing import Any

from marco.env.molecule_validation import Chem, validate_model_output
from marco.env.predictor_client import PredictorClient
from marco.env.similarity import tanimoto_similarity
from marco.utils import get_env_float, get_env_int

_PREDICTOR: PredictorClient | None = None


def _get_predictor() -> PredictorClient:
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = PredictorClient()
    return _PREDICTOR


def _required_properties(properties: Any) -> list[dict[str, Any]]:
    if not isinstance(properties, list):
        return []
    out: list[dict[str, Any]] = []
    for item in properties:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            out.append(item)
    return out


def _canonical_smiles(smiles: str) -> str:
    text = smiles.strip()
    if Chem is None:
        return text

    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return text
    return str(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))


def _is_exact_copy(candidate_smiles: str, x0_smiles: str) -> bool:
    return _canonical_smiles(candidate_smiles) == _canonical_smiles(x0_smiles)


def _is_improved(candidate: float, reference: float, direction: str, epsilon: float) -> bool:
    if direction == "decrease":
        return candidate < reference - epsilon
    return candidate > reference + epsilon


def _get_reference_values(
    predictor: PredictorClient,
    x0_smiles: str,
    properties: list[dict[str, Any]],
) -> dict[str, float]:
    required_names = [str(p["name"]) for p in properties]
    ref_values: dict[str, float] = {}
    missing: list[str] = []

    for p in properties:
        name = str(p["name"])
        source = p.get("source")
        if isinstance(source, (int, float)):
            ref_values[name] = float(source)
        else:
            missing.append(name)

    if missing:
        try:
            predicted = predictor.predict(x0_smiles, missing)
        except Exception:  # noqa: BLE001
            predicted = {name: 0.0 for name in missing}
        for key, value in predicted.items():
            ref_values[key] = float(value)

    return {k: ref_values[k] for k in required_names if k in ref_values}


def _reward_payload(
    *,
    status: str,
    score: float = 0.0,
    similarity: float = 0.0,
    improvement_score: float = 0.0,
    success: float = 0.0,
    constraint_success: float = 0.0,
    invalid: float = 0.0,
    similarity_threshold: float | None = None,
    improved_count: int = 0,
    required_count: int = 0,
    reason: str | None = None,
    predicted_values: dict[str, float] | None = None,
    directional_improvements: dict[str, bool] | None = None,
    exact_copy: bool = False,
    candidate_smiles: str = "",
    x0_smiles: str = "",
    sample_id: str = "",
    subtask: str = "",
) -> dict[str, Any]:
    predicted_values_payload = (
        json.dumps(predicted_values, sort_keys=True) if isinstance(predicted_values, dict) else "{}"
    )
    directional_improvements_payload = (
        json.dumps(directional_improvements, sort_keys=True)
        if isinstance(directional_improvements, dict)
        else "{}"
    )
    payload: dict[str, Any] = {
        "score": float(score),
        "status": status,
        "similarity": float(similarity),
        "improvement_score": float(improvement_score),
        "success": float(success),
        "constraint_success": float(constraint_success),
        "invalid": float(invalid),
        # Keep counts numeric for summaries, but store them as floats so verl's
        # numpy collation does not surface np.int64 values that JSONL dumping
        # cannot serialize.
        "improved_count": float(improved_count),
        "required_count": float(required_count),
        # Keep a stable reward_extra_info schema so agent-loop batch collation
        # does not fail when different samples surface different statuses.
        "similarity_threshold": float(similarity_threshold) if similarity_threshold is not None else float("nan"),
        "reason": "" if reason is None else str(reason),
        "predicted_values": predicted_values_payload,
        "directional_improvements": directional_improvements_payload,
        # Diagnostic-only fields. verl can dump reward_extra_info for training
        # rollouts; these fields do not participate in score computation.
        "exact_copy": 1.0 if exact_copy else 0.0,
        "candidate_smiles": str(candidate_smiles or ""),
        "x0_smiles": str(x0_smiles or ""),
        "sample_id": str(sample_id or ""),
        "subtask": str(subtask or ""),
    }
    return payload


def _ground_truth_fields(ground_truth: Any, extra_info: dict[str, Any]) -> tuple[str | None, list[dict[str, Any]]]:
    if isinstance(ground_truth, dict):
        x0_smiles = ground_truth.get("x0_smiles")
        properties = _required_properties(ground_truth.get("properties"))
        if isinstance(x0_smiles, str) and x0_smiles and properties:
            return x0_smiles, properties

    x0_from_extra = extra_info.get("x0_smiles")
    properties_from_extra = _required_properties(extra_info.get("properties"))
    if isinstance(x0_from_extra, str) and x0_from_extra and properties_from_extra:
        return x0_from_extra, properties_from_extra

    return None, []


def _missing_required_properties(required_names: list[str], values: dict[str, float]) -> list[str]:
    return [name for name in required_names if name not in values]


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None,
    **_: Any,
) -> dict[str, Any]:
    del data_source
    metadata = extra_info if isinstance(extra_info, dict) else {}
    threshold = get_env_float("MARCO_REPO_SIM_THRESHOLD", 0.5)
    improvement_epsilon = max(0.0, get_env_float("MARCO_REPO_IMPROVEMENT_EPSILON", 1e-5))
    exact_copy_score_zero = get_env_int("MARCO_REPO_EXACT_COPY_SCORE_ZERO", 0) != 0

    x0_smiles, properties = _ground_truth_fields(ground_truth, metadata)
    sample_id = str(metadata.get("sample_id") or "")
    subtask = str(metadata.get("subtask") or "")
    if not x0_smiles:
        return _reward_payload(
            status="invalid_metadata",
            reason="missing_x0_smiles",
            sample_id=sample_id,
            subtask=subtask,
        )
    if not properties:
        return _reward_payload(
            status="invalid_metadata",
            reason="missing_properties",
            x0_smiles=x0_smiles,
            sample_id=sample_id,
            subtask=subtask,
        )

    parsed = validate_model_output(solution_str)
    if parsed.status != "ok" or not parsed.smiles:
        return _reward_payload(
            status=str(parsed.status),
            similarity_threshold=threshold,
            invalid=1.0,
            x0_smiles=x0_smiles,
            sample_id=sample_id,
            subtask=subtask,
        )

    candidate_smiles = parsed.smiles
    exact_copy = _is_exact_copy(candidate_smiles, x0_smiles)
    try:
        similarity = float(tanimoto_similarity(candidate_smiles, x0_smiles))
    except Exception as exc:  # noqa: BLE001
        return _reward_payload(
            status="invalid_eval",
            reason=f"similarity_error:{exc}",
            invalid=1.0,
            exact_copy=exact_copy,
            candidate_smiles=candidate_smiles,
            x0_smiles=x0_smiles,
            sample_id=sample_id,
            subtask=subtask,
        )

    predictor = _get_predictor()
    required_names = [str(p["name"]) for p in properties]
    fallback_reason: str | None = None
    ref_values = _get_reference_values(predictor, x0_smiles, properties)
    try:
        cand_values = predictor.predict(candidate_smiles, required_names)
    except Exception as exc:  # noqa: BLE001
        cand_values = {name: 0.0 for name in required_names}
        fallback_reason = f"predictor_error:{exc}"

    missing_ref = _missing_required_properties(required_names, ref_values)
    if missing_ref:
        return _reward_payload(
            status="missing_property",
            score=0.0,
            similarity=similarity,
            similarity_threshold=threshold,
            reason=f"missing_ref_values:{','.join(sorted(missing_ref))}",
            exact_copy=exact_copy,
            candidate_smiles=candidate_smiles,
            x0_smiles=x0_smiles,
            sample_id=sample_id,
            subtask=subtask,
        )

    missing_cand = _missing_required_properties(required_names, cand_values)
    if missing_cand:
        return _reward_payload(
            status="missing_property",
            score=0.0,
            similarity=similarity,
            similarity_threshold=threshold,
            reason=f"missing_candidate_values:{','.join(sorted(missing_cand))}",
            exact_copy=exact_copy,
            candidate_smiles=candidate_smiles,
            x0_smiles=x0_smiles,
            sample_id=sample_id,
            subtask=subtask,
        )

    directional_improvements: dict[str, bool] = {}
    improved = 0
    for p in properties:
        name = str(p["name"])
        if name not in ref_values or name not in cand_values:
            continue
        direction = str(p.get("direction", "increase"))
        is_improved = False
        if not exact_copy:
            is_improved = _is_improved(
                float(cand_values[name]),
                float(ref_values[name]),
                direction,
                improvement_epsilon,
            )
        directional_improvements[name] = bool(is_improved)
        if is_improved:
            improved += 1

    required_count = max(1, len(required_names))
    improvement_score = improved / required_count
    property_success = float(improved >= required_count)
    constraint_success = float(property_success > 0.0 and similarity >= threshold)

    if similarity < threshold:
        return _reward_payload(
            status="below_similarity_threshold",
            score=0.0,
            similarity=similarity,
            improvement_score=improvement_score,
            success=property_success,
            constraint_success=constraint_success,
            similarity_threshold=threshold,
            improved_count=improved,
            required_count=required_count,
            predicted_values=cand_values,
            directional_improvements=directional_improvements,
            exact_copy=exact_copy,
            candidate_smiles=candidate_smiles,
            x0_smiles=x0_smiles,
            sample_id=sample_id,
            subtask=subtask,
        )

    score = 0.0 if exact_copy and exact_copy_score_zero else 0.5 * similarity + 0.5 * improvement_score
    return _reward_payload(
        status="predictor_error_fallback" if fallback_reason else "ok",
        score=score,
        similarity=similarity,
        improvement_score=improvement_score,
        success=property_success,
        constraint_success=constraint_success,
        similarity_threshold=threshold,
        improved_count=improved,
        required_count=required_count,
        reason=fallback_reason,
        predicted_values=cand_values,
        directional_improvements=directional_improvements,
        exact_copy=exact_copy,
        candidate_smiles=candidate_smiles,
        x0_smiles=x0_smiles,
        sample_id=sample_id,
        subtask=subtask,
    )
