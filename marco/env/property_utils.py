from __future__ import annotations

import os
from typing import Any

from marco.core_types import PropertyRole, PropertyTarget


def parse_property_targets(payload: dict[str, Any]) -> list[PropertyTarget]:
    active_names = set(task_property_names(str(payload.get("subtask", ""))))
    weight_overrides = _parse_property_weight_overrides(os.getenv("MARCO_PROPERTY_WEIGHTS"))
    targets: list[PropertyTarget] = []
    for item in payload.get("properties", []):
        if not isinstance(item, dict):
            continue
        delta = float(item.get("delta", 1e-6))
        name = str(item["name"])
        role = _normalize_role(item.get("role"), default=("active" if (not active_names or name in active_names) else "aux"))
        weight = _to_optional_float(item.get("weight"))
        if role == "active" and name in weight_overrides:
            weight = weight_overrides[name]
        progress_clip = _to_optional_float(item.get("progress_clip"))
        if weight is not None and weight <= 0.0:
            weight = None
        if progress_clip is not None and progress_clip <= 0.0:
            progress_clip = None
        targets.append(
            PropertyTarget(
                name=name,
                direction=str(item.get("direction", "increase")),
                delta=max(abs(delta), 1e-6),
                role=role,
                weight=weight,
                progress_clip=progress_clip,
                source=_to_optional_float(item.get("source")),
                target=_to_optional_float(item.get("target")),
            )
        )
    return targets


def _parse_property_weight_overrides(value: str | None) -> dict[str, float]:
    if value is None or not value.strip():
        return {}

    out: dict[str, float] = {}
    for chunk in value.split(","):
        text = chunk.strip()
        if not text or "=" not in text:
            continue
        name, raw_weight = text.split("=", 1)
        name = name.strip()
        if not name:
            continue
        weight = _to_optional_float(raw_weight.strip())
        if weight is None or weight <= 0.0:
            continue
        out[name] = float(weight)
    return out


def task_property_names(task: str) -> list[str]:
    return [p.strip() for p in str(task).split("+") if p.strip()]


def _to_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_role(value: Any, default: PropertyRole = "active") -> PropertyRole:
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"active", "aux"}:
            return v  # type: ignore[return-value]
    return default


def split_property_targets(targets: list[PropertyTarget]) -> tuple[list[PropertyTarget], list[PropertyTarget]]:
    active = [t for t in targets if t.role == "active"]
    aux = [t for t in targets if t.role != "active"]
    return active, aux


def compute_directional_improvements(
    *,
    predictions: dict[str, float],
    x0_predictions: dict[str, float],
    targets: list[PropertyTarget],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for t in targets:
        pred = float(predictions[t.name])
        x0 = float(x0_predictions[t.name])
        out[t.name] = float(t.signed_direction()) * (pred - x0)
    return out


def _resolve_active_weights(active_targets: list[PropertyTarget]) -> dict[str, float]:
    if not active_targets:
        return {}

    provided_sum = 0.0
    provided_count = 0
    for t in active_targets:
        if t.weight is not None and t.weight > 0.0:
            provided_sum += float(t.weight)
            provided_count += 1

    if provided_count == 0:
        uniform = 1.0 / float(len(active_targets))
        return {t.name: uniform for t in active_targets}

    missing = [t for t in active_targets if t.weight is None or t.weight <= 0.0]
    out: dict[str, float] = {}
    if provided_sum >= 1.0:
        # Normalize provided weights and assign zero to unspecified properties.
        for t in active_targets:
            if t.weight is not None and t.weight > 0.0:
                out[t.name] = float(t.weight) / provided_sum
            else:
                out[t.name] = 0.0
        return out

    remaining = max(0.0, 1.0 - provided_sum)
    fill = remaining / float(len(missing)) if missing else 0.0
    for t in active_targets:
        if t.weight is not None and t.weight > 0.0:
            out[t.name] = float(t.weight)
        else:
            out[t.name] = fill
    # Numerical safety normalization.
    total = sum(out.values())
    if total <= 0.0:
        uniform = 1.0 / float(len(active_targets))
        return {t.name: uniform for t in active_targets}
    return {k: v / total for k, v in out.items()}


def _resolve_progress_scale(*, target: PropertyTarget, default_clip: float) -> float:
    if float(target.delta) > 0.0:
        return max(abs(float(target.delta)), 1e-6)
    if target.progress_clip is not None and float(target.progress_clip) > 0.0:
        return max(abs(float(target.progress_clip)), 1e-6)
    return max(abs(float(default_clip)), 1e-6)


def compute_directional_progress_score(
    *,
    directional_improvements: dict[str, float],
    targets: list[PropertyTarget],
    default_clip: float,
) -> tuple[float, dict[str, float], bool]:
    active_targets, _ = split_property_targets(targets)
    if not active_targets:
        return 0.0, {}, False

    weights = _resolve_active_weights(active_targets)
    score = 0.0
    components: dict[str, float] = {}
    for t in active_targets:
        improvement = float(directional_improvements.get(t.name, 0.0))
        scale = _resolve_progress_scale(target=t, default_clip=default_clip)
        normalized = min(max(improvement / scale, -1.0), 1.0)
        components[t.name] = normalized
        score += float(weights[t.name]) * normalized

    bottleneck_weight = _env_float("MARCO_PROGRESS_BOTTLENECK_WEIGHT", 0.0)
    if bottleneck_weight > 0.0 and components:
        blend = min(max(bottleneck_weight, 0.0), 1.0)
        bottleneck_score = min(components.values())
        score = (1.0 - blend) * score + blend * bottleneck_score

    success = all(float(directional_improvements.get(t.name, 0.0)) > 0.0 for t in active_targets)
    return score, components, success


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def compute_property_gaps(
    *,
    predictions: dict[str, float],
    x0_predictions: dict[str, float],
    targets: list[PropertyTarget],
    eps: float,
    gap_max: float,
) -> tuple[dict[str, float], float, bool]:
    gaps: dict[str, float] = {}
    total_gap = 0.0

    for t in targets:
        pred = float(predictions[t.name])
        x0 = float(x0_predictions[t.name])
        direction = t.signed_direction()

        improvement = direction * (pred - x0)
        remaining = (t.delta - improvement) / (t.delta + eps)
        gap = max(0.0, min(gap_max, remaining))
        gaps[t.name] = gap
        total_gap += gap

    if targets:
        total_gap /= float(len(targets))

    met_all = all(v <= 0.0 for v in gaps.values()) if gaps else False
    return gaps, total_gap, met_all
