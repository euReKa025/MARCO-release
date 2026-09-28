from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from marco.env.predictor_client import PredictorClient
from marco.utils import read_jsonl


def _property_targets(row: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for item in row.get("properties", []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if name:
            names.append(name)
    return names


def _property_source_map(row: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in row.get("properties", []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        source = item.get("source")
        if source is None:
            continue
        out[name] = float(source)
    return out


def check_recomputed_predictor_consistency(
    *,
    canonical_dir: str,
    split: str,
    sample_size: int,
    seed: int,
    subtask: str | None = None,
    predictor_client: PredictorClient | None = None,
) -> dict[str, Any]:
    canonical_path = Path(canonical_dir) / f"canonical_mumo_{split}.jsonl"
    rows = read_jsonl(canonical_path)
    if subtask is not None:
        rows = [row for row in rows if str(row.get("subtask", "")).strip() == subtask]

    rng = random.Random(seed)
    sampled_rows = list(rows)
    rng.shuffle(sampled_rows)
    sampled_rows = sampled_rows[: max(sample_size, 0)]

    client = predictor_client or PredictorClient()
    report_rows: list[dict[str, Any]] = []
    property_errors: dict[str, list[float]] = {}

    for row in sampled_rows:
        names = _property_targets(row)
        stored = _property_source_map(row)
        predicted = client.predict(str(row["x0_smiles"]), names)
        row_errors: dict[str, dict[str, float]] = {}
        for name in names:
            stored_source = float(stored[name])
            predicted_source = float(predicted[name])
            abs_error = abs(predicted_source - stored_source)
            row_errors[name] = {
                "stored_source": stored_source,
                "predicted_source": predicted_source,
                "abs_error": abs_error,
            }
            property_errors.setdefault(name, []).append(abs_error)
        report_rows.append(
            {
                "sample_id": row.get("sample_id"),
                "subtask": row.get("subtask"),
                "x0_smiles": row.get("x0_smiles"),
                "property_errors": row_errors,
            }
        )

    all_errors = [error for errors in property_errors.values() for error in errors]
    property_stats = {
        name: {
            "count": len(errors),
            "max_abs_error": max(errors) if errors else 0.0,
            "mean_abs_error": (sum(errors) / len(errors)) if errors else 0.0,
        }
        for name, errors in sorted(property_errors.items())
    }

    return {
        "canonical_jsonl": str(canonical_path),
        "split": split,
        "subtask": subtask,
        "seed": seed,
        "sample_size_requested": sample_size,
        "sample_size_actual": len(report_rows),
        "max_abs_error": max(all_errors) if all_errors else 0.0,
        "mean_abs_error": (sum(all_errors) / len(all_errors)) if all_errors else 0.0,
        "property_stats": property_stats,
        "rows": report_rows,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample recomputed canonical rows and compare stored source values against live predictor outputs.")
    parser.add_argument("--canonical-dir", default="data/canonical_recomputed")
    parser.add_argument("--split", default="test_seen")
    parser.add_argument("--sample-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subtask", default="")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    report = check_recomputed_predictor_consistency(
        canonical_dir=args.canonical_dir,
        split=args.split,
        sample_size=args.sample_size,
        seed=args.seed,
        subtask=args.subtask.strip() or None,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
