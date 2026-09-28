from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from marco.env.predictor_client import PredictorClient
from marco.utils import read_jsonl, write_jsonl


def _candidate_containers(record: dict[str, Any]) -> list[tuple[dict[str, Any], str, list[dict[str, Any]]]]:
    out: list[tuple[dict[str, Any], str, list[dict[str, Any]]]] = []

    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        x0 = metadata.get("x0_smiles")
        props = metadata.get("properties")
        if isinstance(x0, str) and x0 and isinstance(props, list):
            valid_props = [p for p in props if isinstance(p, dict)]
            out.append((metadata, x0, valid_props))

    x0_top = record.get("x0_smiles")
    props_top = record.get("properties")
    if isinstance(x0_top, str) and x0_top and isinstance(props_top, list):
        valid_props_top = [p for p in props_top if isinstance(p, dict)]
        out.append((record, x0_top, valid_props_top))

    return out


def _names_from_properties(properties: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for prop in properties:
        name = prop.get("name")
        if not isinstance(name, str) or not name:
            continue
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def recompute_record_sources(
    record: dict[str, Any],
    predictor: PredictorClient,
    cache: dict[tuple[str, tuple[str, ...]], dict[str, float]],
) -> tuple[dict[str, Any], int]:
    new_record = copy.deepcopy(record)
    updated_properties = 0

    for container, x0_smiles, properties in _candidate_containers(new_record):
        names = _names_from_properties(properties)
        if not names:
            continue

        cache_key = (x0_smiles, tuple(sorted(names)))
        predicted = cache.get(cache_key)
        if predicted is None:
            predicted = predictor.predict(x0_smiles, names)
            cache[cache_key] = {k: float(v) for k, v in predicted.items()}

        for prop in properties:
            name = prop.get("name")
            if not isinstance(name, str) or name not in predicted:
                continue
            prop["source"] = float(predicted[name])
            updated_properties += 1

    return new_record, updated_properties


def build_recomputed_dataset(
    *,
    input_root: str | Path,
    output_root: str | Path,
    predictor: PredictorClient,
    allow_existing_output: bool = False,
) -> dict[str, Any]:
    in_root = Path(input_root)
    out_root = Path(output_root)

    if not in_root.exists():
        raise FileNotFoundError(f"Input root not found: {in_root}")
    if not in_root.is_dir():
        raise NotADirectoryError(f"Input root is not a directory: {in_root}")

    jsonl_files = sorted(p for p in in_root.rglob("*.jsonl") if p.is_file())
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found under: {in_root}")

    if out_root.exists() and any(out_root.rglob("*.jsonl")) and not allow_existing_output:
        raise FileExistsError(
            f"Output root already contains JSONL files: {out_root}. "
            "Use --allow-existing-output to overwrite generated files."
        )

    cache: dict[tuple[str, tuple[str, ...]], dict[str, float]] = {}
    files_processed = 0
    rows_processed = 0
    rows_updated = 0
    properties_updated = 0

    for src_path in jsonl_files:
        rel_path = src_path.relative_to(in_root)
        dst_path = out_root / rel_path

        rows = read_jsonl(src_path)
        out_rows: list[dict[str, Any]] = []

        file_rows_updated = 0
        file_properties_updated = 0
        for row in rows:
            rows_processed += 1
            new_row, updated_count = recompute_record_sources(row, predictor, cache)
            out_rows.append(new_row)
            if updated_count > 0:
                rows_updated += 1
                file_rows_updated += 1
                properties_updated += updated_count
                file_properties_updated += updated_count

        write_jsonl(dst_path, out_rows)
        files_processed += 1
        print(
            f"[recompute-source] {rel_path} rows={len(rows)} "
            f"updated_rows={file_rows_updated} updated_props={file_properties_updated}"
        )

    return {
        "input_root": str(in_root),
        "output_root": str(out_root),
        "files_processed": files_processed,
        "rows_processed": rows_processed,
        "rows_updated": rows_updated,
        "properties_updated": properties_updated,
        "unique_predictor_calls": len(cache),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute metadata.properties[].source (and top-level properties[].source) "
            "for JSONL datasets using predictor APIs, writing results to a new output tree."
        )
    )
    parser.add_argument("--input-root", default="data/baselines", help="Input directory containing JSONL files.")
    parser.add_argument(
        "--output-root",
        default="data/baselines_source_recomputed",
        help="Output directory. Input files are mirrored here; original data is untouched.",
    )
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Allow writing into an output directory that already contains JSONL files.",
    )
    parser.add_argument("--admet-url", default=None, help="Optional override for ADMET predictor endpoint.")
    parser.add_argument("--drd2-url", default=None, help="Optional override for DRD2 predictor endpoint.")
    parser.add_argument("--timeout-s", type=float, default=8.0, help="Predictor request timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=2, help="Predictor request retries per endpoint.")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    predictor = PredictorClient(
        admet_url=args.admet_url,
        drd2_url=args.drd2_url,
        timeout_s=float(args.timeout_s),
        max_retries=int(args.max_retries),
    )
    stats = build_recomputed_dataset(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        predictor=predictor,
        allow_existing_output=bool(args.allow_existing_output),
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
