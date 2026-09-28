#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem


RDLogger.DisableLog("rdApp.*")


def smiles_text(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


@lru_cache(maxsize=None)
def canonical(text: str) -> str:
    if not text:
        return ""
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


@lru_cache(maxsize=None)
def fingerprint(text: str):
    if not text:
        return None
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=2048,
        useChirality=False,
    )


def score(path: Path, expected_n: int = 500) -> dict[str, object]:
    if expected_n <= 0:
        raise ValueError("expected_n must be positive")
    df = pd.read_csv(path)
    required = {
        "x0_smiles",
        "final_candidate_smiles",
        "success",
        "valid_selected_candidate",
        "final_similarity",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{path}: missing columns {sorted(missing)}")
    if len(df) != expected_n:
        raise RuntimeError(f"{path}: expected {expected_n} rows, got {len(df)}")

    valid = df["valid_selected_candidate"].astype(float) > 0.5
    success = df["success"].astype(float) > 0.5
    exact_flags: list[bool] = []
    sws_sum = 0.0
    sws_positive = 0

    for row in df.itertuples(index=False):
        source = smiles_text(row.x0_smiles)
        candidate = smiles_text(row.final_candidate_smiles)
        source_canonical = canonical(source)
        candidate_canonical = canonical(candidate)
        exact = bool(
            source_canonical
            and candidate_canonical
            and source_canonical == candidate_canonical
        )
        exact_flags.append(exact)

        is_valid = float(row.valid_selected_candidate) > 0.5
        is_success = float(row.success) > 0.5
        if not (is_valid and is_success and not exact):
            continue

        source_fp = fingerprint(source)
        candidate_fp = fingerprint(candidate)
        if source_fp is None or candidate_fp is None:
            raise RuntimeError(
                f"{path}: valid successful row has an unparsable source/candidate"
            )
        sws_sum += float(DataStructs.TanimotoSimilarity(source_fp, candidate_fp))
        sws_positive += 1

    exact = pd.Series(exact_flags, index=df.index)
    sim = float(df.loc[valid, "final_similarity"].mean()) if valid.any() else 0.0
    sr = float(success.sum()) / expected_n

    return {
        "N": expected_n,
        "data_rows": len(df),
        "valid": int(valid.sum()),
        "property_success": int(success.sum()),
        "exact_copies": int(exact.sum()),
        "successful_exact_copies": int((valid & success & exact).sum()),
        "sws_positive": sws_positive,
        "sr": sr,
        "sim": sim,
        "sr_x_sim": sr * sim,
        "sws": sws_sum / expected_n,
        "detailed_csv": str(path),
    }



def main() -> None:
    parser = argparse.ArgumentParser(description="Compute strict SWS from per-example evaluations.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--expected-n", type=int, default=500)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    result = score(args.input_csv, args.expected_n)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
