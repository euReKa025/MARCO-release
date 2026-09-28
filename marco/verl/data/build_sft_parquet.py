from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from marco.utils import read_jsonl


def _to_sft_row(row: dict[str, Any], *, row_idx: int) -> dict[str, Any]:
    messages = row.get("messages")
    metadata = row.get("metadata", {})
    if not isinstance(messages, list):
        raise ValueError(f"invalid aligned SFT row at index {row_idx}: messages must be a list")
    if not isinstance(metadata, dict):
        raise ValueError(f"invalid aligned SFT row at index {row_idx}: metadata must be a dict")
    return {
        "messages": messages,
        "metadata": metadata,
    }


def build_sft_parquet(*, input_jsonl: str, output_parquet: str) -> None:
    raw_rows = read_jsonl(input_jsonl)
    if not raw_rows:
        raise ValueError(f"aligned SFT input JSONL is empty: {input_jsonl}")
    rows = [_to_sft_row(row, row_idx=row_idx) for row_idx, row in enumerate(raw_rows)]
    table = pa.Table.from_pylist(rows)
    output_path = Path(output_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-parquet", required=True)
    args = parser.parse_args()
    build_sft_parquet(input_jsonl=args.input_jsonl, output_parquet=args.output_parquet)


if __name__ == "__main__":
    main()
