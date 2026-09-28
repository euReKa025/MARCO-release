from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from marco.utils import read_jsonl
from marco.prompts.system_prompt import PROMPT_MODE_THINK_ANSWER

from .schema import canonical_record_to_verl_row


def build_rlhf_parquet(
    *,
    input_jsonl: str,
    output_parquet: str,
    max_turns: int = 5,
    prompt_mode: str = PROMPT_MODE_THINK_ANSWER,
) -> None:
    rows = [canonical_record_to_verl_row(row, max_turns=max_turns, prompt_mode=prompt_mode) for row in read_jsonl(input_jsonl)]
    table = pa.Table.from_pylist(rows)
    out = Path(output_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-parquet", required=True)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--prompt-mode", default=PROMPT_MODE_THINK_ANSWER)
    args = parser.parse_args()
    build_rlhf_parquet(
        input_jsonl=args.input_jsonl,
        output_parquet=args.output_parquet,
        max_turns=args.max_turns,
        prompt_mode=args.prompt_mode,
    )


if __name__ == "__main__":
    main()
