from __future__ import annotations

import argparse

from marco.utils import read_jsonl, write_json
from marco.verl.validation.eval_rollout import build_eval_rollout_snapshot
from marco.verl.validation.run_metadata import collect_run_metadata_from_env, parse_run_metadata_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build structured eval_rollout snapshot from validation JSONL rows.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--rollout-id", default=None)
    parser.add_argument("--subtask", default=None)
    parser.add_argument("--run-metadata-json", default=None)
    parser.add_argument("--run-metadata-from-env", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.input_jsonl)
    rollout_id = args.rollout_id or f"val-step-{args.step}"
    run_metadata = None
    if args.run_metadata_from_env:
        run_metadata = collect_run_metadata_from_env(metadata_json=args.run_metadata_json)
    elif args.run_metadata_json:
        run_metadata = parse_run_metadata_json(args.run_metadata_json)
    snapshot = build_eval_rollout_snapshot(
        rows=rows,
        step=int(args.step),
        rollout_id=rollout_id,
        default_subtask=str(args.subtask).strip() if args.subtask else None,
        run_metadata=run_metadata,
    )
    write_json(args.output_json, snapshot)


if __name__ == "__main__":
    main()
