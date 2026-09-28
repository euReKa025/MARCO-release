from __future__ import annotations

import argparse
from pathlib import Path

from marco.utils import write_json
from marco.verl.validation.run_metadata import collect_run_metadata_from_env, parse_run_metadata_json
from marco.verl.validation.raw_summary import (
    build_raw_style_summary_from_snapshot,
    resolve_raw_canonical_jsonl,
)


def _infer_subtask(snapshot_json: str | Path) -> str | None:
    from marco.utils import read_json

    payload = read_json(snapshot_json)
    subtasks = {
        str(traj.get("subtask", "")).strip()
        for traj in list(payload.get("trajectories") or [])
        if str(traj.get("subtask", "")).strip()
    }
    if len(subtasks) == 1:
        return next(iter(subtasks))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build raw-source summary metrics from validation rollout snapshot.")
    parser.add_argument("--snapshot-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--subtask", default=None)
    parser.add_argument("--partition", default="seen")
    parser.add_argument("--validation-mode", default="sampled")
    parser.add_argument("--canonical-jsonl", default=None)
    parser.add_argument("--root-dir", default=Path.cwd().as_posix())
    parser.add_argument("--similarity-threshold", type=float, default=0.5)
    parser.add_argument("--similarity-target-low", type=float, default=None)
    parser.add_argument("--similarity-target-high", type=float, default=None)
    parser.add_argument("--similarity-copy-threshold", type=float, default=None)
    parser.add_argument("--run-metadata-json", default=None)
    parser.add_argument("--run-metadata-from-env", action="store_true")
    args = parser.parse_args()

    subtask = args.subtask or _infer_subtask(args.snapshot_json)
    canonical_jsonl = args.canonical_jsonl
    if canonical_jsonl is None and subtask:
        canonical_jsonl = resolve_raw_canonical_jsonl(
            root_dir=args.root_dir,
            subtask=subtask,
            partition=args.partition,
        ).as_posix()

    run_metadata = None
    if args.run_metadata_from_env:
        run_metadata = collect_run_metadata_from_env(metadata_json=args.run_metadata_json)
    elif args.run_metadata_json:
        run_metadata = parse_run_metadata_json(args.run_metadata_json)

    summary = build_raw_style_summary_from_snapshot(
        snapshot_json=args.snapshot_json,
        canonical_jsonl=canonical_jsonl,
        subtask=subtask,
        partition=str(args.partition),
        validation_mode=str(args.validation_mode),
        similarity_threshold=float(args.similarity_threshold),
        similarity_target_low=args.similarity_target_low,
        similarity_target_high=args.similarity_target_high,
        similarity_copy_threshold=args.similarity_copy_threshold,
        run_metadata=run_metadata,
    )
    if subtask:
        summary["subtask"] = subtask
    write_json(args.output_json, summary)


if __name__ == "__main__":
    main()
