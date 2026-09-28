from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from marco.data.align_sft_think_data import align_sft_think_data
from marco.data.build_baseline_data import build_baseline_data
from marco.data.build_canonical_mumo_jsonl import build_canonical
from marco.prompts.system_prompt import PROMPT_MODE_THINK_ANSWER
from marco.verl.data.build_grpo_baseline_parquet import build_grpo_baseline_parquet
from marco.verl.data.build_rlhf_parquet import build_rlhf_parquet
from marco.verl.data.build_sft_parquet import build_sft_parquet


RL_SPLITS = ("train", "val", "test", "test_seen", "test_unseen")


def _iter_subtask_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir())


def _baseline_jobs(baseline_root: Path, parquet_root: Path, *, prompt_mode: str) -> dict[str, Any]:
    merged: dict[str, str] = {}
    for split in RL_SPLITS:
        src = baseline_root / "grpo_single_turn" / f"grpo_{split}.jsonl"
        dst = parquet_root / f"{split}.parquet"
        if src.exists():
            build_grpo_baseline_parquet(input_jsonl=str(src), output_parquet=str(dst), prompt_mode=prompt_mode)
            merged[split] = str(dst)

    by_subtask: dict[str, dict[str, str]] = {}
    subtask_root = baseline_root / "grpo_single_turn" / "by_subtask"
    for subdir in _iter_subtask_dirs(subtask_root):
        split_outputs: dict[str, str] = {}
        for split in RL_SPLITS:
            src = subdir / f"grpo_{split}.jsonl"
            if not src.exists():
                continue
            dst = parquet_root / "by_subtask" / subdir.name / f"{split}.parquet"
            build_grpo_baseline_parquet(input_jsonl=str(src), output_parquet=str(dst), prompt_mode=prompt_mode)
            split_outputs[split] = str(dst)
        if split_outputs:
            by_subtask[subdir.name] = split_outputs

    return {"merged": merged, "by_subtask": by_subtask}


def _rlhf_jobs(
    canonical_root: Path,
    parquet_root: Path,
    *,
    max_turns: int,
    prompt_mode: str = PROMPT_MODE_THINK_ANSWER,
) -> dict[str, Any]:
    merged: dict[str, str] = {}
    for split in RL_SPLITS:
        src = canonical_root / f"canonical_mumo_{split}.jsonl"
        dst = parquet_root / f"{split}.parquet"
        if src.exists():
            build_rlhf_parquet(input_jsonl=str(src), output_parquet=str(dst), max_turns=max_turns, prompt_mode=prompt_mode)
            merged[split] = str(dst)

    by_subtask: dict[str, dict[str, str]] = {}
    subtask_root = canonical_root / "by_subtask"
    for subdir in _iter_subtask_dirs(subtask_root):
        split_outputs: dict[str, str] = {}
        for split in RL_SPLITS:
            src = subdir / f"canonical_mumo_{split}.jsonl"
            if not src.exists():
                continue
            dst = parquet_root / "by_subtask" / subdir.name / f"{split}.parquet"
            build_rlhf_parquet(input_jsonl=str(src), output_parquet=str(dst), max_turns=max_turns, prompt_mode=prompt_mode)
            split_outputs[split] = str(dst)
        if split_outputs:
            by_subtask[subdir.name] = split_outputs

    return {"merged": merged, "by_subtask": by_subtask}


def _sft_jobs(aligned_root: Path, parquet_root: Path) -> dict[str, Any]:
    train_src = aligned_root / "sft_train.jsonl"
    train_dst = parquet_root / "train.parquet"
    build_sft_parquet(input_jsonl=str(train_src), output_parquet=str(train_dst))

    by_subtask: dict[str, dict[str, str]] = {}
    subtask_root = aligned_root / "by_subtask"
    for subdir in _iter_subtask_dirs(subtask_root):
        src = subdir / "sft_train.jsonl"
        if not src.exists():
            continue
        dst = parquet_root / "by_subtask" / subdir.name / "train.parquet"
        build_sft_parquet(input_jsonl=str(src), output_parquet=str(dst))
        by_subtask[subdir.name] = {"train": str(dst)}

    return {"merged": {"train": str(train_dst)}, "by_subtask": by_subtask}


def build_repo_source_datasets(
    *,
    repo_data_root: str,
    sft_think_dir: str,
    canonical_output_dir: str,
    baseline_jsonl_output_dir: str,
    sft_aligned_output_dir: str,
    sft_parquet_output_dir: str,
    baseline_parquet_output_dir: str,
    rlhf_parquet_output_dir: str,
    baseline_prompt_mode: str = PROMPT_MODE_THINK_ANSWER,
    rlhf_prompt_mode: str = PROMPT_MODE_THINK_ANSWER,
    max_turns: int = 5,
) -> dict[str, Any]:
    canonical_outputs = build_canonical(repo_data_root=repo_data_root, output_dir=canonical_output_dir)
    baseline_jsonl_outputs = build_baseline_data(canonical_dir=canonical_output_dir, output_dir=baseline_jsonl_output_dir)
    sft_aligned_outputs = align_sft_think_data(
        sft_think_dir=sft_think_dir,
        canonical_dir=canonical_output_dir,
        output_dir=sft_aligned_output_dir,
    )

    baseline_parquet_outputs = _baseline_jobs(
        Path(baseline_jsonl_output_dir),
        Path(baseline_parquet_output_dir),
        prompt_mode=baseline_prompt_mode,
    )
    rlhf_parquet_outputs = _rlhf_jobs(
        Path(canonical_output_dir),
        Path(rlhf_parquet_output_dir),
        max_turns=max_turns,
        prompt_mode=rlhf_prompt_mode,
    )
    sft_parquet_outputs = _sft_jobs(Path(sft_aligned_output_dir), Path(sft_parquet_output_dir))

    return {
        "canonical": canonical_outputs,
        "baseline_jsonl": baseline_jsonl_outputs,
        "sft_aligned": sft_aligned_outputs,
        "baseline_parquet": baseline_parquet_outputs,
        "rlhf_parquet": rlhf_parquet_outputs,
        "sft_parquet": sft_parquet_outputs,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build canonical, baseline JSONL, aligned SFT, and verl parquet datasets from raw RePO sources.")
    parser.add_argument(
        "--repo-data-root",
        default="../RePO/data",
    )
    parser.add_argument("--sft-think-dir", default="data/sft_think")
    parser.add_argument("--canonical-output-dir", default="data/canonical")
    parser.add_argument("--baseline-jsonl-output-dir", default="data/baselines")
    parser.add_argument("--sft-aligned-output-dir", default="data/sft_think_aligned")
    parser.add_argument("--sft-parquet-output-dir", default="data/sft")
    parser.add_argument("--baseline-parquet-output-dir", default="data/grpo_single_turn")
    parser.add_argument("--rlhf-parquet-output-dir", default="data/rlhf")
    parser.add_argument("--baseline-prompt-mode", default=PROMPT_MODE_THINK_ANSWER)
    parser.add_argument("--rlhf-prompt-mode", default=PROMPT_MODE_THINK_ANSWER)
    parser.add_argument("--max-turns", type=int, default=5)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    outputs = build_repo_source_datasets(
        repo_data_root=args.repo_data_root,
        sft_think_dir=args.sft_think_dir,
        canonical_output_dir=args.canonical_output_dir,
        baseline_jsonl_output_dir=args.baseline_jsonl_output_dir,
        sft_aligned_output_dir=args.sft_aligned_output_dir,
        sft_parquet_output_dir=args.sft_parquet_output_dir,
        baseline_parquet_output_dir=args.baseline_parquet_output_dir,
        rlhf_parquet_output_dir=args.rlhf_parquet_output_dir,
        baseline_prompt_mode=args.baseline_prompt_mode,
        rlhf_prompt_mode=args.rlhf_prompt_mode,
        max_turns=args.max_turns,
    )
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
