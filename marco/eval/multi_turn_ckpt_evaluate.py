from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from marco.eval.eval_multi_turn import evaluate_multi_turn_trajectories, select_multi_turn_reporting_turn


def summarize_trajectories(trajectories: list[dict[str, Any]], *, similarity_threshold: float) -> dict[str, Any]:
    payload = evaluate_multi_turn_trajectories(trajectories, similarity_threshold=similarity_threshold)
    payload["trajectories"] = trajectories
    payload["similarity_threshold"] = float(similarity_threshold)
    return payload


def _flatten_detailed_rows(summary: dict[str, Any], trajectories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(t.get("trajectory_id", "")): t for t in trajectories}
    similarity_threshold = float(summary.get("similarity_threshold", 0.0))
    similarity_target_low = float(summary.get("similarity_target_low", 0.6))
    similarity_copy_threshold = float(summary.get("similarity_copy_threshold", 0.99))
    rows: list[dict[str, Any]] = []
    for row in summary.get("rows", []):
        traj = by_id.get(str(row.get("trajectory_id", "")), {})
        turns = list(traj.get("turns") or [])
        selection = select_multi_turn_reporting_turn(
            turns,
            similarity_target_low=similarity_target_low,
            similarity_copy_threshold=similarity_copy_threshold,
        )
        selected_idx = int(selection.get("selected_idx", -1))
        final_turn = turns[selected_idx] if 0 <= selected_idx < len(turns) else {}
        final_similarity = float(final_turn.get("similarity", 0.0))
        improvements = final_turn.get("directional_improvements") or {}
        active_names = list(final_turn.get("active_property_names") or improvements.keys())
        detailed = {
            "trajectory_id": row.get("trajectory_id", ""),
            "subtask": traj.get("subtask", ""),
            "x0_smiles": traj.get("x0_smiles", ""),
            "final_candidate_smiles": final_turn.get("candidate_smiles"),
            "success": row.get("success", 0.0),
            "constraint_success": row.get("constraint_success", 0.0),
            "turns_used": row.get("turns_used", 0.0),
            "oracle_calls": row.get("oracle_calls", 0.0),
            "final_similarity": final_similarity,
            "valid_selected_candidate": row.get("valid_selected_candidate", 0.0),
            "meets_similarity_threshold": float(final_similarity >= similarity_threshold),
            "invalid_turns": row.get("invalid_turns", 0.0),
        }
        for key in sorted(active_names):
            detailed[f"improvement_{key}"] = float(improvements.get(key, 0.0))
        rows.append(detailed)
    return rows


def write_evaluation_outputs(
    summary: dict[str, Any],
    *,
    experiment_dir: str | Path,
    output_folder: str | Path,
    IND_setting: str,
    seen_setting: str,
    property_setting: str,
    method_name: str,
) -> dict[str, str]:
    experiment_dir = Path(experiment_dir)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    trajectories = list(summary.get("trajectories") or [])
    detailed_rows = _flatten_detailed_rows(summary, trajectories)
    detailed_csv = output_folder / f"detailed_results_{IND_setting}_{seen_setting}_{property_setting}{method_name}.csv"
    avg_csv = output_folder / f"avg_metrics_{IND_setting}_{seen_setting}_{property_setting}{method_name}.csv"
    summary_json = experiment_dir / f"{IND_setting}_sft_{seen_setting}_{property_setting}.json"

    if detailed_rows:
        fieldnames = list(detailed_rows[0].keys())
        with detailed_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(detailed_rows)
    else:
        detailed_csv.write_text("", encoding="utf-8")

    metrics = dict(summary.get("metrics") or {})
    with avg_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    summary_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "detailed_csv": str(detailed_csv),
        "avg_csv": str(avg_csv),
        "summary_json": str(summary_json),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate standalone MARCO multi-turn trajectory outputs.")
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--property_setting", required=True)
    parser.add_argument("--seen_setting", required=True)
    parser.add_argument("--IND_setting", required=True)
    parser.add_argument("--experiment_prefix", default="marco")
    parser.add_argument("--method_name", default="marco")
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--similarity_threshold", type=float, default=0.4)
    args = parser.parse_args()

    input_json = Path(args.input_json)
    trajectories = json.loads(input_json.read_text(encoding="utf-8"))
    summary = summarize_trajectories(trajectories, similarity_threshold=args.similarity_threshold)
    paths = write_evaluation_outputs(
        summary,
        experiment_dir=input_json.parent,
        output_folder=args.output_folder,
        IND_setting=args.IND_setting,
        seen_setting=args.seen_setting,
        property_setting=args.property_setting,
        method_name=args.method_name,
    )
    print(json.dumps({"metrics": summary["metrics"], "paths": paths}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
