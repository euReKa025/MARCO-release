from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from marco.core_types import EpisodeState, StepResult
from marco.env.molecule_env import MoleculeEnv
from marco.env.molecule_validation import validate_model_output
from marco.env.predictor_client import PredictorClient
from marco.prompts.feedback_formatter import format_feedback
from marco.prompts.prompt_builder import build_turn_prompt
from marco.prompts.system_prompt import PROMPT_MODE_THINK_ANSWER, build_system_user_messages, render_messages
from marco.reward.config import build_reward_config_from_env
from marco.utils import read_jsonl

ResponseGenerator = Callable[[str, list[dict[str, str]], int], str]
BatchResponseGenerator = Callable[[list[str], list[list[dict[str, str]]], list[int]], list[str]]

PROMPT_SOURCE_DATASET_INSTRUCTION = "dataset_instruction"


def _default_canonical_data_dir() -> Path:
    return Path.cwd() / "data" / "canonical"


def default_dataset_path(*, property_setting: str, canonical_data_dir: str | Path | None = None) -> Path:
    base = Path(canonical_data_dir) if canonical_data_dir is not None else _default_canonical_data_dir()
    return base / "by_subtask" / property_setting / "canonical_mumo_test.jsonl"


def resolve_eval_records(
    *,
    dataset_path: str | Path,
    seen_setting: str = "seen",
    IND_setting: str = "IND",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    seen_norm = str(seen_setting).strip().lower()
    if seen_norm not in {"seen", "unseen", "all"}:
        raise ValueError(f"Unsupported seen_setting: {seen_setting}")
    ind_norm = str(IND_setting).strip().upper()
    if ind_norm != "IND":
        raise ValueError(f"Unsupported IND_setting: {IND_setting}")

    rows = read_jsonl(str(dataset_path))
    out: list[dict[str, Any]] = []
    for row in rows:
        md = row.get("metadata") or {}
        row_seen = str(md.get("instr_setting", "")).strip().lower()
        source_file = str(md.get("source_file", "")).strip()
        if (
            source_file
            and not source_file.startswith("IND_")
            and source_file != "test.json"
        ):
            continue
        if seen_norm != "all" and row_seen != seen_norm:
            continue
        out.append(row)
        if limit is not None and len(out) >= limit:
            break
    return out


def _feedback_message_from_result(result: StepResult, *, prompt_mode: str) -> str:
    return format_feedback(
        predictions=result.predictions,
        property_gaps=result.property_gaps,
        directional_improvements=result.directional_improvements,
        progress_score=result.progress_score,
        similarity=result.similarity,
        met_all_targets=result.met_all_targets,
        invalid_type=result.status,
        error_message=result.error_message,
        error_detail=result.error_detail,
        recovery_hint=result.recovery_hint,
        message=result.message,
        prompt_mode=prompt_mode,
    )


def _property_targets_for_prompt(state: EpisodeState) -> list[dict[str, Any]]:
    return [
        {"name": t.name, "direction": t.direction, "delta": t.delta}
        for t in state.property_targets
    ]


def _dataset_instruction(record: dict[str, Any]) -> str:
    instruction = record.get("instruction")
    if instruction is not None and str(instruction).strip():
        return str(instruction)

    metadata = record.get("metadata") or {}
    raw_instruction = metadata.get("raw_instruction")
    if raw_instruction is not None and str(raw_instruction).strip():
        return str(raw_instruction)

    sample_id = record.get("sample_id", "<unknown>")
    raise ValueError(
        f"eval record has missing non-empty instruction field; sample_id={sample_id}"
    )


def _build_initial_messages(
    *,
    record: dict[str, Any],
    prompt_mode: str,
) -> tuple[list[dict[str, str]], str | None]:
    instruction = _dataset_instruction(record)
    return build_system_user_messages(instruction, prompt_mode=prompt_mode), instruction


def _append_messages(
    *,
    state: EpisodeState,
    messages: list[dict[str, str]],
    response_text: str,
    result: StepResult,
    turn_id: int,
    prompt_mode: str,
) -> list[dict[str, str]]:
    next_messages = [dict(m) for m in messages]
    if response_text:
        next_messages.append({"role": "assistant", "content": response_text})
    if not result.should_stop and turn_id < state.max_turns:
        next_messages.append(
            {
                "role": "user",
                "content": build_turn_prompt(
                    x0_smiles=state.x0_smiles,
                    current_smiles=state.current_smiles,
                    property_targets=_property_targets_for_prompt(state),
                    turn_id=turn_id + 1,
                    history=[],
                    previous_feedback={
                        "predictions": result.predictions,
                        "property_gaps": result.property_gaps,
                        "directional_improvements": result.directional_improvements,
                        "progress_score": result.progress_score,
                        "similarity": result.similarity,
                        "met_all_targets": result.met_all_targets,
                        "invalid_type": result.status,
                        "error_message": result.error_message,
                        "error_detail": result.error_detail,
                        "recovery_hint": result.recovery_hint,
                        "message": result.message,
                    },
                    prompt_mode=prompt_mode,
                ),
            }
        )
    return next_messages


def _init_trajectory_context(
    record: dict[str, Any],
    *,
    env: Any,
    max_turns: int,
    prompt_mode: str,
) -> dict[str, Any]:
    state = env.init_episode(record=record, max_turns=max_turns)
    trajectory_id = str(record.get("sample_id") or state.episode_id)
    messages, source_instruction = _build_initial_messages(
        record=record,
        prompt_mode=prompt_mode,
    )
    return {
        "record": record,
        "state": state,
        "messages": messages,
        "done": False,
        "trajectory": {
            "trajectory_id": trajectory_id,
            "sample_id": str(record.get("sample_id", trajectory_id)),
            "subtask": str(record.get("subtask", state.subtask)),
            "split": str(record.get("split", "test")),
            "x0_smiles": state.x0_smiles,
            "reference_smiles": record.get("reference_smiles"),
            "metadata": dict(record.get("metadata") or {}),
            "prompt_mode": prompt_mode,
            "prompt_source": PROMPT_SOURCE_DATASET_INSTRUCTION,
            "source_instruction": source_instruction,
            "turns": [],
        },
    }


def _run_turn_step(
    *,
    env: Any,
    state: EpisodeState,
    prompt: str,
    input_messages: list[dict[str, str]],
    response_text: str,
    turn_id: int,
    prompt_mode: str,
) -> tuple[dict[str, Any], StepResult]:
    validation = validate_model_output(response_text)
    if validation.status != "ok":
        result = env.step_invalid(
            state=state,
            turn_id=turn_id,
            invalid_type=validation.status,
            message=validation.message,
        )
    else:
        result = env.step(state=state, candidate_smiles=str(validation.smiles), turn_id=turn_id)

    env.apply_step(state, result)
    feedback = _feedback_message_from_result(result, prompt_mode=prompt_mode)
    turn_row = {
        "turn_id": int(turn_id),
        "prompt": prompt,
        "input_messages": input_messages,
        "output_response": response_text,
        "output_response_repr": repr(response_text),
        "parsed_smiles": validation.smiles,
        "invalid_type": result.status,
        "candidate_smiles": result.candidate_smiles,
        "predictions": dict(result.predictions),
        "directional_improvements": dict(result.directional_improvements),
        "progress_score": float(result.progress_score),
        "similarity": float(result.similarity),
        "met_all_targets": bool(result.met_all_targets),
        "active_property_names": list(result.active_property_names),
        "feedback": feedback,
        "stop_reason": str(result.message),
        "error_message": str(result.error_message),
        "error_detail": str(result.error_detail),
        "recovery_hint": str(result.recovery_hint),
    }
    return turn_row, result


def run_multi_turn_episode(
    record: dict[str, Any],
    *,
    env: Any,
    tokenizer: Any,
    generate_response: ResponseGenerator,
    max_turns: int,
    prompt_mode: str = PROMPT_MODE_THINK_ANSWER,
) -> dict[str, Any]:
    context = _init_trajectory_context(
        record,
        env=env,
        max_turns=max_turns,
        prompt_mode=prompt_mode,
    )
    state = context["state"]
    messages = context["messages"]
    trajectory = context["trajectory"]

    for turn_id in range(1, max_turns + 1):
        prompt = render_messages(messages, tokenizer)
        input_messages = [dict(m) for m in messages]
        response_text = generate_response(prompt, input_messages, turn_id)
        turn_row, result = _run_turn_step(
            env=env,
            state=state,
            prompt=prompt,
            input_messages=input_messages,
            response_text=response_text,
            turn_id=turn_id,
            prompt_mode=prompt_mode,
        )
        trajectory["turns"].append(turn_row)
        messages = _append_messages(
            state=state,
            messages=messages,
            response_text=response_text,
            result=result,
            turn_id=turn_id,
            prompt_mode=prompt_mode,
        )
        if result.should_stop:
            break

    return trajectory


def run_multi_turn_episodes_batched(
    records: Iterable[dict[str, Any]],
    *,
    env: Any,
    tokenizer: Any,
    generate_response_batch: BatchResponseGenerator,
    max_turns: int,
    prompt_mode: str = PROMPT_MODE_THINK_ANSWER,
    batch_size: int = 8,
    progress_interval: int = 0,
) -> list[dict[str, Any]]:
    records_list = list(records)
    if not records_list:
        return []
    batch_size = max(1, int(batch_size))
    progress_interval = max(0, int(progress_interval))
    contexts = [
        _init_trajectory_context(
            record,
            env=env,
            max_turns=max_turns,
            prompt_mode=prompt_mode,
        )
        for record in records_list
    ]
    total = len(contexts)
    processed_turns = 0
    next_progress = progress_interval

    for turn_id in range(1, max_turns + 1):
        active_indices = [idx for idx, context in enumerate(contexts) if not context["done"]]
        if not active_indices:
            break

        for offset in range(0, len(active_indices), batch_size):
            batch_indices = active_indices[offset : offset + batch_size]
            prompts: list[str] = []
            messages_batch: list[list[dict[str, str]]] = []
            for idx in batch_indices:
                messages = contexts[idx]["messages"]
                prompts.append(render_messages(messages, tokenizer))
                messages_batch.append([dict(m) for m in messages])

            responses = generate_response_batch(prompts, messages_batch, [turn_id] * len(batch_indices))
            if len(responses) != len(batch_indices):
                raise ValueError(
                    "generate_response_batch returned "
                    f"{len(responses)} responses for {len(batch_indices)} prompts"
                )

            for idx, prompt, input_messages, response_text in zip(
                batch_indices,
                prompts,
                messages_batch,
                responses,
                strict=True,
            ):
                context = contexts[idx]
                state = context["state"]
                turn_row, result = _run_turn_step(
                    env=env,
                    state=state,
                    prompt=prompt,
                    input_messages=input_messages,
                    response_text=response_text,
                    turn_id=turn_id,
                    prompt_mode=prompt_mode,
                )
                context["trajectory"]["turns"].append(turn_row)
                context["messages"] = _append_messages(
                    state=state,
                    messages=context["messages"],
                    response_text=response_text,
                    result=result,
                    turn_id=turn_id,
                    prompt_mode=prompt_mode,
                )
                if result.should_stop:
                    context["done"] = True

            processed_turns += len(batch_indices)
            if progress_interval and processed_turns >= next_progress:
                completed = sum(1 for context in contexts if context["done"])
                print(
                    "[multi-turn-inference] "
                    f"processed_turns={processed_turns} completed={completed}/{total} "
                    f"active={total - completed} turn={turn_id}",
                    flush=True,
                )
                while next_progress <= processed_turns:
                    next_progress += progress_interval

    if progress_interval:
        completed_or_exhausted = len(contexts)
        print(
            "[multi-turn-inference] "
            f"finished trajectories={completed_or_exhausted}/{total} processed_turns={processed_turns}",
            flush=True,
        )
    return [context["trajectory"] for context in contexts]


class HFResponseGenerator:
    def __init__(
        self,
        model,
        tokenizer,
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
    ):
        self.model = model
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.device = next(model.parameters()).device

    def __call__(self, prompt: str, messages: list[dict[str, str]], turn_id: int) -> str:
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": self.do_sample,
        }
        if self.do_sample:
            kwargs["temperature"] = self.temperature
            kwargs["top_p"] = self.top_p
        with torch.no_grad():
            output = self.model.generate(**encoded, **kwargs)
        gen_ids = output[0, encoded["input_ids"].shape[1] :]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=False)

    def generate_batch(
        self,
        prompts: list[str],
        messages_batch: list[list[dict[str, str]]],
        turn_ids: list[int],
    ) -> list[str]:
        if not prompts:
            return []
        if self.tokenizer.padding_side != "left":
            self.tokenizer.padding_side = "left"
        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": self.do_sample,
        }
        if self.do_sample:
            kwargs["temperature"] = self.temperature
            kwargs["top_p"] = self.top_p
        with torch.no_grad():
            output = self.model.generate(**encoded, **kwargs)
        prompt_length = encoded["input_ids"].shape[1]
        return [
            self.tokenizer.decode(output[row_idx, prompt_length:], skip_special_tokens=False)
            for row_idx in range(len(prompts))
        ]


def generate_trajectories(
    *,
    hf_model_dir: str | Path,
    records: Iterable[dict[str, Any]],
    max_turns: int,
    prompt_mode: str = PROMPT_MODE_THINK_ANSWER,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    generation_batch_size: int = 1,
    progress_interval: int = 0,
) -> list[dict[str, Any]]:
    records_list = list(records)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval()
    generator = HFResponseGenerator(
        model,
        tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )
    env = MoleculeEnv(PredictorClient(), build_reward_config_from_env())
    if generation_batch_size > 1:
        return run_multi_turn_episodes_batched(
            records_list,
            env=env,
            tokenizer=tokenizer,
            generate_response_batch=generator.generate_batch,
            max_turns=max_turns,
            prompt_mode=prompt_mode,
            batch_size=generation_batch_size,
            progress_interval=progress_interval,
        )

    trajectories: list[dict[str, Any]] = []
    for record in records_list:
        trajectories.append(
            run_multi_turn_episode(
                record,
                env=env,
                tokenizer=tokenizer,
                generate_response=generator,
                max_turns=max_turns,
                prompt_mode=prompt_mode,
            )
        )
        if progress_interval and len(trajectories) % progress_interval == 0:
            print(
                "[multi-turn-inference] "
                f"finished trajectories={len(trajectories)}/{len(records_list)}",
                flush=True,
            )
    return trajectories


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run standalone multi-turn MARCO inference from an HF checkpoint.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--property_setting", required=True)
    parser.add_argument("--seen_setting", default="seen")
    parser.add_argument("--IND_setting", default="IND")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_name", required=True)
    parser.add_argument("--method_name", default="marco")
    parser.add_argument("--experiment_prefix", default="marco")
    parser.add_argument("--canonical_data_dir", default=str(_default_canonical_data_dir()))
    parser.add_argument("--prompt_mode", default=PROMPT_MODE_THINK_ANSWER)
    parser.add_argument("--max_turns", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--generation_batch_size", type=int, default=1)
    parser.add_argument("--progress_interval", type=int, default=0)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    dataset_path = default_dataset_path(property_setting=args.property_setting, canonical_data_dir=args.canonical_data_dir)
    records = resolve_eval_records(
        dataset_path=dataset_path,
        seen_setting=args.seen_setting,
        IND_setting=args.IND_setting,
        limit=args.limit,
    )
    trajectories = generate_trajectories(
        hf_model_dir=args.model_path,
        records=records,
        max_turns=args.max_turns,
        prompt_mode=args.prompt_mode,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        generation_batch_size=args.generation_batch_size,
        progress_interval=args.progress_interval,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name
    output_path.write_text(json.dumps(trajectories, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
