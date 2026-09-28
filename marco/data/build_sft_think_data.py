from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import requests

from marco.prompts.system_prompt import SYSTEM_PROMPT
from marco.utils import read_jsonl, write_jsonl


DEFAULT_SPLIT = ("train",)
_THINK_TAG_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL)


class ThinkGenerator(Protocol):
    def generate_think(self, record: dict[str, Any], *, min_sentences: int, max_sentences: int) -> str:
        ...


@dataclass
class _ThinkValidationResult:
    think_text: str
    sentence_count: int


@dataclass
class _LocalThinkValidationError(Exception):
    message: str
    raw_output_response: str

    def __str__(self) -> str:
        return self.message


class OpenAICompatibleThinkGenerator:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        chat_completions_path: str = "/v1/chat/completions",
        timeout_s: float = 60.0,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> None:
        base = base_url.rstrip("/")
        path = chat_completions_path.strip()
        if not path.startswith("/"):
            path = "/" + path
        self.endpoint = f"{base}{path}"
        self.api_key = api_key
        self.model = model
        self.timeout_s = float(timeout_s)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.provider = "openai-compatible-http"

    def generate_think(self, record: dict[str, Any], *, min_sentences: int, max_sentences: int) -> str:
        user_prompt = _build_teacher_user_prompt(
            record,
            min_sentences=min_sentences,
            max_sentences=max_sentences,
        )

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        response = requests.post(self.endpoint, json=payload, headers=headers, timeout=self.timeout_s)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("Teacher response missing choices")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise ValueError("Teacher response missing message object")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Teacher response content is empty")
        return content.strip()


def _canonical_file(canonical_dir: str, split: str) -> str:
    return os.path.join(canonical_dir, f"canonical_mumo_{split}.jsonl")


def _safe_instruction(record: dict[str, Any]) -> str:
    instruction = record.get("instruction")
    if isinstance(instruction, str) and instruction.strip():
        return instruction.strip()
    x0 = str(record.get("x0_smiles", "")).strip()
    subtask = str(record.get("subtask", "molecular optimization")).strip()
    return f"Optimize molecule {x0} for task {subtask}. Return one <SMILES>SMILES</SMILES>."


def _normalized_properties(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _properties_to_text(properties: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for prop in properties:
        name = str(prop.get("name", "")).strip()
        direction = str(prop.get("direction", "")).strip()
        source = prop.get("source")
        target = prop.get("target")
        delta = prop.get("delta")
        lines.append(
            f"- {name}: direction={direction}, source={source}, target={target}, delta={delta}"
        )
    return "\n".join(lines) if lines else "- (none)"


def _property_tokens(properties: list[dict[str, Any]]) -> list[str]:
    tokens: list[str] = []
    for prop in properties:
        name = prop.get("name")
        if not isinstance(name, str):
            continue
        name = name.strip()
        if not name:
            continue
        tokens.append(name)
    return tokens


def _build_teacher_user_prompt(record: dict[str, Any], *, min_sentences: int, max_sentences: int) -> str:
    instruction = _safe_instruction(record)
    x0_smiles = str(record.get("x0_smiles", "")).strip()
    answer = str(record.get("reference_smiles", "")).strip()
    properties = _normalized_properties(record.get("properties"))
    tokens = _property_tokens(properties)
    token_text = ", ".join(tokens) if tokens else "(none)"

    return (
        "Generate only one reasoning block for SFT data.\n"
        "Output exactly one block: <think>...</think>\n\n"
        "Hard constraints:\n"
        f"1) Exactly {min_sentences}-{max_sentences} sentences.\n"
        f"2) You MUST use these exact property tokens: {token_text}\n"
        "3) For each property token, include its direction word explicitly: increase or decrease.\n"
        "4) Do not output <SMILES>, </SMILES>, or any SMILES string.\n"
        "5) Do not include <think> or </think> inside the reasoning content.\n"
        "6) Do not include escaped tags such as \\<think\\> or \\</think\\>.\n"
        "7) Keep the content concise, chemistry-focused, and actionable.\n\n"
        "Input context:\n"
        f"Instruction:\n{instruction}\n\n"
        f"Input molecule (x0): {x0_smiles}\n"
        f"Reference molecule (for answer only, never output): {answer}\n"
        "Properties:\n"
        f"{_properties_to_text(properties)}\n\n"
        "Return only the <think> block."
    )


def _sentence_count(text: str) -> int:
    segments = [seg.strip() for seg in re.split(r"[.!?。！？\n]+", text) if seg.strip()]
    return len(segments)


def _extract_think_text(raw_text: str) -> str:
    match = _THINK_TAG_RE.search(raw_text)
    if match:
        return match.group(1).strip()
    return raw_text.strip()


def _sanitize_think_text(think_text: str) -> str:
    text = think_text
    text = re.sub(r"\\<\s*/?\s*think\s*\\>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"</?\s*think\s*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _validate_think_text(
    think_text: str,
    properties: list[dict[str, Any]],
    *,
    min_sentences: int,
    max_sentences: int,
) -> _ThinkValidationResult:
    lowered = think_text.lower()
    if "<answer>" in lowered or "<smiles>" in lowered or "</smiles>" in lowered:
        raise ValueError("think text unexpectedly contains tagged molecule output")

    sentence_count = _sentence_count(think_text)
    if sentence_count < min_sentences or sentence_count > max_sentences:
        raise ValueError(
            f"think sentence_count={sentence_count} out of range [{min_sentences}, {max_sentences}]"
        )

    missing_property_names: list[str] = []
    direction_needs: set[str] = set()
    for prop in properties:
        name = prop.get("name")
        if isinstance(name, str) and name.strip():
            if name.lower() not in lowered:
                missing_property_names.append(name)
        direction = prop.get("direction")
        if isinstance(direction, str) and direction.strip():
            direction_needs.add(direction.lower())

    if missing_property_names:
        raise ValueError(f"think missing property names: {missing_property_names}")

    direction_tokens = {
        "increase": ("increase", "improve", "higher", "raise", "boost"),
        "decrease": ("decrease", "reduce", "lower", "drop"),
    }
    for direction in direction_needs:
        tokens = direction_tokens.get(direction, (direction,))
        if not any(token in lowered for token in tokens):
            raise ValueError(f"think missing direction wording for '{direction}'")

    return _ThinkValidationResult(think_text=think_text.strip(), sentence_count=sentence_count)


def _build_sft_row(
    record: dict[str, Any],
    think_generator: ThinkGenerator,
    *,
    min_sentences: int,
    max_sentences: int,
    max_retries: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    answer = record.get("reference_smiles")
    if not isinstance(answer, str) or not answer.strip():
        return None, {
            "sample_id": record.get("sample_id"),
            "subtask": record.get("subtask"),
            "split": record.get("split"),
            "status": "invalid_reference_smiles",
            "error": "reference_smiles is empty",
        }

    properties = _normalized_properties(record.get("properties"))
    last_error: str | None = None
    last_error_type: str = "teacher_generation_failed"
    last_raw_output_response: str | None = None

    for attempt in range(max_retries + 1):
        raw_think = ""
        try:
            raw_think = think_generator.generate_think(
                record,
                min_sentences=min_sentences,
                max_sentences=max_sentences,
            )
            think_text = _sanitize_think_text(_extract_think_text(raw_think))
            try:
                validated = _validate_think_text(
                    think_text,
                    properties,
                    min_sentences=min_sentences,
                    max_sentences=max_sentences,
                )
            except ValueError as exc:
                raise _LocalThinkValidationError(
                    message=str(exc),
                    raw_output_response=raw_think,
                ) from exc
            assistant_content = f"<think>{validated.think_text}</think>\n<SMILES>{answer.strip()}</SMILES>"
            metadata = {
                "sample_id": record.get("sample_id"),
                "subtask": record.get("subtask"),
                "split": record.get("split"),
                "x0_smiles": record.get("x0_smiles"),
                "reference_smiles": answer.strip(),
                "properties": properties,
                "teacher": {
                    "provider": getattr(think_generator, "provider", type(think_generator).__name__),
                    "model": getattr(think_generator, "model", None),
                    "retries_used": attempt,
                },
                "think_quality": {
                    "sentence_count": validated.sentence_count,
                    "min_sentences": min_sentences,
                    "max_sentences": max_sentences,
                },
            }
            return {
                "messages": [
                    {"role": "user", "content": _safe_instruction(record)},
                    {"role": "assistant", "content": assistant_content},
                ],
                "metadata": metadata,
            }, None
        except _LocalThinkValidationError as exc:
            last_error = str(exc)
            last_error_type = "local_validation_failed"
            last_raw_output_response = exc.raw_output_response
            print(
                "[sft-think][local-validation-failed] "
                f"sample_id={record.get('sample_id')} attempt={attempt} "
                f"error={last_error} raw_output_response={last_raw_output_response}"
            )
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            last_error_type = "teacher_request_failed"
            last_raw_output_response = None

    failure: dict[str, Any] = {
        "sample_id": record.get("sample_id"),
        "subtask": record.get("subtask"),
        "split": record.get("split"),
        "status": "teacher_generation_failed",
        "error_type": last_error_type,
        "error": last_error or "unknown error",
    }
    if last_raw_output_response is not None:
        failure["raw_output_response"] = last_raw_output_response
    return None, failure


def _write_split_outputs(
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    output_dir: str,
    split: str,
) -> dict[str, Any]:
    merged_path = os.path.join(output_dir, f"sft_{split}.jsonl")
    failure_manifest = os.path.join(output_dir, f"sft_{split}_failures.jsonl")

    write_jsonl(merged_path, rows)
    write_jsonl(failure_manifest, failures)

    by_subtask: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            continue
        subtask = str(metadata.get("subtask", "unknown"))
        by_subtask[subtask].append(row)

    by_subtask_paths: dict[str, str] = {}
    for subtask, sub_rows in by_subtask.items():
        subtask_path = os.path.join(output_dir, "by_subtask", subtask, f"sft_{split}.jsonl")
        write_jsonl(subtask_path, sub_rows)
        by_subtask_paths[subtask] = subtask_path

    return {
        "sft_merged": merged_path,
        "failure_manifest": failure_manifest,
        "sft_by_subtask": by_subtask_paths,
    }


def _sample_id_from_sft_row(row: dict[str, Any]) -> str | None:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        return None
    sample_id = metadata.get("sample_id")
    if isinstance(sample_id, str) and sample_id:
        return sample_id
    return None


def _sample_id_from_failure_row(row: dict[str, Any]) -> str | None:
    sample_id = row.get("sample_id")
    if isinstance(sample_id, str) and sample_id:
        return sample_id
    return None


def _merge_rows_by_sample_id(
    existing_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in [*existing_rows, *new_rows]:
        sid = _sample_id_from_sft_row(row)
        if sid is not None:
            if sid in seen:
                continue
            seen.add(sid)
        merged.append(row)
    return merged


def _merge_failures_for_resume(
    existing_failures: list[dict[str, Any]],
    new_failures: list[dict[str, Any]],
    success_ids: set[str],
) -> list[dict[str, Any]]:
    merged_with_ids: dict[str, dict[str, Any]] = {}
    merged_without_ids: list[dict[str, Any]] = []

    def _consume(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            sid = _sample_id_from_failure_row(row)
            if sid is not None:
                if sid in success_ids:
                    continue
                merged_with_ids[sid] = row
                continue
            merged_without_ids.append(row)

    _consume(existing_failures)
    _consume(new_failures)
    return [*merged_without_ids, *merged_with_ids.values()]


def build_sft_think_data(
    *,
    canonical_dir: str,
    output_dir: str,
    think_generator: ThinkGenerator,
    splits: Sequence[str] = DEFAULT_SPLIT,
    max_samples: int = 0,
    min_sentences: int = 3,
    max_sentences: int = 5,
    max_retries: int = 2,
    on_failure: str = "error",
    allow_existing_output: bool = False,
    resume: bool = False,
    concurrency: int = 1,
) -> dict[str, Any]:
    if min_sentences <= 0 or max_sentences <= 0 or min_sentences > max_sentences:
        raise ValueError("Invalid sentence range")
    if max_samples < 0:
        raise ValueError("max_samples must be >= 0")
    if concurrency <= 0:
        raise ValueError("concurrency must be >= 1")
    if on_failure not in {"error", "skip"}:
        raise ValueError("on_failure must be one of: error, skip")

    output_root = output_dir
    if os.path.exists(output_root) and not allow_existing_output and not resume:
        if os.path.isdir(output_root) and any(os.scandir(output_root)):
            raise FileExistsError(
                f"Output directory already contains files: {output_root}. "
                "Use --allow-existing-output to overwrite."
            )

    result: dict[str, Any] = {}

    for split in splits:
        path = _canonical_file(canonical_dir, split)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Canonical split file not found: {path}")

        existing_merged_path = os.path.join(output_dir, f"sft_{split}.jsonl")
        existing_failure_path = os.path.join(output_dir, f"sft_{split}_failures.jsonl")
        existing_rows: list[dict[str, Any]] = []
        existing_failures: list[dict[str, Any]] = []
        existing_success_ids: set[str] = set()
        if resume:
            if os.path.exists(existing_merged_path):
                existing_rows = read_jsonl(existing_merged_path)
                existing_success_ids = {
                    sid
                    for sid in (_sample_id_from_sft_row(row) for row in existing_rows)
                    if sid is not None
                }
            if os.path.exists(existing_failure_path):
                existing_failures = read_jsonl(existing_failure_path)

        all_records = read_jsonl(path)
        records = all_records[:max_samples] if max_samples > 0 else all_records

        new_rows: list[dict[str, Any]] = []
        new_failures: list[dict[str, Any]] = []
        skipped_existing_success_rows = 0

        pending_records: list[dict[str, Any]] = []
        for record in records:
            sample_id = record.get("sample_id")
            if resume and isinstance(sample_id, str) and sample_id in existing_success_ids:
                skipped_existing_success_rows += 1
                continue
            pending_records.append(record)

        def _process_record(record: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
            return _build_sft_row(
                record,
                think_generator,
                min_sentences=min_sentences,
                max_sentences=max_sentences,
                max_retries=max_retries,
            )

        if concurrency == 1 or len(pending_records) <= 1:
            processed = [_process_record(record) for record in pending_records]
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                processed = list(executor.map(_process_record, pending_records))

        for row, failure in processed:
            if row is not None:
                new_rows.append(row)
            if failure is not None:
                new_failures.append(failure)

        final_rows = _merge_rows_by_sample_id(existing_rows, new_rows) if resume else new_rows
        final_success_ids = {
            sid
            for sid in (_sample_id_from_sft_row(row) for row in final_rows)
            if sid is not None
        }
        final_failures = (
            _merge_failures_for_resume(existing_failures, new_failures, final_success_ids)
            if resume
            else new_failures
        )

        split_outputs = _write_split_outputs(final_rows, final_failures, output_dir, split)
        processed_rows = len(records) - skipped_existing_success_rows
        split_outputs["stats"] = {
            "input_rows_total": len(all_records),
            "input_rows_selected": len(records),
            "processed_rows": processed_rows,
            "skipped_existing_success_rows": skipped_existing_success_rows,
            "new_output_rows": len(new_rows),
            "new_failure_rows": len(new_failures),
            "output_rows": len(final_rows),
            "failure_rows": len(final_failures),
            "success_rate": (len(new_rows) / processed_rows) if processed_rows > 0 else 1.0,
            "resume": bool(resume),
            "concurrency": int(concurrency),
        }
        result[split] = split_outputs

        if final_failures and on_failure == "error":
            raise RuntimeError(
                f"Split '{split}' has {len(final_failures)} failed rows. "
                f"See failure manifest: {split_outputs['failure_manifest']}"
            )

    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build SFT training data with teacher-generated <think> and fixed "
            "<SMILES> from canonical_mumo_{split}.jsonl."
        )
    )
    parser.add_argument("--canonical-dir", default="data/canonical")
    parser.add_argument("--output-dir", default="data/sft_think")
    parser.add_argument("--splits", default="train", help="Comma-separated splits, default: train")
    parser.add_argument("--base-url", default=os.getenv("SFT_TEACHER_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.getenv("SFT_TEACHER_API_KEY", ""))
    parser.add_argument("--model", default=os.getenv("SFT_TEACHER_MODEL", ""))
    parser.add_argument("--chat-completions-path", default="/v1/chat/completions")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--max-samples", type=int, default=0, help="Process only the first N rows per split; 0 means all rows.")
    parser.add_argument("--min-sentences", type=int, default=2)
    parser.add_argument("--max-sentences", type=int, default=5)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--on-failure", choices=("error", "skip"), default="error")
    parser.add_argument("--allow-existing-output", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output files and skip already successful sample_id rows.")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.getenv("SFT_THINK_CONCURRENCY", "1")),
        help="Number of concurrent teacher API requests per split. Default: 1.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        raise ValueError("No valid split provided")

    if not args.base_url:
        raise ValueError("--base-url is required")
    if not args.model:
        raise ValueError("--model is required")

    generator = OpenAICompatibleThinkGenerator(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        chat_completions_path=args.chat_completions_path,
        timeout_s=float(args.timeout_s),
        temperature=float(args.temperature),
        max_tokens=int(args.max_tokens),
    )
    outputs = build_sft_think_data(
        canonical_dir=args.canonical_dir,
        output_dir=args.output_dir,
        think_generator=generator,
        splits=splits,
        max_samples=int(args.max_samples),
        min_sentences=int(args.min_sentences),
        max_sentences=int(args.max_sentences),
        max_retries=int(args.max_retries),
        on_failure=args.on_failure,
        allow_existing_output=bool(args.allow_existing_output),
        resume=bool(args.resume),
        concurrency=int(args.concurrency),
    )
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
