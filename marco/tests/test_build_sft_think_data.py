from __future__ import annotations

import threading
import time
from pathlib import Path

from marco.utils import read_jsonl, write_jsonl


def _row(sample_id: str, subtask: str, answer: str) -> dict:
    return {
        "sample_id": sample_id,
        "split": "train",
        "subtask": subtask,
        "instruction": (
            "Your task is to modify the given molecule to adjust specific molecular properties while keeping "
            "structural changes as minimal as possible."
        ),
        "x0_smiles": "CCO",
        "reference_smiles": answer,
        "properties": [
            {"name": "bbbp", "direction": "increase", "source": 0.1, "target": 0.2},
            {"name": "drd2", "direction": "increase", "source": 0.2, "target": 0.4},
            {"name": "plogp", "direction": "increase", "source": -0.1, "target": 0.3},
        ],
    }


def _existing_sft_row(sample_id: str, subtask: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": "instruction"},
            {"role": "assistant", "content": f"<think>existing think for {sample_id}.</think>\n<SMILES>{answer}</SMILES>"},
        ],
        "metadata": {
            "sample_id": sample_id,
            "subtask": subtask,
            "split": "train",
            "x0_smiles": "CCO",
            "reference_smiles": answer,
            "properties": [
                {"name": "bbbp", "direction": "increase", "source": 0.1},
                {"name": "drd2", "direction": "increase", "source": 0.2},
                {"name": "plogp", "direction": "increase", "source": -0.1},
            ],
            "teacher": {"provider": "existing", "model": "existing", "retries_used": 0},
            "think_quality": {"sentence_count": 1, "min_sentences": 1, "max_sentences": 1},
        },
    }


class _DummyThinkGenerator:
    def __init__(self, responses: dict[str, list[str]]):
        self._responses = {k: list(v) for k, v in responses.items()}
        self.calls: dict[str, int] = {}

    def generate_think(self, record: dict, *, min_sentences: int, max_sentences: int) -> str:
        sample_id = str(record.get("sample_id"))
        self.calls[sample_id] = self.calls.get(sample_id, 0) + 1
        queue = self._responses.get(sample_id, [])
        if not queue:
            raise RuntimeError(f"no response for sample {sample_id}")
        return queue.pop(0)


class _SlowThreadSafeThinkGenerator:
    def __init__(self, response: str, sleep_s: float = 0.05):
        self._response = response
        self._sleep_s = sleep_s
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def generate_think(self, record: dict, *, min_sentences: int, max_sentences: int) -> str:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self._sleep_s)
            return self._response
        finally:
            with self._lock:
                self.active -= 1


def test_build_sft_think_data_train_outputs_and_subtask_split(tmp_path: Path):
    from marco.data.build_sft_think_data import build_sft_think_data

    canonical_dir = tmp_path / "canonical"
    write_jsonl(
        canonical_dir / "canonical_mumo_train.jsonl",
        [
            _row("s1", "bbbp+drd2+plogp", "CCN"),
            _row("s2", "bbbp+drd2+qed", "CCC"),
        ],
    )

    think = (
        "<think>Increase bbbp by adding a small polarity adjustment. "
        "Increase drd2 by preserving the core pharmacophore. "
        "Increase plogp with a limited hydrophobic substitution.</think>"
    )
    generator = _DummyThinkGenerator({"s1": [think], "s2": [think]})

    output_dir = tmp_path / "sft_think"
    outputs = build_sft_think_data(
        canonical_dir=str(canonical_dir),
        output_dir=str(output_dir),
        think_generator=generator,
    )

    merged_rows = read_jsonl(outputs["train"]["sft_merged"])
    assert len(merged_rows) == 2
    assistant_0 = merged_rows[0]["messages"][1]["content"]
    assert "<think>" in assistant_0 and "</think>" in assistant_0
    assert "<SMILES>CCN</SMILES>" in assistant_0
    assert merged_rows[0]["metadata"]["think_quality"]["sentence_count"] == 3
    assert merged_rows[0]["metadata"]["teacher"]["retries_used"] == 0

    by_subtask_path = outputs["train"]["sft_by_subtask"]["bbbp+drd2+plogp"]
    by_subtask_rows = read_jsonl(by_subtask_path)
    assert len(by_subtask_rows) == 1
    assert by_subtask_rows[0]["metadata"]["sample_id"] == "s1"


def test_build_sft_think_data_concurrency_preserves_output_order(tmp_path: Path):
    from marco.data.build_sft_think_data import build_sft_think_data

    canonical_dir = tmp_path / "canonical"
    write_jsonl(
        canonical_dir / "canonical_mumo_train.jsonl",
        [
            _row("s1", "bbbp+drd2+plogp", "CCN"),
            _row("s2", "bbbp+drd2+plogp", "CCO"),
            _row("s3", "bbbp+drd2+plogp", "CCC"),
        ],
    )

    think = (
        "<think>Increase bbbp while controlling polarity. "
        "Increase drd2 by preserving key interactions. "
        "Increase plogp with a conservative hydrophobic edit.</think>"
    )
    generator = _SlowThreadSafeThinkGenerator(think)

    output_dir = tmp_path / "sft_think"
    outputs = build_sft_think_data(
        canonical_dir=str(canonical_dir),
        output_dir=str(output_dir),
        think_generator=generator,
        concurrency=3,
    )

    merged_rows = read_jsonl(outputs["train"]["sft_merged"])
    assert [row["metadata"]["sample_id"] for row in merged_rows] == ["s1", "s2", "s3"]
    assert generator.max_active > 1


def test_build_sft_think_data_retries_and_failure_manifest(tmp_path: Path):
    from marco.data.build_sft_think_data import build_sft_think_data

    canonical_dir = tmp_path / "canonical"
    write_jsonl(canonical_dir / "canonical_mumo_train.jsonl", [_row("s1", "bbbp+drd2+plogp", "CCN"), _row("s2", "bbbp+drd2+plogp", "CCO")])

    bad_think = "<think>Increase bbbp only.</think>"
    good_think = (
        "<think>Increase bbbp while preserving scaffold balance. "
        "Increase drd2 by keeping key interaction motifs. "
        "Increase plogp with a conservative hydrophobic edit.</think>"
    )
    generator = _DummyThinkGenerator(
        {
            "s1": [bad_think, good_think],
            "s2": [bad_think, bad_think, bad_think],
        }
    )

    output_dir = tmp_path / "sft_think"
    outputs = build_sft_think_data(
        canonical_dir=str(canonical_dir),
        output_dir=str(output_dir),
        think_generator=generator,
        max_retries=2,
        on_failure="skip",
    )

    merged_rows = read_jsonl(outputs["train"]["sft_merged"])
    assert len(merged_rows) == 1
    assert merged_rows[0]["metadata"]["sample_id"] == "s1"
    assert merged_rows[0]["metadata"]["teacher"]["retries_used"] == 1
    assert generator.calls["s1"] == 2
    assert generator.calls["s2"] == 3

    failures = read_jsonl(outputs["train"]["failure_manifest"])
    assert len(failures) == 1
    assert failures[0]["sample_id"] == "s2"


def test_build_sft_think_data_max_samples_limits_processed_rows(tmp_path: Path):
    from marco.data.build_sft_think_data import build_sft_think_data

    canonical_dir = tmp_path / "canonical"
    write_jsonl(
        canonical_dir / "canonical_mumo_train.jsonl",
        [
            _row("s1", "bbbp+drd2+plogp", "CCN"),
            _row("s2", "bbbp+drd2+plogp", "CCO"),
            _row("s3", "bbbp+drd2+plogp", "CCC"),
        ],
    )

    think = (
        "<think>Increase bbbp with a mild polarity adjustment. "
        "Increase drd2 by preserving key interaction motifs. "
        "Increase plogp through a conservative hydrophobic edit.</think>"
    )
    generator = _DummyThinkGenerator({"s1": [think], "s2": [think], "s3": [think]})

    output_dir = tmp_path / "sft_think"
    outputs = build_sft_think_data(
        canonical_dir=str(canonical_dir),
        output_dir=str(output_dir),
        think_generator=generator,
        max_samples=2,
    )

    merged_rows = read_jsonl(outputs["train"]["sft_merged"])
    assert len(merged_rows) == 2
    assert [row["metadata"]["sample_id"] for row in merged_rows] == ["s1", "s2"]
    assert generator.calls == {"s1": 1, "s2": 1}


def test_build_sft_think_data_local_validation_failure_contains_raw_output(tmp_path: Path):
    from marco.data.build_sft_think_data import build_sft_think_data

    canonical_dir = tmp_path / "canonical"
    write_jsonl(canonical_dir / "canonical_mumo_train.jsonl", [_row("s1", "bbbp+drd2+plogp", "CCN")])

    invalid_raw_output = (
        "<think>This response only talks about drd2 and omits other required property names.</think>"
    )
    generator = _DummyThinkGenerator({"s1": [invalid_raw_output]})

    output_dir = tmp_path / "sft_think"
    outputs = build_sft_think_data(
        canonical_dir=str(canonical_dir),
        output_dir=str(output_dir),
        think_generator=generator,
        max_retries=0,
        on_failure="skip",
    )

    failures = read_jsonl(outputs["train"]["failure_manifest"])
    assert len(failures) == 1
    assert failures[0]["sample_id"] == "s1"
    assert failures[0]["error_type"] == "local_validation_failed"
    assert failures[0]["raw_output_response"] == invalid_raw_output


def test_teacher_prompt_is_strengthened_with_explicit_property_tokens():
    from marco.data.build_sft_think_data import _build_teacher_user_prompt

    prompt = _build_teacher_user_prompt(_row("s1", "bbbp+drd2+plogp", "CCN"), min_sentences=3, max_sentences=5)
    assert "Output exactly one block: <think>...</think>" in prompt
    assert "You MUST use these exact property tokens: bbbp, drd2, plogp" in prompt
    assert "Do not output <SMILES>, </SMILES>, or any SMILES string." in prompt
    assert "Do not include <think> or </think> inside the reasoning content." in prompt


def test_build_sft_think_data_sanitizes_extra_think_markers(tmp_path: Path):
    from marco.data.build_sft_think_data import build_sft_think_data

    canonical_dir = tmp_path / "canonical"
    write_jsonl(canonical_dir / "canonical_mumo_train.jsonl", [_row("s1", "bbbp+drd2+plogp", "CCN")])

    raw_with_extra_tags = (
        "<think>\\<think\\>Increase bbbp with lower polarity. "
        "Increase drd2 by preserving the core scaffold. "
        "Increase plogp with a mild hydrophobic edit.\\</think\\></think>"
    )
    generator = _DummyThinkGenerator({"s1": [raw_with_extra_tags]})

    output_dir = tmp_path / "sft_think"
    outputs = build_sft_think_data(
        canonical_dir=str(canonical_dir),
        output_dir=str(output_dir),
        think_generator=generator,
        on_failure="skip",
    )
    rows = read_jsonl(outputs["train"]["sft_merged"])
    assert len(rows) == 1
    assistant = rows[0]["messages"][1]["content"]
    assert assistant.count("<think>") == 1
    assert assistant.count("</think>") == 1
    assert "\\<think\\>" not in assistant
    assert "\\</think\\>" not in assistant


def test_build_sft_think_data_resume_skips_success_and_retries_failures(tmp_path: Path):
    from marco.data.build_sft_think_data import build_sft_think_data

    canonical_dir = tmp_path / "canonical"
    write_jsonl(
        canonical_dir / "canonical_mumo_train.jsonl",
        [
            _row("s1", "bbbp+drd2+plogp", "CCN"),
            _row("s2", "bbbp+drd2+plogp", "CCO"),
            _row("s3", "bbbp+drd2+plogp", "CCC"),
        ],
    )

    output_dir = tmp_path / "sft_think"
    write_jsonl(output_dir / "sft_train.jsonl", [_existing_sft_row("s1", "bbbp+drd2+plogp", "CCN")])
    write_jsonl(
        output_dir / "sft_train_failures.jsonl",
        [
            {
                "sample_id": "s2",
                "subtask": "bbbp+drd2+plogp",
                "split": "train",
                "status": "teacher_generation_failed",
                "error_type": "teacher_request_failed",
                "error": "temporary network error",
            }
        ],
    )

    think = (
        "<think>Increase bbbp while controlling polarity. "
        "Increase drd2 by preserving key interactions. "
        "Increase plogp with a conservative hydrophobic edit.</think>"
    )
    generator = _DummyThinkGenerator({"s2": [think], "s3": [think]})

    outputs = build_sft_think_data(
        canonical_dir=str(canonical_dir),
        output_dir=str(output_dir),
        think_generator=generator,
        on_failure="skip",
        resume=True,
    )

    merged_rows = read_jsonl(outputs["train"]["sft_merged"])
    merged_ids = [row["metadata"]["sample_id"] for row in merged_rows]
    assert merged_ids == ["s1", "s2", "s3"]
    assert generator.calls == {"s2": 1, "s3": 1}

    failures = read_jsonl(outputs["train"]["failure_manifest"])
    assert failures == []
    assert outputs["train"]["stats"]["skipped_existing_success_rows"] == 1
