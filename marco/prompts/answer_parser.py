from __future__ import annotations

import re
from dataclasses import dataclass

@dataclass
class ParseResult:
    status: str
    smiles: str | None
    reason: str = ""


ANSWER_PATTERNS = [
    re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL),
    re.compile(r"<SMILES>(.*?)</SMILES>", re.IGNORECASE | re.DOTALL),
]


def _normalize_candidate(text: str) -> str:
    text = text.strip()
    text = text.replace("\n", " ").strip()
    return text


def parse_answer_smiles(text: str | None) -> ParseResult:
    if not text or not text.strip():
        return ParseResult(status="invalid_format", smiles=None, reason="empty_output")

    tagged_candidates: list[tuple[int, str]] = []

    for pattern in ANSWER_PATTERNS:
        for match in pattern.finditer(text):
            cand = _normalize_candidate(match.group(1))
            if cand:
                tagged_candidates.append((match.end(), cand))

    if not tagged_candidates:
        return ParseResult(status="invalid_format", smiles=None, reason="no_tagged_candidate")

    tagged_candidates.sort(key=lambda item: item[0])
    return ParseResult(status="ok", smiles=tagged_candidates[-1][1])
