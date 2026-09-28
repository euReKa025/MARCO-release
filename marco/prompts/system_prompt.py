from __future__ import annotations

import os
from typing import Any

from marco.utils import get_env_float


PROMPT_MODE_THINK_ANSWER = "think_answer"
PROMPT_MODE_DIRECT_SMILES = "direct_smiles"

THINK_ANSWER_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. "
    "The user asks a question, and the Assistant solves it. "
    "The assistant first thinks briefly about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and molecule are enclosed within <think> </think> and <SMILES> </SMILES> tags, respectively, "
    "i.e., <think> reasoning process here </think><SMILES> molecule here </SMILES>. "
    "Keep the <think> block concise and stop immediately after the closing </SMILES> tag."
)

DIRECT_SMILES_SYSTEM_PROMPT = (
    "You are a molecular optimization assistant. "
    "Return exactly one modified molecule in the tag format requested by the user. "
    "Do not explain your reasoning. Do not output markdown, bullet points, or multiple candidates. "
    "Do not output any text outside the single tagged molecule."
)

SYSTEM_PROMPT = THINK_ANSWER_SYSTEM_PROMPT

_PROMPTS = {
    PROMPT_MODE_THINK_ANSWER: THINK_ANSWER_SYSTEM_PROMPT,
    PROMPT_MODE_DIRECT_SMILES: DIRECT_SMILES_SYSTEM_PROMPT,
}

_INPUT_MESSAGE_KEYS = {
    PROMPT_MODE_THINK_ANSWER: "input_messages_think_answer",
    PROMPT_MODE_DIRECT_SMILES: "input_messages_direct_smiles",
}


def normalize_prompt_mode(prompt_mode: str | None, *, default: str = PROMPT_MODE_THINK_ANSWER) -> str:
    mode = str(prompt_mode or os.getenv("MARCO_PROMPT_MODE", default)).strip().lower()
    if mode not in _PROMPTS:
        raise ValueError(f"Unsupported prompt_mode: {mode}")
    return mode


def get_system_prompt(prompt_mode: str | None = None, *, default: str = PROMPT_MODE_THINK_ANSWER) -> str:
    return _PROMPTS[normalize_prompt_mode(prompt_mode, default=default)]


def get_input_messages_key(prompt_mode: str | None = None, *, default: str = PROMPT_MODE_DIRECT_SMILES) -> str:
    return _INPUT_MESSAGE_KEYS[normalize_prompt_mode(prompt_mode, default=default)]


def get_similarity_target_range() -> tuple[float, float]:
    low = get_env_float("MARCO_SIMILARITY_TARGET_LOW", 0.6)
    high = get_env_float("MARCO_SIMILARITY_TARGET_HIGH", 0.95)
    if low > high:
        low, high = high, low
    return low, high


def format_similarity_target_range() -> str:
    low, high = get_similarity_target_range()
    return f"[{low:.2f}, {high:.2f}]"


def get_user_response_instruction(prompt_mode: str | None = None, *, default: str = PROMPT_MODE_THINK_ANSWER) -> str:
    mode = normalize_prompt_mode(prompt_mode, default=default)
    if mode == PROMPT_MODE_THINK_ANSWER:
        return (
            "Respond with exactly one brief <think>...</think> block followed by exactly one "
            "<SMILES>...</SMILES> block. Keep the <think> block to one short sentence. "
            "Stop immediately after </SMILES>. Do not output multiple <SMILES> tags."
        )
    return "Return only <SMILES>SMILES</SMILES>."


def get_feedback_response_instruction(
    prompt_mode: str | None = None,
    *,
    require_valid_smiles: bool,
    default: str = PROMPT_MODE_THINK_ANSWER,
) -> str:
    mode = normalize_prompt_mode(prompt_mode, default=default)
    if mode == PROMPT_MODE_THINK_ANSWER:
        answer_phrase = (
            "exactly one valid molecule in exactly one <SMILES>...</SMILES> block"
            if require_valid_smiles
            else "exactly one <SMILES>...</SMILES> block"
        )
        return (
            "Please output exactly one brief <think>...</think> block followed by "
            f"{answer_phrase}. Keep the <think> block to one short sentence. "
            "Stop immediately after </SMILES>. Do not output multiple <SMILES> tags."
        )
    if require_valid_smiles:
        return "Please output exactly one valid molecule as <SMILES>SMILES</SMILES>."
    return "Please output exactly one molecule as <SMILES>SMILES</SMILES>."


def build_system_user_messages(
    user_content: str,
    *,
    prompt_mode: str | None = None,
    default: str = PROMPT_MODE_THINK_ANSWER,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": get_system_prompt(prompt_mode, default=default)},
        {"role": "user", "content": str(user_content)},
    ]


def render_messages(messages: list[dict[str, Any]], tokenizer: Any, *, add_generation_prompt: bool = True) -> str:
    return str(
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    )
