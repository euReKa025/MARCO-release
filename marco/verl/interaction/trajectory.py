from __future__ import annotations

from typing import Any, Protocol

from marco.core_types import EpisodeState
from marco.env.molecule_env import MoleculeEnv
from marco.env.predictor_client import PredictorClient
from marco.prompts.system_prompt import PROMPT_MODE_THINK_ANSWER, build_system_user_messages
from marco.reward.config import build_reward_config_from_env
from marco.verl.data.schema import dataset_instruction_from_record


class EpisodeBootstrapEnv(Protocol):
    def init_episode(self, *, record: dict[str, Any], max_turns: int) -> EpisodeState: ...

    @staticmethod
    def state_to_payload(state: EpisodeState) -> dict[str, Any]: ...


def bootstrap_trajectory_state(
    row: dict[str, Any],
    *,
    max_turns: int,
    env: EpisodeBootstrapEnv | None = None,
    prompt_mode: str = PROMPT_MODE_THINK_ANSWER,
) -> dict[str, Any]:
    record = dict((row.get("extra_info") or {}).get("canonical_record") or {})
    runtime_env = env or MoleculeEnv(PredictorClient(), build_reward_config_from_env())
    state = runtime_env.init_episode(record=record, max_turns=max_turns)
    payload = runtime_env.state_to_payload(state)
    instruction = dataset_instruction_from_record(record)
    payload["instruction"] = instruction
    payload["messages"] = build_system_user_messages(instruction, prompt_mode=prompt_mode)
    return payload
