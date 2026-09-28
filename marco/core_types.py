from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Direction = Literal["increase", "decrease"]
PropertyRole = Literal["active", "aux"]
InvalidType = Literal["ok", "invalid_format", "invalid_parse", "invalid_eval", "env_error"]


@dataclass
class PropertyTarget:
    name: str
    direction: Direction
    delta: float
    role: PropertyRole = "active"
    weight: float | None = None
    progress_clip: float | None = None
    source: float | None = None
    target: float | None = None

    def signed_direction(self) -> int:
        return 1 if self.direction == "increase" else -1


@dataclass
class CanonicalRecord:
    sample_id: str
    split: Literal["train", "val", "test"]
    source_repo: str
    benchmark: str
    subtask: str
    instruction: str
    x0_smiles: str
    reference_smiles: str | None
    properties: list[PropertyTarget]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeState:
    episode_id: str
    subtask: str
    x0_smiles: str
    current_smiles: str
    property_targets: list[PropertyTarget]
    x0_predictions: dict[str, float]
    current_predictions: dict[str, float]
    current_total_gap: float
    current_directional_improvements: dict[str, float] = field(default_factory=dict)
    current_progress_score: float = 0.0
    success_turn: int | None = None
    turn_id: int = 0
    max_turns: int = 4
    history: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class StepResult:
    status: InvalidType
    turn_id: int
    candidate_smiles: str | None
    predictions: dict[str, float]
    property_gaps: dict[str, float]
    total_gap: float
    similarity: float
    met_all_targets: bool
    reward_total: float
    reward_gap: float
    reward_similarity: float
    reward_success: float
    should_stop: bool
    message: str = ""
    error_message: str = ""
    error_detail: str = ""
    recovery_hint: str = ""
    directional_improvements: dict[str, float] = field(default_factory=dict)
    progress_components: dict[str, float] = field(default_factory=dict)
    progress_score: float = 0.0
    active_property_names: list[str] = field(default_factory=list)
    success_this_turn: bool = False

    @property
    def invalid_type(self) -> InvalidType:
        return self.status


@dataclass
class RewardConfig:
    alpha: float = 1.0
    lambda_sim: float = 0.5
    in_range_sim_bonus: float = 0.5
    success_bonus: float = 1.0
    failure_penalty: float = 0.0
    invalid_penalty: float = 1.0
    reward_eps: float = 1e-8
    gap_max: float = 2.0
    clip_gap_delta: float = 0.5
    progress_clip: float = 1.0
    similarity_threshold: float = 0.4
    similarity_target_low: float = 0.6
    similarity_target_high: float = 0.95
    similarity_cap: float = 0.9
    over_similarity_penalty: float = 1.0
    progress_min_similarity: float = 0.3
    turn_weight_temperature: float = 1.0
    state_prop_weight: float = 1.0
    state_sim_weight: float = 1.0
    state_success_weight: float = 1.5
    trend_improve_weight: float = 0.5
    trend_regress_weight: float = 0.75
    trend_improve_margin: float = 0.02
    trend_regress_margin: float = 0.02
    terminal_q_weight: float = 0.5


@dataclass
class StopResult:
    should_stop: bool
    reason: str
