from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DataSource:
    def get_samples(self, num_samples: int):
        raise NotImplementedError

    def add_samples(self, samples):
        raise NotImplementedError

    def save(self, rollout_id):
        raise NotImplementedError

    def load(self, rollout_id=None):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


@dataclass
class Sample:
    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"
        FAILED = "failed"

    group_index: int | None = None
    index: int | None = None
    prompt: str | list[dict[str, str]] = ""
    tokens: list[int] = field(default_factory=list)
    response: str = ""
    response_length: int = 0
    label: str | None = None
    reward: float | dict[str, Any] | None = None
    loss_mask: list[int] | None = None
    status: Status = Status.PENDING
    metadata: dict[str, Any] = field(default_factory=dict)
    train_metadata: dict[str, Any] | None = None
    rollout_log_probs: list[float] | None = None
    remove_sample: bool = False
