from __future__ import annotations

import os

import yaml

from marco.core_types import RewardConfig
from marco.utils import get_env_float


def build_reward_config_from_env() -> RewardConfig:
    lambda_sim = get_env_float("MARCO_REWARD_LAMBDA_SIM", 0.5)
    cfg = RewardConfig(
        alpha=get_env_float("MARCO_REWARD_ALPHA", 1.0),
        lambda_sim=lambda_sim,
        in_range_sim_bonus=get_env_float("MARCO_REWARD_IN_RANGE_SIM_BONUS", lambda_sim),
        success_bonus=get_env_float("MARCO_REWARD_SUCCESS_BONUS", 1.0),
        failure_penalty=get_env_float("MARCO_REWARD_FAILURE_PENALTY", 0.0),
        invalid_penalty=get_env_float("MARCO_REWARD_INVALID_PENALTY", 1.0),
        reward_eps=get_env_float("MARCO_REWARD_EPS", 1e-8),
        gap_max=get_env_float("MARCO_REWARD_GAP_MAX", 2.0),
        clip_gap_delta=get_env_float("MARCO_REWARD_CLIP_GAP_DELTA", 0.5),
        progress_clip=get_env_float("MARCO_DIRECTIONAL_PROGRESS_CLIP", 1.0),
        similarity_threshold=get_env_float("MARCO_SIMILARITY_THRESHOLD", 0.4),
        similarity_target_low=get_env_float("MARCO_SIMILARITY_TARGET_LOW", 0.6),
        similarity_target_high=get_env_float("MARCO_SIMILARITY_TARGET_HIGH", 0.95),
        similarity_cap=get_env_float("MARCO_SIMILARITY_CAP", 0.9),
        over_similarity_penalty=get_env_float("MARCO_REWARD_OVER_SIM_PENALTY", 1.0),
        progress_min_similarity=get_env_float("MARCO_PROGRESS_MIN_SIMILARITY", 0.3),
        state_prop_weight=get_env_float("MARCO_STATE_PROP_WEIGHT", 1.0),
        state_sim_weight=get_env_float("MARCO_STATE_SIM_WEIGHT", 1.0),
        state_success_weight=get_env_float("MARCO_STATE_SUCCESS_WEIGHT", 1.5),
        trend_improve_weight=get_env_float("MARCO_TREND_IMPROVE_WEIGHT", 0.5),
        trend_regress_weight=get_env_float("MARCO_TREND_REGRESS_WEIGHT", 0.75),
        trend_improve_margin=get_env_float("MARCO_TREND_IMPROVE_MARGIN", 0.02),
        trend_regress_margin=get_env_float("MARCO_TREND_REGRESS_MARGIN", 0.02),
        terminal_q_weight=get_env_float("MARCO_TERMINAL_Q_WEIGHT", 0.5),
    )

    cfg_path = os.getenv("MARCO_REWARD_CONFIG")
    if not cfg_path or not os.path.exists(cfg_path):
        return cfg

    with open(cfg_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        return cfg

    for key in cfg.__dataclass_fields__.keys():
        if key in payload:
            setattr(cfg, key, float(payload[key]))
    return cfg
