from __future__ import annotations

import math
import os
import re
from collections import defaultdict

from marco.trainer_types import Sample
from marco.utils import get_env_float, get_env_int

try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover
    AutoTokenizer = None


def _load_tokenizer(hf_checkpoint: str):
    if AutoTokenizer is None:
        return None
    return AutoTokenizer.from_pretrained(hf_checkpoint, trust_remote_code=True)


ANSWER_RE = re.compile(r"<(?P<tag>answer|SMILES)>.*?</(?P=tag)>", re.IGNORECASE | re.DOTALL)
MODEL_INVALID_TYPES = {"invalid_format", "invalid_parse", "invalid_eval"}


def _build_loss_mask(sample: Sample, tokenizer) -> list[int]:
    response_len = int(sample.response_length)
    if response_len <= 0:
        return []

    response_text = sample.response or ""
    match = ANSWER_RE.search(response_text)
    if match is None:
        return [0] * response_len

    if tokenizer is None:
        # Fallback when tokenizer is unavailable: approximate with ratio.
        full_len = max(len(response_text), 1)
        answer_len = max(len(match.group(0)), 1)
        n = max(1, int(response_len * answer_len / full_len))
        return [0] * (response_len - n) + [1] * n

    prefix_text = response_text[: match.start()]
    answer_text = match.group(0)
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)

    start = min(len(prefix_ids), response_len)
    end = min(start + len(answer_ids), response_len)

    mask = [0] * response_len
    for i in range(start, end):
        mask[i] = 1
    return mask


def _trajectory_id(sample: Sample, idx: int) -> str:
    md = sample.train_metadata or {}
    return str(md.get("trajectory_id") or md.get("episode_id", f"traj-{idx}"))


def _group_id(sample: Sample, idx: int, trajectory_id: str) -> str:
    md = sample.train_metadata or {}
    return str(md.get("group_id") or md.get("episode_id", f"group-{idx}-{trajectory_id}"))


def _compute_trajectory_advantages(
    trajectory_returns: dict[str, float],
    trajectory_group: dict[str, str],
    eps: float,
) -> dict[str, float]:
    grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for traj_id, ret in trajectory_returns.items():
        grouped[trajectory_group.get(traj_id, traj_id)].append((traj_id, ret))

    trajectory_advantages: dict[str, float] = {}
    for _, rows in grouped.items():
        values = [ret for _, ret in rows]
        mean = sum(values) / max(1, len(values))
        variance = sum((ret - mean) ** 2 for ret in values) / max(1, len(values))
        std = math.sqrt(max(variance, 0.0))
        for traj_id, ret in rows:
            trajectory_advantages[traj_id] = (ret - mean) / (std + eps)
    return trajectory_advantages


def _compute_similarity_quality(
    *,
    similarity: float,
    similarity_min: float,
    similarity_copy_threshold: float,
    low_weight: float,
    copy_penalty_weight: float,
) -> float:
    if similarity < similarity_min:
        return -low_weight * float(similarity_min - similarity) / max(similarity_min, 1e-6)
    if similarity < similarity_copy_threshold:
        return float(similarity - similarity_min) / max(similarity_copy_threshold - similarity_min, 1e-6)
    return 1.0 - copy_penalty_weight * float(similarity - similarity_copy_threshold) / max(
        1.0 - similarity_copy_threshold,
        1e-6,
    )


def _compute_quality_score(
    *,
    property_quality: float,
    similarity_quality: float,
    met_all_targets: bool,
    similarity_acceptable: bool,
    prop_weight: float,
    sim_weight: float,
    success_bonus: float,
) -> float:
    gated_success = float(success_bonus) if (met_all_targets and similarity_acceptable) else 0.0
    return float(prop_weight) * property_quality + float(sim_weight) * similarity_quality + gated_success


def _settle_posthoc_rewards(samples: list[Sample]) -> tuple[
    dict[int, float],
    dict[int, dict[str, float]],
    dict[str, float],
    dict[str, str],
    dict[int, dict[str, int | bool]],
]:
    invalid_penalty = get_env_float("MARCO_REWARD_INVALID_PENALTY", 1.0)
    similarity_target_low = get_env_float("MARCO_SIMILARITY_TARGET_LOW", 0.6)
    similarity_target_high = get_env_float("MARCO_SIMILARITY_TARGET_HIGH", 0.75)
    similarity_copy_threshold = get_env_float("MARCO_SIMILARITY_COPY_THRESHOLD", 0.99)
    similarity_low_weight = get_env_float("MARCO_SIMILARITY_LOW_WEIGHT", 1.0)
    similarity_copy_penalty_weight = get_env_float("MARCO_SIMILARITY_COPY_PENALTY_WEIGHT", 2.0)
    state_success_bonus = get_env_float("MARCO_QUALITY_SUCCESS_BONUS", 0.15)
    progress_min_similarity = get_env_float("MARCO_PROGRESS_MIN_SIMILARITY", 0.3)
    state_prop_weight = get_env_float("MARCO_STATE_PROP_WEIGHT", 1.0)
    state_sim_weight = get_env_float("MARCO_STATE_SIM_WEIGHT", 1.0)
    trend_improve_weight = get_env_float("MARCO_TREND_IMPROVE_WEIGHT", 0.5)
    trend_regress_weight = get_env_float("MARCO_TREND_REGRESS_WEIGHT", 0.75)
    trend_improve_margin = get_env_float("MARCO_TREND_IMPROVE_MARGIN", 0.02)
    trend_regress_margin = get_env_float("MARCO_TREND_REGRESS_MARGIN", 0.02)
    # Legacy vars remain readable for compatibility but are inactive in the new reward math.
    get_env_float("MARCO_TERMINAL_Q_WEIGHT", 0.5)
    get_env_float("MARCO_REWARD_FAILURE_PENALTY", 0.0)
    length_penalty_threshold = get_env_int("MARCO_LENGTH_PENALTY_THRESHOLD", 256)
    length_penalty_value = get_env_float("MARCO_LENGTH_PENALTY_VALUE", 0.2)
    get_env_int("ROLLOUT_MAX_RESPONSE_LEN", 512)
    get_env_float("MARCO_REWARD_SUCCESS_BONUS", 1.0)
    if similarity_target_low > similarity_target_high:
        similarity_target_low, similarity_target_high = similarity_target_high, similarity_target_low

    grouped: dict[str, list[tuple[int, Sample]]] = defaultdict(list)
    trajectory_group: dict[str, str] = {}
    for idx, sample in enumerate(samples):
        traj_id = _trajectory_id(sample, idx)
        grouped[traj_id].append((idx, sample))
        trajectory_group.setdefault(traj_id, _group_id(sample, idx, traj_id))

    step_rewards: dict[int, float] = {}
    breakdowns: dict[int, dict[str, float]] = {}
    trajectory_returns: dict[str, float] = {}
    step_meta: dict[int, dict[str, int | bool]] = {}

    for traj_id, items in grouped.items():
        items.sort(key=lambda it: int(((it[1].train_metadata or {}).get("turn_id", 0))))
        if not items:
            trajectory_returns[traj_id] = 0.0
            continue

        prev_q = 0.0
        best_q = 0.0
        prev_property_state = 0.0
        prev_similarity_state = 0.0
        prev_success_state = 0.0
        success_awarded = False
        success_turn: int | None = None
        max_turns = int(((items[0][1].train_metadata or {}).get("max_turns", len(items))))
        trajectory_return = 0.0
        quality_candidates: list[tuple[int, int, float]] = []

        for pos, (idx, sample) in enumerate(items):
            md = sample.train_metadata or {}
            turn_id = int(md.get("turn_id", pos + 1))
            status = str(md.get("invalid_type", "ok"))
            similarity = float(md.get("similarity", 0.0))
            met_all_targets = bool(md.get("met_all_targets", False))
            similarity_acceptable = similarity >= similarity_target_low and similarity < similarity_copy_threshold

            property_state = prev_property_state
            similarity_state = prev_similarity_state
            success_state = prev_success_state
            q_t = prev_q

            reward_state_quality = 0.0
            reward_progress = 0.0
            reward_similarity = 0.0
            reward_success_state = 0.0
            reward_success = 0.0
            reward_improve = 0.0
            reward_regress = 0.0
            reward_invalid = 0.0
            reward_terminal_q = 0.0
            reward_failure = 0.0
            reward_length = 0.0
            improve_t = 0.0
            regress_t = 0.0

            if status in MODEL_INVALID_TYPES:
                r_total = -abs(invalid_penalty)
                reward_invalid = float(r_total)
            elif status == "env_error":
                r_total = 0.0
            else:
                property_state = float(md.get("progress_score", prev_property_state))
                if similarity < progress_min_similarity and property_state > 0.0:
                    property_state = 0.0
                property_state = max(-1.0, min(1.0, property_state))

                similarity_state = _compute_similarity_quality(
                    similarity=similarity,
                    similarity_min=similarity_target_low,
                    similarity_copy_threshold=similarity_copy_threshold,
                    low_weight=similarity_low_weight,
                    copy_penalty_weight=similarity_copy_penalty_weight,
                )
                success_state = float(state_success_bonus) if (met_all_targets and similarity_acceptable) else 0.0
                q_t = _compute_quality_score(
                    property_quality=property_state,
                    similarity_quality=similarity_state,
                    met_all_targets=met_all_targets,
                    similarity_acceptable=similarity_acceptable,
                    prop_weight=state_prop_weight,
                    sim_weight=state_sim_weight,
                    success_bonus=state_success_bonus,
                )
                response_length = int(getattr(sample, "response_length", 0) or 0)
                if response_length > int(length_penalty_threshold):
                    reward_length = -abs(float(length_penalty_value))
                    q_t += reward_length

                if pos == 0:
                    improve_t = max(0.0, q_t - float(trend_improve_margin))
                    regress_t = 0.0
                else:
                    improve_t = max(0.0, q_t - best_q - float(trend_improve_margin))
                    regress_t = max(0.0, prev_q - q_t - float(trend_regress_margin))

                reward_progress = float(state_prop_weight) * property_state
                reward_similarity = float(state_sim_weight) * similarity_state
                reward_success_state = float(success_state)
                reward_state_quality = float(q_t)
                reward_improve = float(trend_improve_weight) * improve_t
                reward_regress = -float(trend_regress_weight) * regress_t
                r_total = float(q_t) + reward_improve + reward_regress

                if met_all_targets and similarity_acceptable and not success_awarded:
                    success_awarded = True
                    success_turn = turn_id
                quality_candidates.append((idx, turn_id, float(q_t)))

            is_last_turn = pos == len(items) - 1

            step_rewards[idx] = float(r_total)
            breakdowns[idx] = {
                "reward_total": float(r_total),
                "reward_state_quality": float(reward_state_quality),
                "reward_progress": float(reward_progress),
                "reward_property_state": float(reward_progress),
                "reward_similarity": float(reward_similarity),
                "reward_success_state": float(reward_success_state),
                "reward_success": float(reward_success),
                "reward_improve": float(reward_improve),
                "reward_regress": float(reward_regress),
                "reward_terminal_q": float(reward_terminal_q),
                "reward_failure": float(reward_failure),
                "reward_invalid": float(reward_invalid),
                "reward_length": float(reward_length),
                "property_state_score": float(property_state),
                "similarity_state_score": float(similarity_state),
                "success_state_score": float(success_state),
                "state_quality_score": float(q_t),
                "trend_improve": float(improve_t),
                "trend_regress": float(regress_t),
            }
            trajectory_return += float(r_total)

            step_meta[idx] = {
                "effective_horizon": int(len(items)),
                "is_effective_last_turn": bool(is_last_turn),
                "posthoc_truncated_by_success": bool(success_turn is not None and success_turn < max_turns),
                "success_turn": int(success_turn) if success_turn is not None else -1,
            }

            prev_q = q_t
            prev_property_state = property_state
            prev_similarity_state = similarity_state
            prev_success_state = success_state
            best_q = max(best_q, q_t)

        trajectory_returns[traj_id] = float(trajectory_return)
        if quality_candidates:
            best_quality_idx, best_quality_turn, _ = max(quality_candidates, key=lambda row: (row[2], row[1]))
        else:
            best_quality_idx, best_quality_turn = items[-1][0], int(
                ((items[-1][1].train_metadata or {}).get("turn_id", len(items)))
            )
        for idx, _sample in items:
            step_meta.setdefault(idx, {})
            step_meta[idx]["is_best_quality_turn"] = bool(idx == best_quality_idx)
            step_meta[idx]["best_quality_turn"] = int(best_quality_turn)

    return step_rewards, breakdowns, trajectory_returns, trajectory_group, step_meta


def convert_samples_to_train_data(args, samples):
    if not samples:
        return {
            "tokens": [],
            "response_lengths": [],
            "rewards": [],
            "raw_reward": [],
            "truncated": [],
            "sample_indices": [],
            "loss_masks": [],
            "round_number": [],
            "metadata": [],
        }

    tokenizer = None
    if getattr(args, "hf_checkpoint", None):
        try:
            tokenizer = _load_tokenizer(args.hf_checkpoint)
        except Exception:
            tokenizer = None

    adv_eps = float(os.getenv("MARCO_TRAJECTORY_ADV_EPS", get_env_float("MARCO_TRAJECTORY_ADV_EPS", 1e-6)))

    step_rewards, breakdowns, trajectory_returns, trajectory_group, step_meta = _settle_posthoc_rewards(samples)
    trajectory_advantages = _compute_trajectory_advantages(trajectory_returns, trajectory_group, adv_eps)
    train_turn_filter = os.getenv("MARCO_TRAIN_TURN_FILTER", "all").strip().lower().replace("-", "_")
    if train_turn_filter not in {"all", "best_quality"}:
        train_turn_filter = "all"

    raw_rewards: list[float] = []
    rewards: list[float] = []
    loss_masks = []
    metadata = []

    for idx, sample in enumerate(samples):
        traj_id = _trajectory_id(sample, idx)
        step_reward = float(step_rewards.get(idx, 0.0))
        raw_rewards.append(step_reward)

        md = dict(sample.train_metadata or {})
        md["trajectory_id"] = traj_id
        md["group_id"] = trajectory_group.get(traj_id, md.get("group_id", ""))
        md["trajectory_reward"] = float(trajectory_returns.get(traj_id, 0.0))
        md["trajectory_return"] = float(trajectory_returns.get(traj_id, 0.0))
        md["trajectory_advantage"] = float(trajectory_advantages.get(traj_id, 0.0))
        md["step_reward"] = step_reward
        md.update(breakdowns.get(idx, {}))
        md.update(step_meta.get(idx, {}))
        turn_filter_masked = bool(
            train_turn_filter == "best_quality" and not bool(md.get("is_best_quality_turn", False))
        )
        md["train_turn_filter"] = train_turn_filter
        md["turn_filter_masked"] = turn_filter_masked

        if str(md.get("invalid_type", "ok")) == "env_error":
            sample.remove_sample = True
            md["skip_training"] = True

        sample.train_metadata = md
        metadata.append(md)

        mask = _build_loss_mask(sample, tokenizer)
        if sample.remove_sample or turn_filter_masked:
            mask = [0] * len(mask)
        sample.loss_mask = mask
        loss_masks.append(mask)

        rewards.append(
            0.0 if (sample.remove_sample or turn_filter_masked) else float(trajectory_advantages.get(traj_id, 0.0))
        )

    round_numbers = [int((sample.train_metadata or {}).get("turn_id", 1)) for sample in samples]

    train_data = {
        "tokens": [sample.tokens for sample in samples],
        "response_lengths": [sample.response_length for sample in samples],
        "rewards": rewards,
        "raw_reward": raw_rewards,
        "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
        "sample_indices": [sample.index for sample in samples],
        "loss_masks": loss_masks,
        "round_number": round_numbers,
        "metadata": metadata,
    }
    return train_data
