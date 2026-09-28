from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any


TRAINING_ENV_KEYS = {
    "CONFIG_PATH": "config_path",
    "CONFIG_NAME": "config_name",
    "PROJECT_NAME": "project_name",
    "EXPERIMENT_NAME": "experiment_name",
    "EXPERIMENT_ROOT_NAME": "experiment_root_name",
    "SUBTASK_NAME": "subtask_name",
    "MODEL_PATH": "model_path",
    "TRAIN_PARQUET": "train_parquet",
    "VAL_PARQUET": "val_parquet",
    "OUTPUT_DIR": "output_dir",
    "LOG_DIR": "log_dir",
    "TRAIN_LOG_PATH": "train_log_path",
    "TRAIN_BATCH_SIZE": "train_batch_size",
    "MAX_PROMPT_LENGTH": "max_prompt_length",
    "MAX_RESPONSE_LENGTH": "max_response_length",
    "TOTAL_EPOCHS": "total_epochs",
    "TOTAL_TRAINING_STEPS": "total_training_steps",
    "TRAINER_TOTAL_EPOCHS": "trainer_total_epochs",
    "TRAINER_TEST_FREQ": "trainer_test_freq",
    "TRAINER_SAVE_FREQ": "trainer_save_freq",
    "TRAINER_RESUME_MODE": "trainer_resume_mode",
    "TRAINER_VAL_BEFORE_TRAIN": "trainer_val_before_train",
    "MAX_TURNS": "max_turns",
    "MARCO_MAX_TURNS": "marco_max_turns",
    "ROLLOUT_N": "rollout_n",
    "ROLLOUT_TEMPERATURE": "rollout_temperature",
    "ROLLOUT_TOP_P": "rollout_top_p",
    "ROLLOUT_MAX_RESPONSE_LEN": "rollout_max_response_len",
    "ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE": "rollout_tensor_model_parallel_size",
    "ROLLOUT_GPU_MEMORY_UTILIZATION": "rollout_gpu_memory_utilization",
    "AGENT_NUM_WORKERS": "agent_num_workers",
    "ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU": "actor_ppo_micro_batch_size_per_gpu",
    "ACTOR_PPO_MINI_BATCH_SIZE": "actor_ppo_mini_batch_size",
    "LOG_PROB_MICRO_BATCH_SIZE_PER_GPU": "log_prob_micro_batch_size_per_gpu",
    "ACTOR_LR": "actor_lr",
    "ACTOR_LR_SCHEDULER_TYPE": "actor_lr_scheduler_type",
    "ACTOR_LR_WARMUP_STEPS_RATIO": "actor_lr_warmup_steps_ratio",
    "ACTOR_CHECKPOINT_LOAD_CONTENTS": "actor_checkpoint_load_contents",
    "USE_KL_LOSS": "use_kl_loss",
    "KL_LOSS_COEF": "kl_loss_coef",
    "KL_LOSS_TYPE": "kl_loss_type",
    "NNODES": "nnodes",
    "N_GPUS_PER_NODE": "n_gpus_per_node",
    "CUDA_VISIBLE_DEVICES": "cuda_visible_devices",
    "LOGGER": "logger",
    "WANDB_MODE": "wandb_mode",
}

EVALUATION_ENV_KEYS = {
    "TRAINER_VALIDATION_DATA_DIR": "trainer_validation_data_dir",
    "TRAINER_EVAL_ROLLOUT_DIR": "trainer_eval_rollout_dir",
    "TRAINER_EVAL_SUBTASK": "trainer_eval_subtask",
    "TRAINER_EVAL_PARTITION": "trainer_eval_partition",
    "TRAINER_EVAL_MODE": "trainer_eval_mode",
    "VALIDATION_MODE": "validation_mode",
    "VALIDATION_DO_SAMPLE": "validation_do_sample",
    "VALIDATION_TEMPERATURE": "validation_temperature",
    "VALIDATION_TOP_P": "validation_top_p",
    "VALIDATION_N": "validation_n",
    "VAL_MAX_SAMPLES": "val_max_samples",
    "VALIDATION_SHUFFLE": "validation_shuffle",
    "VALIDATION_EVAL_WATCH_POLL_SECONDS": "validation_eval_watch_poll_seconds",
    "TRAINER_RAW_EVAL_ROOT_DIR": "trainer_raw_eval_root_dir",
    "TRAINER_RAW_EVAL_SIMILARITY_THRESHOLD": "trainer_raw_eval_similarity_threshold",
    "TRAINER_RAW_EVAL_SIMILARITY_TARGET_LOW": "trainer_raw_eval_similarity_target_low",
    "TRAINER_RAW_EVAL_SIMILARITY_TARGET_HIGH": "trainer_raw_eval_similarity_target_high",
    "TRAINER_RAW_EVAL_SIMILARITY_COPY_THRESHOLD": "trainer_raw_eval_similarity_copy_threshold",
}

PREDICTOR_ENV_KEYS = {
    "MARCO_ADMET_API": "marco_admet_api",
    "MARCO_DRD2_API": "marco_drd2_api",
    "MARCO_PREDICTOR_CACHE_PATH": "marco_predictor_cache_path",
    "MARCO_PREDICTOR_TIMEOUT_S": "marco_predictor_timeout_s",
    "MARCO_PREDICTOR_MAX_RETRIES": "marco_predictor_max_retries",
    "MARCO_MOCK_PREDICTOR": "marco_mock_predictor",
    "PREDICTOR_HOST": "predictor_host",
    "ADMET_PORT": "admet_port",
    "DRD2_PORT": "drd2_port",
    "START_PREDICTORS": "start_predictors",
    "ALLOW_EXISTING_SERVERS": "allow_existing_servers",
    "RESTART_UNHEALTHY_PREDICTORS": "restart_unhealthy_predictors",
    "PREDICTOR_REQUIRE_SEMANTIC_PROBE": "predictor_require_semantic_probe",
    "POST_SERVER_READY_SLEEP": "post_server_ready_sleep",
}

REWARD_ENV_PREFIXES = (
    "MARCO_REWARD_",
    "MARCO_DIRECTIONAL_",
    "MARCO_SIMILARITY_",
    "MARCO_PROGRESS_",
    "MARCO_PROPERTY_",
    "MARCO_LENGTH_",
    "MARCO_STATE_",
    "MARCO_QUALITY_",
    "MARCO_TREND_",
    "MARCO_TERMINAL_",
    "MARCO_TRAJECTORY_",
    "MARCO_TRAIN_",
    "MARCO_REPO_",
)


def _coerce_value(value: str) -> Any:
    text = str(value).strip()
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        if text and all(ch not in text for ch in ".eE"):
            return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return value


def _collect_mapped_env(env: Mapping[str, str], keys: Mapping[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for env_key, metadata_key in keys.items():
        value = env.get(env_key)
        if value is None or str(value).strip() == "":
            continue
        out[metadata_key] = _coerce_value(str(value))
    return out


def _collect_reward_env(env: Mapping[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in sorted(env):
        if not key.startswith(REWARD_ENV_PREFIXES):
            continue
        value = env.get(key)
        if value is None or str(value).strip() == "":
            continue
        out[key] = _coerce_value(str(value))
    return out


def _merge_dicts(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def parse_run_metadata_json(value: str | None) -> dict[str, Any]:
    if value is None or not str(value).strip():
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return {"parse_error": "invalid_json", "raw": value}
    return dict(payload) if isinstance(payload, dict) else {"parse_error": "json_not_object", "raw": value}


def collect_run_metadata_from_env(
    env: Mapping[str, str] | None = None,
    *,
    metadata_json: str | None = None,
) -> dict[str, Any]:
    source = os.environ if env is None else env
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "training": _collect_mapped_env(source, TRAINING_ENV_KEYS),
        "evaluation": _collect_mapped_env(source, EVALUATION_ENV_KEYS),
        "reward": _collect_reward_env(source),
        "predictor": _collect_mapped_env(source, PREDICTOR_ENV_KEYS),
    }
    override = parse_run_metadata_json(metadata_json or source.get("TRAINER_EVAL_RUN_METADATA_JSON"))
    if override:
        metadata = _merge_dicts(metadata, override)
    return metadata
