from __future__ import annotations

from dataclasses import asdict

from marco.core_types import EpisodeState, RewardConfig, StepResult
from marco.env.predictor_client import PredictorClient
from marco.env.property_utils import (
    compute_directional_improvements,
    compute_directional_progress_score,
    parse_property_targets,
    split_property_targets,
)
from marco.env.similarity import tanimoto_similarity
from marco.env.stop_rules import should_stop_episode
from marco.errors import InvalidEvaluationError, PredictorServiceError
from marco.prompts.system_prompt import get_feedback_response_instruction


class MoleculeEnv:
    def __init__(self, predictor_client: PredictorClient, reward_config: RewardConfig):
        self.predictor = predictor_client
        self.cfg = reward_config

    @staticmethod
    def _active_names(targets) -> list[str]:
        active, _ = split_property_targets(targets)
        return [t.name for t in active]

    @staticmethod
    def _naive_directional_gap(
        directional_improvements: dict[str, float],
        active_names: list[str],
    ) -> float:
        # Legacy logging field only; not used for reward in V1.
        if not active_names:
            return 0.0
        vals = [max(0.0, -float(directional_improvements.get(name, 0.0))) for name in active_names]
        return float(sum(vals) / len(vals))

    def init_episode(self, *, record: dict, max_turns: int) -> EpisodeState:
        targets = parse_property_targets(record)
        x0_smiles = record["x0_smiles"]
        required = [t.name for t in targets]
        x0_predictions = self.predictor.predict(x0_smiles, required)
        directional_improvements = compute_directional_improvements(
            predictions=x0_predictions,
            x0_predictions=x0_predictions,
            targets=targets,
        )
        progress_score, progress_components, success = compute_directional_progress_score(
            directional_improvements=directional_improvements,
            targets=targets,
            default_clip=self.cfg.progress_clip,
        )
        active_names = self._active_names(targets)
        total_gap = self._naive_directional_gap(directional_improvements, active_names)

        state = EpisodeState(
            episode_id=record["sample_id"],
            subtask=record["subtask"],
            x0_smiles=x0_smiles,
            current_smiles=x0_smiles,
            property_targets=targets,
            x0_predictions=x0_predictions,
            current_predictions=x0_predictions,
            current_total_gap=total_gap,
            current_directional_improvements=directional_improvements,
            current_progress_score=progress_score,
            turn_id=0,
            max_turns=max_turns,
            history=[
                {
                    "turn_id": 0,
                    "candidate_smiles": x0_smiles,
                    "predictions": x0_predictions,
                    "property_gaps": {},
                    "total_gap": total_gap,
                    "directional_improvements": directional_improvements,
                    "progress_components": progress_components,
                    "progress_score": progress_score,
                    "similarity": 1.0,
                    "reward": 0.0,  # reward is post-hoc in V1.
                    "invalid_type": "ok",
                    "met_all_targets": success,
                }
            ],
        )
        return state

    def step_invalid(self, *, state: EpisodeState, turn_id: int, invalid_type: str, message: str) -> StepResult:
        default_messages = {
            "invalid_format": "Could not extract a single SMILES from model output.",
            "invalid_parse": "Extracted SMILES failed RDKit parse/sanitize.",
            "invalid_eval": "Parsed SMILES but failed downstream evaluation.",
        }
        recovery_hints = {
            "invalid_format": get_feedback_response_instruction(None, require_valid_smiles=True),
            "invalid_parse": get_feedback_response_instruction(None, require_valid_smiles=True),
            "invalid_eval": get_feedback_response_instruction(None, require_valid_smiles=True),
        }
        error_message = message.strip() if message and message.strip() else default_messages.get(invalid_type, "Invalid action.")
        stop = should_stop_episode(
            turn_id=turn_id,
            max_turns=state.max_turns,
            invalid_type=invalid_type,
            met_all_targets=False,
            similarity_acceptable=False,
        )
        active_names = self._active_names(state.property_targets)
        return StepResult(
            status=invalid_type,
            turn_id=turn_id,
            candidate_smiles=None,
            predictions={},
            property_gaps={},
            total_gap=state.current_total_gap,
            similarity=0.0,
            met_all_targets=False,
            reward_total=0.0,  # invalid penalty is applied post-hoc in trajectory settlement.
            reward_gap=0.0,
            reward_similarity=0.0,
            reward_success=0.0,
            should_stop=stop.should_stop,
            message=stop.reason,
            error_message=error_message,
            error_detail=message.strip() if message else "",
            recovery_hint=recovery_hints.get(invalid_type, get_feedback_response_instruction(None, require_valid_smiles=True)),
            directional_improvements={},
            progress_components={},
            progress_score=0.0,
            active_property_names=active_names,
            success_this_turn=False,
        )

    def step(self, *, state: EpisodeState, candidate_smiles: str, turn_id: int) -> StepResult:
        required = [t.name for t in state.property_targets]
        try:
            pred = self.predictor.predict(candidate_smiles, required)
            similarity = tanimoto_similarity(candidate_smiles, state.x0_smiles)
            directional_improvements = compute_directional_improvements(
                predictions=pred,
                x0_predictions=state.x0_predictions,
                targets=state.property_targets,
            )
            progress_score, progress_components, met_all = compute_directional_progress_score(
                directional_improvements=directional_improvements,
                targets=state.property_targets,
                default_clip=self.cfg.progress_clip,
            )
        except PredictorServiceError:
            try:
                similarity = tanimoto_similarity(state.current_smiles, state.x0_smiles)
            except Exception:  # noqa: BLE001
                similarity = 0.0
            stop = should_stop_episode(
                turn_id=turn_id,
                max_turns=state.max_turns,
                invalid_type="env_error",
                met_all_targets=False,
                similarity_acceptable=False,
            )
            return StepResult(
                status="env_error",
                turn_id=turn_id,
                candidate_smiles=candidate_smiles,
                predictions=dict(state.current_predictions),
                property_gaps={},
                total_gap=state.current_total_gap,
                similarity=similarity,
                met_all_targets=False,
                reward_total=0.0,
                reward_gap=0.0,
                reward_similarity=0.0,
                reward_success=0.0,
                should_stop=stop.should_stop,
                message=stop.reason,
                error_message="Predictor service is unavailable. This is treated as env_error.",
                error_detail="predictor_service_error",
                recovery_hint="Retry this step later; do not penalize the model for env_error.",
                directional_improvements=dict(state.current_directional_improvements),
                progress_components={},
                progress_score=float(state.current_progress_score),
                active_property_names=self._active_names(state.property_targets),
                success_this_turn=False,
            )
        except InvalidEvaluationError as exc:
            return self.step_invalid(state=state, turn_id=turn_id, invalid_type="invalid_eval", message=str(exc))

        similarity_min = float(self.cfg.similarity_target_low)
        similarity_copy_threshold = 0.99
        similarity_acceptable = similarity >= similarity_min and similarity < similarity_copy_threshold
        stop = should_stop_episode(
            turn_id=turn_id,
            max_turns=state.max_turns,
            invalid_type="ok",
            met_all_targets=met_all,
            similarity_acceptable=similarity_acceptable,
        )
        active_names = self._active_names(state.property_targets)
        total_gap = self._naive_directional_gap(directional_improvements, active_names)

        return StepResult(
            status="ok",
            turn_id=turn_id,
            candidate_smiles=candidate_smiles,
            predictions=pred,
            property_gaps={},
            total_gap=total_gap,
            similarity=similarity,
            met_all_targets=met_all,
            reward_total=0.0,  # post-hoc reward settlement.
            reward_gap=0.0,
            reward_similarity=0.0,
            reward_success=0.0,
            should_stop=stop.should_stop,
            message=stop.reason,
            directional_improvements=directional_improvements,
            progress_components=progress_components,
            progress_score=progress_score,
            active_property_names=active_names,
            success_this_turn=met_all,
        )

    def apply_step(self, state: EpisodeState, result: StepResult) -> EpisodeState:
        state.turn_id = result.turn_id
        if result.status == "ok" and result.candidate_smiles is not None:
            state.current_smiles = result.candidate_smiles
            state.current_predictions = dict(result.predictions)
            state.current_total_gap = float(result.total_gap)
            state.current_directional_improvements = dict(result.directional_improvements)
            state.current_progress_score = float(result.progress_score)
            if result.success_this_turn and state.success_turn is None:
                state.success_turn = result.turn_id

        state.history.append(
            {
                "turn_id": result.turn_id,
                "candidate_smiles": result.candidate_smiles,
                "predictions": result.predictions,
                "property_gaps": result.property_gaps,
                "total_gap": result.total_gap,
                "directional_improvements": result.directional_improvements,
                "progress_components": result.progress_components,
                "progress_score": result.progress_score,
                "similarity": result.similarity,
                "reward": result.reward_total,
                "reward_gap": result.reward_gap,
                "reward_similarity": result.reward_similarity,
                "reward_success": result.reward_success,
                "invalid_type": result.status,
                "met_all_targets": result.met_all_targets,
                "message": result.message,
                "error_message": result.error_message,
                "error_detail": result.error_detail,
                "recovery_hint": result.recovery_hint,
            }
        )
        return state

    @staticmethod
    def state_to_payload(state: EpisodeState) -> dict:
        payload = asdict(state)
        payload["property_targets"] = [asdict(x) for x in state.property_targets]
        return payload
