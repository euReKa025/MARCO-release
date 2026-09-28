from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop, register
from verl.interactions.base import BaseInteraction
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput


def _resolve_interaction_kwargs(
    *,
    extra_info: dict[str, Any],
    raw_interaction_kwargs: Any,
    messages: list[dict[str, Any]],
    reward_model: Any,
    interaction_map: dict[str, BaseInteraction],
    default_max_turns: int | None = None,
) -> tuple[str, dict[str, Any]]:
    interaction_kwargs = dict(raw_interaction_kwargs or {}) if isinstance(raw_interaction_kwargs, dict) else {}
    explicit_name = "name" in interaction_kwargs
    if not explicit_name:
        if "marco" in interaction_map:
            interaction_kwargs["name"] = "marco"
        elif "repo" in interaction_map:
            interaction_kwargs["name"] = "repo"
        elif interaction_map:
            interaction_kwargs["name"] = next(iter(interaction_map.keys()))
        else:  # pragma: no cover - defensive guard
            raise ValueError("interaction map is empty")

    if "record" not in interaction_kwargs:
        interaction_kwargs["record"] = {
            "prompt": list(messages),
            "reward_model": reward_model if isinstance(reward_model, dict) else {},
            "extra_info": dict(extra_info),
        }
    if "extra_info" not in interaction_kwargs:
        interaction_kwargs["extra_info"] = dict(extra_info)
    if "max_turns" not in interaction_kwargs:
        resolved_max_turns = None
        if isinstance(default_max_turns, int) and default_max_turns > 0:
            resolved_max_turns = default_max_turns
        elif extra_info.get("max_turns") is not None:
            resolved_max_turns = int(extra_info.get("max_turns"))
        if resolved_max_turns is not None:
            interaction_kwargs["max_turns"] = int(resolved_max_turns)

    interaction_name = str(interaction_kwargs.get("name"))
    if interaction_name not in interaction_map:
        if explicit_name:
            available = ", ".join(sorted(interaction_map.keys()))
            raise ValueError(f"Unknown interaction name '{interaction_name}'. Available: {available}")
        if not interaction_map:  # pragma: no cover - defensive guard
            raise ValueError("interaction map is empty")
        interaction_name = next(iter(interaction_map.keys()))
        interaction_kwargs["name"] = interaction_name
    return interaction_name, interaction_kwargs


@register("marco_tool_agent")
class MarcoToolAgentLoop(ToolAgentLoop):
    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        interaction: BaseInteraction | None = None
        interaction_kwargs = {}
        if self.interaction_config_file:
            extra_info = kwargs.get("extra_info")
            if not isinstance(extra_info, dict):
                extra_info = {}
            interaction_name, interaction_kwargs = _resolve_interaction_kwargs(
                extra_info=extra_info,
                raw_interaction_kwargs=extra_info.get("interaction_kwargs"),
                messages=messages,
                reward_model=kwargs.get("reward_model"),
                interaction_map=self.interaction_map,
                default_max_turns=self.max_assistant_turns,
            )
            interaction = self.interaction_map[interaction_name]
            await interaction.start_interaction(request_id, **interaction_kwargs)

        agent_data = AgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
        )

        try:
            state = AgentState.PENDING
            while state != AgentState.TERMINATED:
                if state == AgentState.PENDING:
                    state = await self._handle_pending_state(agent_data, sampling_params)
                elif state == AgentState.GENERATING:
                    state = await self._handle_generating_state(agent_data, sampling_params)
                elif state == AgentState.PROCESSING_TOOLS:
                    state = await self._handle_processing_tools_state(agent_data)
                elif state == AgentState.INTERACTING:
                    state = await self._handle_interacting_state(agent_data)
                else:
                    state = AgentState.TERMINATED
        finally:
            if interaction is not None:
                await interaction.finalize_interaction(request_id)

        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_output = {}
        if agent_data.image_data is not None:
            multi_modal_output["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            multi_modal_output["videos"] = agent_data.video_data

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_output,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            routed_experts=agent_data.routed_experts,
            extra_fields=agent_data.extra_fields,
            reward_score=agent_data.extra_fields.get("marco_trajectory_reward"),
        )
        output.extra_fields.update({"turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards})
        return output

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        add_messages: list[dict[str, Any]] = []

        with simple_timer("generate_sequences", agent_data.metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
            )
        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        else:
            agent_data.metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

        if not agent_data.extra_fields:
            agent_data.extra_fields.update(output.extra_fields)
        else:
            max_global_steps = output.extra_fields.get("max_global_steps", None)
            if max_global_steps:
                agent_data.extra_fields["max_global_steps"] = max_global_steps

        agent_data.extra_fields["last_stop_reason"] = output.stop_reason
        agent_data.extra_fields["last_response_length"] = len(output.token_ids)

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        hit_response_limit = (not ignore_termination) and len(agent_data.response_mask) >= self.response_length
        hit_assistant_turn_limit = bool(self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns)
        hit_user_turn_limit = bool(self.max_user_turns and agent_data.user_turns >= self.max_user_turns)
        force_terminate_after_interaction = bool(
            hit_response_limit or hit_assistant_turn_limit or hit_user_turn_limit
        )

        tools = [tool.tool_schema for tool in self.tools.values()]
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids, tools)

        if self.interaction_config_file:
            assistant_message = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )
            add_messages.append({"role": "assistant", "content": assistant_message})
            agent_data.messages.extend(add_messages)
            if force_terminate_after_interaction:
                agent_data.extra_fields["force_terminate_after_interaction"] = True

        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        if self.interaction_config_file:
            return AgentState.INTERACTING
        if force_terminate_after_interaction:
            return AgentState.TERMINATED
        return AgentState.TERMINATED

    async def _handle_interacting_state(self, agent_data: AgentData) -> AgentState:
        force_terminate_after_interaction = bool(agent_data.extra_fields.pop("force_terminate_after_interaction", False))
        (
            should_terminate_sequence,
            interaction_response,
            reward,
            additional_data,
        ) = await agent_data.interaction.generate_response(
            agent_data.request_id,
            agent_data.messages,
            stop_reason=agent_data.extra_fields.get("last_stop_reason"),
            response_length=int(agent_data.extra_fields.get("last_response_length", 0) or 0),
            rollout_max_response_len=self.response_length,
            force_terminate=force_terminate_after_interaction,
            **agent_data.interaction_kwargs,
        )
        agent_data.user_turns += 1

        if reward is not None:
            agent_data.turn_scores.append(reward)

        if additional_data:
            marco_turns = agent_data.extra_fields.setdefault("marco_turns", [])
            turn_payload = additional_data.get("turn_payload")
            if turn_payload is not None:
                marco_turns.append(turn_payload)
            if "trajectory_reward" in additional_data:
                agent_data.extra_fields["marco_trajectory_reward"] = additional_data["trajectory_reward"]
            if "reward_breakdowns" in additional_data:
                agent_data.extra_fields["marco_reward_breakdowns"] = additional_data["reward_breakdowns"]
            if "step_rewards" in additional_data:
                agent_data.extra_fields["marco_step_rewards"] = additional_data["step_rewards"]
            if "reward_extra_info" in additional_data:
                agent_data.extra_fields["reward_extra_info"] = additional_data["reward_extra_info"]

        if not should_terminate_sequence and (not force_terminate_after_interaction) and interaction_response:
            add_messages = [{"role": "user", "content": interaction_response}]
            agent_data.messages.extend(add_messages)

            response_ids = await self.apply_chat_template(
                add_messages,
                remove_system_prompt=True,
            )
            agent_data.prompt_ids += response_ids
            agent_data.response_mask += [0] * len(response_ids)
            if agent_data.response_logprobs:
                agent_data.response_logprobs += [0.0] * len(response_ids)

        if should_terminate_sequence or force_terminate_after_interaction:
            return AgentState.TERMINATED
        return AgentState.GENERATING
