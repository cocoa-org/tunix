"""NanoRollout oh-core agent — Tunix-compatible wrap of OpenHands CodeActAgent.

Migration of examples/tinyflow_swe/tinyflow_agent.py to nanorollout (Phase X
in-process). Drives oh-core's 5-tool schema (execute_bash, str_replace_editor,
task_tracker, think, finish) through tunix's ConversationAgentBase.
"""

from __future__ import annotations

import json
from typing import Any, Optional

# NEW: nanorollout imports (api_mapping.md §1, symbols #5-6).
from nanorollout.harness.agents.swe.openhands.prompts import get_system_prompt
# Symbol #6: function `get_default_tools` was renamed to `build_tools`.
from nanorollout.harness.agents.swe.openhands.tools import build_tools as get_default_tools

from tunix.rl.agentic.agents.agent_types import Action, Step
from tunix.rl.agentic.agents.base_agent import ConversationAgentBase
from vllm.tool_parsers import ToolParserManager


_TOOL_CALL_OUTPUT_INSTRUCTION = (
    "For each function call, return a JSON object with function name and "
    "arguments within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
    "</tool_call>\n"
)


def _build_tools_block(tools) -> str:
    """Render oh-core tool schemas into a `<tools>...</tools>` block + format instruction.

    Tunix's chat_template parsers (`Default` / `Qwen`) don't pass `tools=` to
    `apply_chat_template`, so we can't rely on the model's chat_template to
    inject tool schemas — they have to live verbatim inside SYSTEM_PROMPT.
    """
    schemas = [t.to_openai_schema() for t in tools]
    block = "<tools>\n"
    for s in schemas:
        block += json.dumps(s) + "\n"
    block += "</tools>\n\n"
    block += _TOOL_CALL_OUTPUT_INSTRUCTION
    return block


class NanoRolloutOhCoreAgent(ConversationAgentBase):
    """Drives oh-core's 5-tool schema via tunix `ConversationAgentBase`.

    Constructor matches tunix's agent instantiation pattern
    (`agentic_rl_learner.py:375-377`):
      `agent_class(**{"system_prompt": algo_config.system_prompt, **agent_kwargs})`
    Injected `system_prompt` kwarg is accepted but overridden by the oh-core
    prompt + tools block we build here.
    """

    def __init__(
        self,
        tokenizer_path: str = "Qwen/Qwen3-4B-Instruct-2507",
        workspace_dir: str = "/testbed",
        enable_tito: bool = False,
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        del system_prompt, kwargs  # intentionally unused
        from transformers import AutoTokenizer  # lazy: tokenizer load cheap on cache hit

        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self._tools = get_default_tools(workspace_mount_path_in_sandbox=workspace_dir)
        self._tool_map = {t.name: t for t in self._tools}

        # get_system_prompt signature gained `prompt_variant='core'` (kwarg)
        # — 1-arg call site preserves backward-compat (api_mapping.md §1 #5).
        base_prompt = get_system_prompt(tokenizer_path)
        full_prompt = base_prompt + "\n\n# Tools\n\n" + _build_tools_block(self._tools)
        super().__init__(system_prompt=full_prompt)

        parser_cls = ToolParserManager.get_tool_parser("hermes")
        self._parser = parser_cls(tokenizer=self._tokenizer, tools=[])

        if enable_tito:
            no_gen = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": "."}],
                tokenize=False,
                add_generation_prompt=False,
            )
            with_gen = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": "."}],
                tokenize=False,
                add_generation_prompt=True,
            )
            assistant_prefix_ids = self._tokenizer.encode(
                with_gen[len(no_gen):], add_special_tokens=False
            )
            im_end_token_id = self._tokenizer.convert_tokens_to_ids("<|im_end|>")
            newline_token_id = self._tokenizer.encode("\n")[-1]
            self.configure_tito(
                assistant_prefix_ids=assistant_prefix_ids,
                im_end_token_id=im_end_token_id,
                newline_token_id=newline_token_id,
            )

    def _observation_to_messages(
        self, observation: Any, reward: float, done: bool, info: dict[str, Any]
    ) -> None:
        del reward, done, info
        self._messages.append({"role": "user", "content": str(observation)})

    def update_from_env(
        self,
        observation: Any,
        reward: float,
        done: bool,
        info: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        observation = str(observation)
        if info is None:
            info = {}
        super().update_from_env(observation, reward, done, info, **kwargs)
        self._cur_step = Step(observation=observation)

    def update_from_model(self, response: str, **kwargs) -> Action:
        del kwargs
        # 1. Append in-flight cur_step before any other state mutation.
        self._trajectory.steps.append(self._cur_step)
        cur_step = self._trajectory.steps[-1]
        cur_step.model_response = response

        # 2. Persist assistant turn verbatim for next prompt round's chat_template.
        self._messages.append({"role": "assistant", "content": response})
        self.step += 1

        # 3. Extract structured tool_call(s) via vLLM's hermes parser.
        result = self._parser.extract_tool_calls(response, request=None)
        if not result.tools_called or not result.tool_calls:
            envelope = json.dumps({
                "tool_name": "_format_error",
                "args": {"reason": "no <tool_call> emitted"},
            })
            cur_step.thought = result.content or response
            cur_step.action = envelope
            return Action(action=envelope)

        # 4. Dispatch envelope: take the first call (oh-core convention).
        tc = result.tool_calls[0]
        name = tc.function.name
        try:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
        except json.JSONDecodeError as exc:
            envelope = json.dumps({
                "tool_name": "_format_error",
                "args": {"reason": f"bad json: {exc}"},
            })
            cur_step.thought = result.content or ""
            cur_step.action = envelope
            return Action(action=envelope)
        if name not in self._tool_map and name != "finish":
            envelope = json.dumps({
                "tool_name": "_format_error",
                "args": {"reason": f"unknown tool: {name}"},
            })
            cur_step.thought = result.content or ""
            cur_step.action = envelope
            return Action(action=envelope)

        envelope = json.dumps({"tool_name": name, "args": args})
        cur_step.thought = result.content or ""
        cur_step.action = envelope
        return Action(action=envelope)
