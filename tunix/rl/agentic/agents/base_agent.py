# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base classes for Large Language Model powered agents.

This module defines:

* `LLMBaseAgent`: the minimal abstract base class that provides a standard
  interface for agents interacting with LLMs and environments.

* `ConversationAgentBase`: a higher-level base class for chat-style agents
  that maintain conversation history and trajectories. Most concrete agents
  (single-turn, tool-using, gaming, etc.) should subclass this instead of
  `LLMBaseAgent` directly.

* TITO accumulator (`_token_history`): when an agent has TITO enabled
  (`enable_tito=True`), it maintains a per-episode list of token-id
  segments — initial prompt, each assistant turn, each env turn —
  whose concatenation is fed verbatim to vLLM at the next turn. This
  guarantees byte-equality between rollout-time vLLM input and
  trainer-time `conversation_tokens`. Mirrors miles
  `submodules/miles/uda/swe_agent/proxy/tito_state.py:TaskState`.
"""

import abc
import asyncio
import copy
from typing import Any, Dict, List, Optional

import numpy as np

from tunix.rl.agentic.agents import agent_types


class LLMBaseAgent(abc.ABC):
  """Abstract base class for Large Language Model powered agents."""

  # ──────────────────────────────────────────────────────────────
  # State Access Properties
  # ──────────────────────────────────────────────────────────────

  @property
  @abc.abstractmethod
  def chat_completions(self) -> list[dict[str, str]]:
    """Get the current conversation context for the LLM."""
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def trajectory(self) -> agent_types.Trajectory:
    """Get the complete trajectory for the current task/episode."""
    raise NotImplementedError

  # ──────────────────────────────────────────────────────────────
  # Environment Interaction Interface
  # ──────────────────────────────────────────────────────────────

  @abc.abstractmethod
  def update_from_env(
      self,
      observation: Any,
      reward: float,
      done: bool,
      info: Dict[str, Any] | None = None,
      **kwargs,
  ) -> None:
    """Process feedback from environment after action execution."""
    raise NotImplementedError("update_from_env is not implemented.")

  async def update_from_env_async(self, *args, **kwargs) -> None:
    """Asynchronous version of update_from_env."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, self.update_from_env, *args, **kwargs
    )

  # ──────────────────────────────────────────────────────────────
  # Model Interaction Interface
  # ──────────────────────────────────────────────────────────────

  @abc.abstractmethod
  def update_from_model(self, response: str, **kwargs) -> agent_types.Action:
    """Process LLM response and extract structured action."""
    raise NotImplementedError("update_from_model is not implemented.")

  async def update_from_model_async(
      self, *args, **kwargs
  ) -> agent_types.Action:
    """Asynchronous version of update_from_model."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, self.update_from_model, *args, **kwargs
    )

  # ──────────────────────────────────────────────────────────────
  # Lifecycle Management
  # ──────────────────────────────────────────────────────────────

  @abc.abstractmethod
  def reset(self) -> None:
    """Reset agent state for a new episode."""
    ...

  # ──────────────────────────────────────────────────────────────
  # Debugging and Introspection
  # ──────────────────────────────────────────────────────────────

  def get_current_step(self) -> agent_types.Step | None:
    """Get the most recent step for debugging and introspection."""
    if not self.trajectory.steps:
      return None
    return self.trajectory.steps[-1]


class ConversationAgentBase(LLMBaseAgent):
  """Base class for chat-style LLM agents with trajectory support.

  This class implements common functionality for agents that:
  * Maintain a list of chat messages (`_messages`) to send to the LLM.
  * Maintain a `Trajectory` of `Step` objects for RL training.
  * Cache the last environment observation for step recording.

  Subclasses are expected to:
  * Provide a system prompt via constructor.
  * Implement `_observation_to_messages()` to convert environment observations
    into chat messages.
  * Implement `update_from_model()` to parse LLM responses into `Action`s and
    append new `Step`s to the trajectory.
  """

  def __init__(self, system_prompt: str):
    self.system_prompt = system_prompt
    self._trajectory = agent_types.Trajectory()
    self._messages: list[dict[str, Any]] = []
    self._init_messages(system_prompt)
    self.step = 0

    # TITO accumulator state (inert until configure_tito is called).
    self.enable_tito: bool = False
    self._token_history: list[np.ndarray] = []
    self._tito_logprob_history: list[np.ndarray] = []
    self._assistant_prefix_ids: Optional[List[int]] = None
    self._im_end_token_id: Optional[int] = None
    self._newline_token_id: Optional[int] = None

  # ---------- TITO support (mirrors miles TaskState) ----------

  def configure_tito(
      self,
      *,
      assistant_prefix_ids: List[int],
      im_end_token_id: int,
      newline_token_id: int,
  ) -> None:
    """Enable TITO accumulator and seed boundary tokens (im_end, newline,
    assistant prefix). See plan.md Phase 0/3."""
    self.enable_tito = True
    self._assistant_prefix_ids = list(assistant_prefix_ids)
    self._im_end_token_id = int(im_end_token_id)
    self._newline_token_id = int(newline_token_id)

  def append_init_tokens(
      self,
      prompt_tokens: np.ndarray,
  ) -> None:
    """Seed the accumulator with the initial prompt tokens (called by
    trajectory engine after _reset)."""
    if not self.enable_tito:
      return
    arr = np.asarray(prompt_tokens, dtype=np.int32)
    self._token_history.append(arr)
    self._tito_logprob_history.append(
        np.zeros(arr.shape[0], dtype=np.float32)
    )

  def append_assistant_response(
      self,
      content_tokens: np.ndarray,
      content_logprobs: Optional[np.ndarray] = None,
  ) -> None:
    """Append assistant turn with miles-style suffix wrap [im_end, newline].
    Defensively strips trailing eos. content_logprobs may be None for
    legacy callers; suffix positions get 0.0 logprob."""
    if not self.enable_tito:
      return
    if self._im_end_token_id is None or self._newline_token_id is None:
      raise RuntimeError(
          "TITO is enabled but boundary-token constants are not set;"
          " call configure_tito(...) first."
      )
    content_tokens = np.asarray(content_tokens, dtype=np.int32)
    if content_logprobs is None:
      content_logprobs = np.zeros(content_tokens.shape[0], dtype=np.float32)
    else:
      content_logprobs = np.asarray(content_logprobs, dtype=np.float32)
    if content_tokens.shape[0] != content_logprobs.shape[0]:
      raise ValueError(
          f"content_tokens length ({content_tokens.shape[0]}) does not"
          f" match content_logprobs length ({content_logprobs.shape[0]})"
      )

    # Defensive eos strip (matches miles tito_state.py:34-46).
    if (
        content_tokens.shape[0] > 0
        and int(content_tokens[-1]) == self._im_end_token_id
    ):
      content_tokens = content_tokens[:-1]
      content_logprobs = content_logprobs[:-1]

    suffix_ids = np.array(
        [self._im_end_token_id, self._newline_token_id], dtype=np.int32
    )
    suffix_logprobs = np.zeros(suffix_ids.shape[0], dtype=np.float32)

    wrapped = np.concatenate([content_tokens, suffix_ids])
    wrapped_logprobs = np.concatenate([content_logprobs, suffix_logprobs])
    self._token_history.append(wrapped)
    self._tito_logprob_history.append(wrapped_logprobs)

  def append_env_observation_tokens(
      self,
      env_tokens: np.ndarray,
  ) -> None:
    """Append env observation tokens. Drops leading newline if prior
    segment already ends in newline (avoids double-newline at the role
    boundary; preserves byte-equality with apply_chat_template)."""
    if not self.enable_tito:
      return
    arr = np.asarray(env_tokens, dtype=np.int32)
    if (
        arr.shape[0] > 0
        and self._newline_token_id is not None
        and int(arr[0]) == self._newline_token_id
        and self._token_history
        and self._token_history[-1].shape[0] > 0
        and int(self._token_history[-1][-1]) == self._newline_token_id
    ):
      arr = arr[1:]
    self._token_history.append(arr)
    self._tito_logprob_history.append(
        np.zeros(arr.shape[0], dtype=np.float32)
    )

  @property
  def token_prompt_for_next_turn(self) -> List[int]:
    """Concatenated TITO token-id stream (Python ints) for the next vLLM call."""
    if not self._token_history:
      return []
    return np.concatenate(self._token_history).astype(np.int32).tolist()

  @property
  def all_logprobs(self) -> np.ndarray:
    """Rollout logprobs aligned with token_prompt_for_next_turn (float32, zero at non-content positions)."""
    if not self._tito_logprob_history:
      return np.zeros(0, dtype=np.float32)
    return np.concatenate(self._tito_logprob_history).astype(np.float32)

  # ---------- Internal helpers ----------

  def _init_messages(self, system_prompt: str) -> None:
    """Initialize conversation history with a system prompt.

    Subclasses may override this to inject additional content (e.g., tool
    documentation) into the initial system message.

    Args:
      system_prompt: The system prompt to use.
    """
    self._messages = [{"role": "system", "content": system_prompt or ""}]

  def _observation_to_messages(
      self, observation: Any, reward: float, done: bool, info: Dict[str, Any]
  ) -> None:
    """Convert environment observation into chat messages.

    Default behavior:
    * If observation is a dict containing "question", use it as user content.
    * If observation is a string, append as a user message.
    * Otherwise, do nothing.

    Subclasses can override this to handle richer observation formats.

    Args:
      observation: The observation from the environment.
      reward: The reward from the environment.
      done: Whether the episode is done.
      info: Additional information from the environment.
    """
    del reward, done, info  # Unused in default implementation.
    # prompts should not be applied with template beforehand to avoid double
    # templating.
    if isinstance(observation, dict) and "prompts" in observation:
      self._messages.append(
          {"role": "user", "content": observation["prompts"] or ""}
      )
    elif isinstance(observation, dict) and "question" in observation:
      self._messages.append(
          {"role": "user", "content": observation["question"] or ""}
      )
    elif isinstance(observation, str):
      self._messages.append({"role": "user", "content": observation})

  # ---------- Properties ----------

  @property
  def chat_completions(self) -> list[dict[str, str]]:
    return self._messages

  @property
  def trajectory(self) -> agent_types.Trajectory:
    return self._trajectory

  # ---------- Public interface implementations ----------

  def update_from_env(
      self,
      observation: Any,
      reward: float,
      done: bool,
      info: Dict[str, Any] | None = None,
      **kwargs,
  ) -> None:
    """Update current step with environment feedback and extend conversation."""
    # First observation from env is the task specification.
    if self._trajectory.task is None:
      if isinstance(observation, str):
        self._trajectory.task = {"prompts": [observation]}
      else:
        self._trajectory.task = copy.deepcopy(observation)

    step = self.get_current_step()
    if step:
      step.observation = observation
      step.reward = reward
      step.done = done
      step.info = info or {}

    # Let subclass / default handler convert observation into messages.
    if observation is not None:
      self._observation_to_messages(observation, reward, done, info)

  def reset(self) -> None:
    """Reset trajectory, cache, conversation history, and TITO accumulator."""
    self._trajectory = agent_types.Trajectory()
    self._init_messages(self.system_prompt)
    self.step = 0
    # Preserve TITO config (enable_tito + constants) across episodes;
    # only the per-episode segment lists are cleared.
    self._token_history = []
    self._tito_logprob_history = []
