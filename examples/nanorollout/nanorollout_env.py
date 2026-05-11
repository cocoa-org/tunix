"""NanoRollout GCE backend wrapped as a Tunix BaseTaskEnv.

Migration of examples/tinyflow_swe/tinyflow_env.py to the new
NanoRollout package (Phase X — in-process import; per api_mapping.md §1).
Imports come from nanorollout.envs.* and nanorollout.harness.* directly.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Optional, cast

import numpy as np

from tunix.rl.agentic.environments.base_environment import BaseTaskEnv, EnvStepResult

# NEW: nanorollout imports (api_mapping.md §1, symbols #1-4).
try:
  from nanorollout.envs.shell_env.gce import GCEEnvironment
  from nanorollout.envs.shell_env._pool import PoolCoordinator
except ImportError:
  GCEEnvironment = cast(Any, None)
  PoolCoordinator = cast(Any, None)

try:
  from nanorollout.harness.runner.swe.common import _run_eval
except ImportError:
  _run_eval = cast(Any, None)

try:
  from nanorollout.harness.agents.swe.openhands.tools.base import FinishSignal
except ImportError:
  FinishSignal = cast(Any, None)

try:
  import grpc
except ImportError:
  grpc = cast(Any, None)


# Hard cap on env close so a hung gRPC StopContainer can't deadlock the
# producer threadpool (lessons.md 2026-04-30 / Phase 3.7). If exceeded,
# the close thread is abandoned (daemon=True) and the run.sh trap's MIG
# resize=0 reclaims the underlying GCE worker.
GCE_STOP_TIMEOUT_SEC = 900  # 15 minutes


def _is_unreachable_image_error(exc: BaseException) -> bool:
  """Match `grpc.aio.AioRpcError(INTERNAL, ...404/manifest...)` from broken images.

  Pre-filter via Docker Hub HEAD is insufficient — manifest index returns 200 but
  blob layers can be missing/broken, so `docker pull` fails at runtime. Narrow
  match: only `INTERNAL + 404/manifest/not-found` patterns. Other AioRpcErrors
  (UNAVAILABLE, DEADLINE_EXCEEDED) still propagate.
  """
  if grpc is None:
    return False
  if not isinstance(exc, grpc.aio.AioRpcError):
    return False
  if exc.code() != grpc.StatusCode.INTERNAL:
    return False
  details = (exc.details() or "").lower()
  return any(p in details for p in ("404", "manifest", "no such image", "not found"))


logger = logging.getLogger(__name__)


# Module-level singleton PoolCoordinator. Per nanorollout/envs/shell_env/_pool.py
# docstring: thread-safe via threading.Lock; safe to share across many
# GCEEnvironment instances each running on its own asyncio loop.
_POOL: Optional["PoolCoordinator"] = None
_POOL_LOCK = threading.Lock()


def _get_pool(mig_name: str, project: str, zone: str) -> "PoolCoordinator":
  global _POOL
  with _POOL_LOCK:
    if _POOL is None:
      assert PoolCoordinator is not None, "nanorollout not importable"
      _POOL = PoolCoordinator(mig_name=mig_name, project=project, zone=zone)
      logger.info("NanoRolloutGCEEnv: built PoolCoordinator(mig=%s zone=%s)", mig_name, zone)
    return _POOL


def _unpack_entry(entry: dict) -> dict:
  """Same as deepswe/swe_env.py: lists-of-1 + numpy scalars become plain values."""
  unpacked = {}
  for k, v in entry.items():
    if isinstance(v, np.ndarray):
      unpacked[k] = v.item()
    elif isinstance(v, list):
      if len(v) != 1:
        raise ValueError(f"Can only convert a list of size 1; got size {len(v)}")
      unpacked[k] = v[0]
    else:
      unpacked[k] = v
  return unpacked


class NanoRolloutGCEEnv(BaseTaskEnv):
  """SWE env executing on a GCE worker MIG via nanorollout GCEEnvironment."""

  def __init__(
      self,
      entry: dict,
      group_id: int | None = None,
      pair_index: int | None = None,
      mig_name: str = "tunix-worker-mig",
      gcp_project: str = "hao-ai-lab-trc",
      gcp_zone: str = "us-west1-a",
      workspace_dir: str = "/testbed",
      step_timeout: int = 120,
      eval_timeout: int = 600,
      max_steps: int = 4,
      verbose: bool = False,
      **kwargs,
  ):
    # BaseTaskEnv.__init__ sets self.task = {}, self.max_steps, self.step_count,
    # self.extra_kwargs (from **kwargs). _model_call writes env.task["policy_version"],
    # so self.task MUST exist (lessons.md 2026-04-30 super().__init__ rule).
    super().__init__(
        task={},
        max_steps=max_steps,
        group_id=group_id,
        pair_index=pair_index,
    )
    self.entry = _unpack_entry(entry)
    self.group_id = group_id
    self.pair_index = pair_index
    self.mig_name = mig_name
    self.gcp_project = gcp_project
    self.gcp_zone = gcp_zone
    self.workspace_dir = workspace_dir
    self.step_timeout = step_timeout
    self.eval_timeout = eval_timeout
    self.verbose = verbose
    self.total_steps = 0
    self._env: Optional["GCEEnvironment"] = None
    self.final_reward_fn = None
    self._skip_reason: Optional[str] = None

  # ----- BaseTaskEnv abstract methods -----

  def _initial_observation(self) -> Any:
    if GCEEnvironment is None:
      raise ImportError(
          "nanorollout.envs.shell_env.gce.GCEEnvironment not importable; "
          "check nanorollout install (uv pip install -e .)"
      )
    image = self.entry.get("docker_image") or self.entry.get("image_name")
    if not image:
      raise ValueError(f"entry missing docker_image/image_name: keys={list(self.entry)}")

    pool = _get_pool(self.mig_name, self.gcp_project, self.gcp_zone)
    try:
      self._env = GCEEnvironment(
          image=image,
          pool=pool,
          workspace_dir=self.workspace_dir,
          timeout=self.step_timeout,
      )
      self._env.start()
    except Exception as exc:
      if _is_unreachable_image_error(exc):
        iid = self.entry.get("instance_id", "<unknown>")
        logger.warning(
            "[skip] instance %s unreachable (docker 404 / manifest): %s",
            iid, str(exc)[:200],
        )
        self._skip_reason = "docker_image_unreachable"
        self._env = None
        return f"[ENV ERROR: instance {iid} unreachable; skipping with reward=0]"
      raise

    self.total_steps = 0
    self.final_reward_fn = self._compute_reward

    problem = self.entry.get("problem_statement") or self.entry.get("problem") or ""
    return problem

  def _step_impl(self, action: Any) -> EnvStepResult:
    """Router: JSON dispatch envelope (oh-core) vs raw bash (legacy fallback)."""
    if self._skip_reason is not None:
      return EnvStepResult(
          observation=f"[skipped: {self._skip_reason}]",
          reward=0,
          done=True,
          info={"skipped": True, "reason": self._skip_reason},
      )
    if self._env is None:
      raise RuntimeError("NanoRolloutGCEEnv not initialized")
    if isinstance(action, str):
      try:
        env_action = json.loads(action)
      except json.JSONDecodeError:
        env_action = None
      if isinstance(env_action, dict) and "tool_name" in env_action:
        return self._step_oh_core(env_action)
    return self._step_bash(action)

  def _step_oh_core(self, env_action: dict) -> EnvStepResult:
    """Dispatch oh-core tool envelope `{tool_name, args}` to env.execute_tool().

    `_format_error` / `finish` / `think` / `task_tracker` are handled in-process
    (not via env round-trip) so they don't waste a turn or hit `tool_missing`.
    """
    name = env_action.get("tool_name")
    args = env_action.get("args") or {}

    if name == "_format_error":
      reason = args.get("reason", "unknown")
      return EnvStepResult(
          observation=f"FORMAT ERROR: {reason}",
          reward=0,
          done=False,
          info={"format_error": reason},
      )

    if name == "finish":
      msg = args.get("message", "")
      reward = self._compute_reward()
      return EnvStepResult(
          observation=f"[Task finished: {msg}]",
          reward=reward,
          done=True,
          info={"finish_message": msg},
      )

    if name == "think":
      thought = args.get("thought", "") or args.get("text", "")
      return EnvStepResult(
          observation=f"[Recorded thought: {str(thought)[:200]}]",
          reward=0,
          done=False,
          info={"think": True},
      )

    if name == "task_tracker":
      tasks = args.get("tasks") or args.get("subtasks") or args
      return EnvStepResult(
          observation=f"[Task tracker noop: {str(tasks)[:200]}]",
          reward=0,
          done=False,
          info={"task_tracker": True},
      )

    try:
      result = self._env.execute_tool(name, **args)
    except Exception as exc:
      if FinishSignal is not None and isinstance(exc, FinishSignal):
        reward = self._compute_reward()
        return EnvStepResult(
            observation=f"[Finished: {exc.message}]",
            reward=reward,
            done=True,
            info={"finish_message": exc.message},
        )
      logger.warning("execute_tool(%s) raised %s: %s", name, type(exc).__name__, exc)
      return EnvStepResult(
          observation=f"[tool error] {type(exc).__name__}: {exc}",
          reward=0,
          done=False,
          info={"error": str(exc)},
      )
    self.total_steps += 1
    return EnvStepResult(
        observation=result.output or "",
        reward=0,
        done=False,
        info={"tool_success": result.success, "tool_name": name},
    )

  def _step_bash(self, action: Any) -> EnvStepResult:
    """Legacy raw-bash path; preserved for ad-hoc debugging."""
    cmd = action if isinstance(action, str) else str(action)
    try:
      result = self._env.execute(cmd, timeout=self.step_timeout)
    except Exception as exc:
      logger.warning("execute raised %s: %s", type(exc).__name__, exc)
      return EnvStepResult(
          observation=f"[env error] {type(exc).__name__}: {exc}",
          reward=0,
          done=False,
          info={"error": str(exc)},
      )
    self.total_steps += 1
    done = "finish" in cmd.lower() and "<function" in cmd.lower()
    return EnvStepResult(
        observation=result.output or "",
        reward=0,
        done=done,
        info={"exit_code": result.exit_code},
    )

  def close(self) -> None:
    if self._env is None:
      return
    captured: list[BaseException] = []
    def _stop() -> None:
      try:
        self._env.stop()
      except BaseException as exc:
        captured.append(exc)
    t = threading.Thread(target=_stop, daemon=True, name="gce-stop")
    t.start()
    t.join(timeout=GCE_STOP_TIMEOUT_SEC)
    if t.is_alive():
      logger.warning(
          "close: gce.stop did not return in %ds; abandoning thread "
          "(MIG resize=0 will reclaim worker).",
          GCE_STOP_TIMEOUT_SEC,
      )
    elif captured:
      exc = captured[0]
      logger.warning("close: stop raised %s: %s", type(exc).__name__, exc)
    self._env = None

  def _compute_reward(self) -> float:
    """Hand the live GCEEnvironment to nanorollout's _run_eval (D1 reward)."""
    if _run_eval is None:
      logger.warning("_run_eval not importable; reward=0")
      return 0.0
    if self._env is None:
      logger.warning("_compute_reward called with no live env; reward=0")
      return 0.0
    try:
      eval_payload, _ = _run_eval(
          env_obj=self._env,
          instance=self.entry,
          eval_timeout=self.eval_timeout,
          workspace_dir=self.workspace_dir,
          dataset="gym",
      )
      r = eval_payload.get("reward", 0)
      return float(r) if r is not None else 0.0
    except Exception as exc:
      logger.warning("_compute_reward raised %s: %s; reward=0", type(exc).__name__, exc)
      return 0.0

  @classmethod
  def from_dict(cls, env_args: dict | str) -> "NanoRolloutGCEEnv":
    if isinstance(env_args, str):
      import json as _json
      env_args = _json.loads(env_args)
    return cls(**env_args)
