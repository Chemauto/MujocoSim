from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime

from mujocosim.config import ConfigError, RobotMeta
from mujocosim.world import SimWorld

from .action_mapper import gains_in_motor_order, map_actions
from .sources import assemble_obs
from .wiring import PolicyProfile


class PolicyRunner:
    def __init__(self, profile: PolicyProfile, policy_dir: Path):
        model_path = policy_dir / profile.model
        if not model_path.is_file():
            raise ConfigError(f"找不到策略模型: {model_path}")
        self.profile = profile
        self.session = onnxruntime.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._input_name = self.session.get_inputs()[0].name
        self._output_name = self.session.get_outputs()[0].name
        in_dim = self.session.get_inputs()[0].shape[-1]
        out_dim = self.session.get_outputs()[0].shape[-1]
        if isinstance(in_dim, int) and in_dim != profile.obs_dim:
            raise ConfigError(
                f"ONNX 输入维度 {in_dim} != 接线表 obs_dim {profile.obs_dim}"
            )
        if isinstance(out_dim, int) and out_dim != profile.action_dim:
            raise ConfigError(
                f"ONNX 输出维度 {out_dim} != 接线表 action_dim {profile.action_dim}"
            )

    def infer(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        out = self.session.run([self._output_name], {self._input_name: obs})[0]
        return np.asarray(out, dtype=float).reshape(-1)


class RlController:
    def __init__(
        self,
        profile: PolicyProfile,
        meta: RobotMeta,
        world: SimWorld,
        policy_dir: Path,
        command_dim: int = 3,
        suite=None,
    ):
        self.profile = profile
        self.meta = meta
        self.world = world
        self.command_dim = command_dim
        self.suite = suite
        self.runner = PolicyRunner(profile, policy_dir)
        self.command = np.zeros(command_dim, dtype=float)
        self.last_action = np.zeros(profile.action_dim, dtype=float)
        self.kp, self.kd = gains_in_motor_order(
            profile.actions, meta, profile.joint_order
        )

    def reset(self) -> None:
        self.last_action[:] = 0.0

    def set_command(self, command) -> None:
        command = np.asarray(command, dtype=float)
        if command.shape != (self.command_dim,):
            raise ValueError(f"command 维度 {command.shape} != ({self.command_dim},)")
        self.command = command

    def __call__(self, world: SimWorld) -> np.ndarray:
        obs = assemble_obs(
            self.profile.observations,
            world,
            self.meta,
            self.profile.joint_order,
            self.last_action,
            self.command,
            self.suite.latest() if self.suite is not None else None,
        )
        action = self.runner.infer(obs)
        self.last_action = action.copy()
        return map_actions(
            action, self.profile.actions, self.meta, self.profile.joint_order
        )
