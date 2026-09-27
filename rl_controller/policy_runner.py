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
        cmd_smoothing: float = 0.0,
    ):
        self.profile = profile
        self.meta = meta
        self.world = world
        self.command_dim = command_dim
        self.suite = suite
        self.cmd_smoothing = float(cmd_smoothing)
        self._cmd_target = np.zeros(command_dim, dtype=float)
        self.motion = None
        if profile.motion:
            from .motion import MotionClip, MotionContext

            self.motion = MotionContext(MotionClip(policy_dir / profile.motion))
        self.runner = PolicyRunner(profile, policy_dir)
        self.command = np.zeros(command_dim, dtype=float)
        self.last_action = np.zeros(profile.action_dim, dtype=float)
        self.qd_des: np.ndarray | None = None
        self.kp, self.kd = gains_in_motor_order(
            profile.actions, meta, profile.joint_order
        )
        self.reset()

    def reset(self) -> None:
        self.last_action[:] = 0.0
        if self.motion is None:
            return
        self.motion.reset()
        clip = self.motion.clip
        world = self.world
        with world.lock:
            if world._free_jid is not None:
                adr = int(world.model.jnt_qposadr[world._free_jid])
                pos, quat = clip.anchor_pose()
                world.data.qpos[adr : adr + 3] = pos
                world.data.qpos[adr + 3 : adr + 7] = quat
            jp, _ = clip.commands()
            order = self.profile.joint_order or self.meta.joint_names
            for pol_i, name in enumerate(order):
                motor_i = self.meta.joint_names.index(name)
                world.data.qpos[world._joint_qadr[motor_i]] = jp[pol_i]
            world.data.qvel[:] = 0.0
            import mujoco

            mujoco.mj_forward(world.model, world.data)

    def set_command(self, command) -> None:
        command = np.asarray(command, dtype=float)
        if command.shape != (self.command_dim,):
            raise ValueError(f"command 维度 {command.shape} != ({self.command_dim},)")
        self._cmd_target = command.copy()

    def _smooth_command(self) -> None:
        if self.cmd_smoothing > 0.0:
            a = self.cmd_smoothing
            self.command = self.command + a * (self._cmd_target - self.command)
        else:
            self.command = self._cmd_target.copy()

    def __call__(self, world: SimWorld) -> np.ndarray:
        self._smooth_command()
        obs = assemble_obs(
            self.profile.observations,
            world,
            self.meta,
            self.profile.joint_order,
            self.last_action,
            self.command,
            self.suite.latest() if self.suite is not None else None,
            self.motion,
        )
        action = self.runner.infer(obs)
        self.last_action = action.copy()
        jv = None
        if self.motion is not None:
            _, jv = self.motion.clip.commands()
            self.motion.clip.advance()
        targets = map_actions(
            action, self.profile.actions, self.meta, self.profile.joint_order
        )
        if jv is not None:
            order = self.profile.joint_order or self.meta.joint_names
            qd = np.zeros(len(self.meta.joint_names))
            for pol_i, name in enumerate(order):
                qd[self.meta.joint_names.index(name)] = jv[pol_i]
            self.qd_des = qd
        return targets
