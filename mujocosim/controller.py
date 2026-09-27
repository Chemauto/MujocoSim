from __future__ import annotations

from typing import Callable

import numpy as np

from .config import CONTROLLER_MODES, SimConfig
from .world import SimWorld


class RobotController:
    def __init__(
        self,
        cfg: SimConfig,
        world: SimWorld,
        action_fn: Callable[[SimWorld], np.ndarray] | None = None,
    ):
        self.cfg = cfg
        self.world = world
        self.action_fn = action_fn
        self.mode = cfg.controller.mode
        n = len(cfg.robot_meta.joint_names)
        self.kp = np.full(n, float(cfg.controller.kp), dtype=float)
        self.kd = np.full(n, float(cfg.controller.kd), dtype=float)
        self._q_default = np.array(cfg.robot_meta.default_joint_pos, dtype=float)
        self._q_target = self._q_default.copy()
        self._q_des = self._q_default.copy()
        self._pd_q_des = self._q_default.copy()
        self._slew = float(cfg.controller.slew_rate) * float(cfg.timing.control_dt)
        self._position_actuated = cfg.robot_meta.control == "position"

    def set_gains(self, kp, kd) -> None:
        kp_arr = np.asarray(kp, dtype=float)
        kd_arr = np.asarray(kd, dtype=float)
        if kp_arr.ndim == 0:
            kp_arr = np.full_like(self.kp, float(kp_arr))
        if kd_arr.ndim == 0:
            kd_arr = np.full_like(self.kd, float(kd_arr))
        if kp_arr.shape != self.kp.shape or kd_arr.shape != self.kd.shape:
            raise ValueError("kp/kd 维度与关节数不符")
        self.kp = kp_arr
        self.kd = kd_arr

    def set_mode(self, mode: str) -> None:
        if mode not in CONTROLLER_MODES:
            raise ValueError(f"未知控制模式 '{mode}'，可选 {sorted(CONTROLLER_MODES)}")
        self.mode = mode
        if mode in ("pd_stand", "position"):
            self._q_des = self.world.joint_qpos().copy()
            self._q_target = (
                self._q_default.copy() if mode == "pd_stand" else self._q_des.copy()
            )

    def apply_joint_target(self, q_des) -> None:
        q = np.asarray(q_des, dtype=float)
        if q.shape != self._q_default.shape:
            raise ValueError(f"关节目标维度 {q.shape} != {self._q_default.shape}")
        self._q_target = q.copy()

    def _slew_toward_target(self) -> None:
        delta = np.clip(self._q_target - self._q_des, -self._slew, self._slew)
        self._q_des = self._q_des + delta

    def step(self) -> None:
        """控制步：确定本周期的位置目标。力矩由 pd_refresh() 在每个物理步刷新。"""
        world = self.world
        if self.mode == "damping":
            if self._position_actuated:
                world.set_ctrl(world.joint_qpos().copy())
            else:
                world.set_ctrl(-self.kd * world.joint_qvel())
            return
        if self.mode in ("pd_stand", "position"):
            self._slew_toward_target()
            self._pd_q_des = self._q_des.copy()
        else:
            if self.action_fn is None:
                raise RuntimeError(
                    "motion 模式需要策略推理（rl_controller），当前未接入"
                )
            targets = np.asarray(self.action_fn(world), dtype=float)
            if targets.shape != self._q_default.shape:
                raise ValueError(
                    f"关节目标维度 {targets.shape} != {self._q_default.shape}"
                )
            self._pd_q_des = targets
        if self._position_actuated:
            world.set_ctrl(self._pd_q_des.copy())
        else:
            self.pd_refresh()

    def pd_refresh(self) -> None:
        """每个物理步用最新 q/qd 刷新 PD 力矩（位置目标保持本控制周期的值）。

        与训练侧 IsaacLab 隐式执行器一致：PD 在物理步频率上求解，
        而非 50Hz 零阶保持，否则力矩滞后一个控制周期，腿部会下沉/抖动。
        """
        if self._position_actuated or self.mode == "damping":
            return
        q = self.world.joint_qpos()
        qd = self.world.joint_qvel()
        self.world.set_ctrl(self.kp * (self._pd_q_des - q) - self.kd * qd)
