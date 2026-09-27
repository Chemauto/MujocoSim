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
        self._q_des = self._q_default.copy()
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

    def apply_joint_target(self, q_des) -> None:
        q = np.asarray(q_des, dtype=float)
        if q.shape != self._q_default.shape:
            raise ValueError(f"关节目标维度 {q.shape} != {self._q_default.shape}")
        self._q_des = q.copy()

    def step(self) -> None:
        world = self.world
        q = world.joint_qpos()
        qd = world.joint_qvel()
        if self.mode == "damping":
            ctrl = q.copy() if self._position_actuated else -self.kd * qd
        elif self.mode == "pd_stand":
            ctrl = self._pd(self._q_default, q, qd)
        elif self.mode == "position":
            ctrl = self._pd(self._q_des, q, qd)
        else:
            if self.action_fn is None:
                raise RuntimeError(
                    "motion 模式需要策略推理（rl_controller），当前未接入"
                )
            targets = np.asarray(self.action_fn(world), dtype=float)
            ctrl = self._pd(targets, q, qd)
        world.set_ctrl(ctrl)

    def _pd(self, q_des: np.ndarray, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        if self._position_actuated:
            return q_des.copy()
        qd_des = (
            getattr(self.action_fn, "qd_des", None) if self.mode == "motion" else None
        )
        if qd_des is None:
            return self.kp * (q_des - q) - self.kd * qd
        return self.kp * (q_des - q) + self.kd * (np.asarray(qd_des, dtype=float) - qd)
