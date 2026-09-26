"""JoystickController：对外的优雅入口——封装手柄+映射+积分一步到位。

用法:
    with JoystickController("sim", max_vel=30.0) as ctrl:
        while True:
            q_cmd, actions = ctrl.step(dt)
            # q_cmd: np.ndarray(6) — 当前关节目标位置（度+gripper%）
            # actions: dict — 按键动作 {"save": True, ...}
"""
import numpy as np

from .base import Joystick
from .factory import make_joystick
from .mapping import JointMapping, state_to_joint_vel


class JoystickController:
    """手柄控制器：一步读状态+映射+积分，返回关节目标位置。"""

    def __init__(self, kind: str = "sim", max_vel: float = 30.0, mapping: JointMapping | None = None):
        """
        Args:
            kind: "sim"（虚拟手柄）或 "xbox"（真实 Xbox）。
            max_vel: 摇杆满杆对应的关节速度（度/秒）。
            mapping: 自定义映射，留空走默认 SO101 布局。
        """
        self._js: Joystick = make_joystick(kind)
        self._mapping = mapping or JointMapping(max_vel=max_vel)
        self._q_cmd = np.zeros(6, dtype=np.float32)

    def step(self, dt: float) -> tuple[np.ndarray, dict]:
        """每帧调用一次。返回 (q_cmd, actions)。

        Args:
            dt: 帧间隔（秒）。

        Returns:
            q_cmd: np.ndarray(6) — 当前关节目标位置（积分后）。
            actions: dict — 按键动作（边缘检测，按下瞬间 True）：
                {"toggle_recording": bool, "save": bool, "discard": bool, "reset": bool}
        """
        self._js.update()
        vel, actions = state_to_joint_vel(self._js.get_state(), self._mapping)
        self._q_cmd += vel * dt
        return self._q_cmd.copy(), actions

    @property
    def q_cmd(self) -> np.ndarray:
        """当前关节目标位置（只读副本）。"""
        return self._q_cmd.copy()

    @property
    def state(self):
        """当前手柄原始状态。"""
        return self._js.get_state()

    def reset(self, q_cmd: np.ndarray | None = None) -> None:
        """重置关节目标位置（默认归零，或传入指定位置）。"""
        self._q_cmd = q_cmd.copy().astype(np.float32) if q_cmd is not None else np.zeros(6, dtype=np.float32)

    def close(self) -> None:
        self._js.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
