"""手柄状态 → 关节速度映射（默认 Xbox 布局，可通过 config 覆盖）。

默认映射（SO101 6 关节）:
  left_y  → shoulder_lift (1)    left_x  → shoulder_pan (0)
  right_y → elbow_flex   (2)    right_x → wrist_roll   (4)
  dpad 上/下 → wrist_flex (3)
  r2(开) / l2(合) → gripper (5)

摇杆满杆 = max_vel 度/秒（默认 30）；扳机 0~1 映射到 gripper 方向。
"""
from dataclasses import dataclass, field

import numpy as np

from .base import JoystickState


@dataclass
class JointMapping:
    """手柄轴/按键 → 关节速度的映射配置。"""
    # axis_name → (joint_index, sign)；sign 翻转方向
    axis_map: dict[str, tuple[int, float]] = field(default_factory=lambda: {
        "left_y":  (1, +1.0),   # 左摇杆 Y → shoulder_lift
        "left_x":  (0, +1.0),   # 左摇杆 X → shoulder_pan
        "right_y": (2, +1.0),   # 右摇杆 Y → elbow_flex
        "right_x": (4, +1.0),   # 右摇杆 X → wrist_roll
    })
    dpad_joint: int = 3          # 十字键上下控制的关节索引
    dpad_speed: float = 1.0      # 十字键按住时的速度系数（× max_vel）
    max_vel: float = 30.0        # 摇杆满杆对应的关节速度（度/秒）


def state_to_joint_vel(state: JoystickState, mapping: JointMapping | None = None) -> tuple[np.ndarray, dict]:
    """手柄状态 → (关节速度[6], 附加动作 dict)。

    附加动作: {"toggle_recording": bool, "save": bool, "discard": bool, "reset": bool}
    按键只在按下瞬间返回 True（需外部去重）。
    """
    m = mapping or JointMapping()
    vel = np.zeros(6, dtype=np.float32)

    # 摇杆轴 → 关节速度
    for axis_name, (j, sign) in m.axis_map.items():
        val = state.axes.get(axis_name, 0.0)
        vel[j] += sign * val * m.max_vel

    # Dpad 上下 → dpad_joint
    if state.buttons.get("dpad_up", False):
        vel[m.dpad_joint] += m.dpad_speed * m.max_vel
    if state.buttons.get("dpad_down", False):
        vel[m.dpad_joint] -= m.dpad_speed * m.max_vel

    # 扳机 R2(开夹爪) / L2(合夹爪) → gripper
    r2 = state.axes.get("r2", 0.0)
    l2 = state.axes.get("l2", 0.0)
    vel[5] = (r2 - l2) * m.max_vel

    # 按键动作
    actions = {
        "toggle_recording": state.buttons.get("start", False),
        "save":             state.buttons.get("a", False),
        "discard":          state.buttons.get("b", False),
        "reset":            state.buttons.get("back", False),
    }
    return vel, actions
