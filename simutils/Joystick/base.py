"""手柄基础类型：JoystickState(统一状态) + Joystick(抽象基类)。

虚拟手柄(SimulatedJoystick)和真实 Xbox(XboxJoystick)实现同一接口，
collect_data.py 只用 get_state() 拿到 axes/buttons，无需区分来源。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class JoystickState:
    """手柄状态（Xbox 布局）。

    axes: 各轴 ∈ [-1, 1]；buttons: 按下为 True。
    """
    axes: dict[str, float] = field(default_factory=lambda: {
        "left_x": 0.0, "left_y": 0.0,
        "right_x": 0.0, "right_y": 0.0,
        "l2": 0.0, "r2": 0.0,
    })
    buttons: dict[str, bool] = field(default_factory=lambda: {
        "a": False, "b": False, "x": False, "y": False,
        "lb": False, "rb": False,
        "back": False, "start": False, "guide": False,
        "lstick": False, "rstick": False,
        "dpad_up": False, "dpad_down": False,
        "dpad_left": False, "dpad_right": False,
    })

    def copy(self):
        return JoystickState(
            axes=dict(self.axes),
            buttons=dict(self.buttons),
        )


class Joystick(ABC):
    """手柄抽象基类。"""

    @abstractmethod
    def update(self) -> None:
        """读取最新事件（每帧调用一次）。"""

    @abstractmethod
    def get_state(self) -> JoystickState:
        """返回当前手柄状态。"""

    def close(self) -> None:
        """释放资源。"""
