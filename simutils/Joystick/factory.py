"""手柄工厂：根据 kind 构建虚拟或真实手柄。"""
from .base import Joystick


def make_joystick(kind: str, **kwargs) -> Joystick:
    """构建手柄实例。

    Args:
        kind: "sim"（虚拟 Xbox 手柄窗口）或 "xbox"（真实 Xbox 手柄）。
        **kwargs: 传给具体实现的额外参数。
    """
    if kind == "sim":
        from .simulated import SimulatedJoystick
        return SimulatedJoystick(**kwargs)
    elif kind == "xbox":
        from .xbox import XboxJoystick
        return XboxJoystick(**kwargs)
    else:
        raise ValueError(f"未知手柄类型: {kind}（可选 sim / xbox）")
