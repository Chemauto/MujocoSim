"""simutils.Joystick：统一手柄接口（虚拟 Xbox + 真实 Xbox）。

优雅用法（推荐）:
    from simutils.Joystick import JoystickController

    with JoystickController("sim", max_vel=30.0) as ctrl:
        while True:
            q_cmd, actions = ctrl.step(dt)

底层用法:
    from simutils.Joystick import make_joystick, state_to_joint_vel
    js = make_joystick("sim")
    js.update()
    vel, actions = state_to_joint_vel(js.get_state())
"""
from .base import Joystick, JoystickState
from .controller import JoystickController
from .factory import make_joystick
from .mapping import JointMapping, state_to_joint_vel

__all__ = [
    "Joystick",
    "JoystickState",
    "JoystickController",
    "JointMapping",
    "make_joystick",
    "state_to_joint_vel",
]
