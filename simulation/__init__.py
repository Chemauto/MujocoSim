"""simulation —— 仿真场景层(见 docs/simulation.md)。

只负责场景搭建与运行:MJCF 组装(builder)、程序化地形(terrain)、
仿真世界与机器人访问接口(world/robot_api)。控制器在 rl_control/,
配置加载在 common/config.py。
"""
from .builder import build
from .world import SimWorld
from .robot_api import RobotAPI

__all__ = ["build", "SimWorld", "RobotAPI"]
