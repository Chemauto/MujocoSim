"""simulation —— 场景组装 + 仿真世界。

一个机器人模型 + 若干物体 + 若干相机,全部由 YAML 配置驱动,
组装用的是 MuJoCo 3.x 的 MjSpec(运行时拼装,不需要改任何 XML)。
"""
from .config import SimConfig, load_config
from .world import SimWorld

__all__ = ["SimConfig", "load_config", "SimWorld"]
