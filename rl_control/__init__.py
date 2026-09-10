"""rl_control —— RL 控制层(见 docs/rl_control.md)。

五种控制模式(阻尼/力控/PD站立/位控/RL行走)与策略推理。
只依赖 common(配置/数据模型),通过组合根注入的 RobotAPI 访问机器人,
不 import simulation 与 mujoco——控制逻辑可直接搬到实机部署侧。
"""
from .controller import RobotController
from .policy import PolicyRunner

__all__ = ["RobotController", "PolicyRunner"]
