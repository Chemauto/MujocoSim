"""data_model.py —— 数据分类模型:机器人的、环境的、传感器的,各归各位。

三类快照(dataclass + to_dict),是 DataExporter 发布到数据总线的标准载荷:

  RobotState        本体感知:关节(角/速/力矩)与 base 位姿/角速度/重力投影
  EnvironmentState  环境:仿真时间、场景物体位姿
  SensorData        外感知:高度扫描等(相机帧体积大,走 sensors 层单独通道)

任何新数据先归类到这里,再经 common/data_bus.py 发布/导出。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _round(v, nd=4):
    """递归把 float 保留 nd 位(导出文件别太大),list/ndarray -> list。"""
    if isinstance(v, (np.floating, float)):
        return round(float(v), nd)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.ndarray, list, tuple)):
        return [_round(x, nd) for x in v]
    if isinstance(v, dict):
        return {k: _round(x, nd) for k, x in v.items()}
    return v


@dataclass
class RobotState:
    """机器人本体状态(本体感知)。joint_* 均为"策略关节序"。"""
    joint_names: list[str]
    q: list[float]                          # 关节角 rad
    dq: list[float]                         # 关节速 rad/s
    tau: list[float]                        # 关节力矩 N·m(上一控制周期输出)
    base_pos: list[float]                   # 世界系
    base_quat: list[float]                  # wxyz 世界系
    base_ang_vel_b: list[float]             # 体系角速度 rad/s
    projected_gravity_b: list[float]        # 重力方向在体系的投影(单位向量)

    def to_dict(self) -> dict:
        return _round({
            "joint_names": self.joint_names,
            "q": self.q, "dq": self.dq, "tau": self.tau,
            "base_pos": self.base_pos, "base_quat": self.base_quat,
            "base_ang_vel_b": self.base_ang_vel_b,
            "projected_gravity_b": self.projected_gravity_b,
        })


@dataclass
class EnvironmentState:
    """环境状态:仿真时间 + 场景物体位姿(世界系)。"""
    sim_time: float
    objects: dict = field(default_factory=dict)   # {name: {pos:[3], quat:[4]}}

    def to_dict(self) -> dict:
        return _round({"sim_time": self.sim_time, "objects": self.objects})


@dataclass
class SensorData:
    """传感器数据(外感知)。camera 帧体积大,不进 JSONL,单独通道处理。"""
    height_scan: list[float] | None = None  # 187 点相对高度

    def to_dict(self) -> dict:
        return _round({"height_scan": self.height_scan})
