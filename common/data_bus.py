"""data_bus.py —— ROS2 风格的数据总线与导出器(零依赖,纯 Python 实现)。

  DataBus          话题发布/订阅(形似 ROS2 topic:publish/subscribe)
  JsonlRecorder    订阅若干话题,按行写 JSONL 文件({t, topic, data})
  DataExporter     组合根使用:按配置频率采集 robot/env/sensor 三类快照并发布

话题约定:
  robot/state       RobotState.to_dict()   关节角/速/力矩 + base 位姿
  env/state         EnvironmentState       仿真时间 + 物体位姿
  sensor/height_scan SensorData            高度扫描(相机帧体积大,不进总线文件)

导出文件在 configs/*.yaml 的 data: 段配置(enabled/format/dir/rate/topics)。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .config import DataCfg
from .data_model import EnvironmentState, RobotState, SensorData

ROBOT_TOPIC = "robot/state"
ENV_TOPIC = "env/state"
SCAN_TOPIC = "sensor/height_scan"


class DataBus:
    """最小话题总线:publish(topic, data) -> 通知所有订阅者。"""

    def __init__(self):
        self._subs: dict[str, list] = {}

    def subscribe(self, topic: str, callback):
        """callback(topic: str, stamp: float, data: dict)。"""
        self._subs.setdefault(topic, []).append(callback)

    def publish(self, topic: str, stamp: float, data: dict):
        for cb in self._subs.get(topic, []):
            cb(topic, stamp, data)


class JsonlRecorder:
    """把订阅的话题按行追加写入 JSONL,适合 pandas/plotly 后处理。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")
        self._n = 0

    def __call__(self, topic: str, stamp: float, data: dict):
        self._fh.write(json.dumps({"t": round(float(stamp), 4), "topic": topic,
                                   "data": data}, ensure_ascii=False) + "\n")
        self._n += 1
        if self._n % 200 == 0:
            self._fh.flush()

    def close(self):
        self._fh.close()


class DataExporter:
    """组合根在仿真循环里每步调用 update();内部按 data.rate 节流采集发布。

    world 需要提供: sim_time, object_poses(), robot(RobotAPI);
    controller/height_scanner 可选(有才发对应话题)。
    """

    def __init__(self, cfg: DataCfg, world, controller=None, height_scanner=None):
        self.cfg = cfg
        self.world = world
        self.controller = controller
        self.height_scanner = height_scanner
        self.bus = DataBus()
        self.recorder = None
        if cfg.enabled:
            if cfg.format != "jsonl":
                print(f"[data] 警告: 暂只支持 jsonl 导出,忽略 format={cfg.format!r}")
            known = {ROBOT_TOPIC, ENV_TOPIC, SCAN_TOPIC}
            unknown = [t for t in cfg.topics if t not in known]
            if unknown:
                print(f"[data] 警告: 未知话题(将忽略): {unknown},可用: {sorted(known)}")
            out = Path(cfg.dir)
            if not out.is_absolute():
                out = Path(__file__).resolve().parent.parent / out
            out = out / f"run_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
            try:
                self.recorder = JsonlRecorder(out)
            except OSError as e:
                print(f"[data] 警告: 导出文件不可写({e}),本次运行不导出")
                self.recorder = None
            if self.recorder is not None:
                for topic in cfg.topics:
                    if topic in known:
                        self.bus.subscribe(topic, self.recorder)
                print(f"[data] 导出开启: {out}  话题={cfg.topics}  频率={cfg.rate}Hz")
        self._last_t = -1e9

    def update(self):
        """由组合根的仿真循环每个物理步调用;内部按 rate 节流。"""
        t = self.world.sim_time
        if t - self._last_t < 1.0 / max(self.cfg.rate, 1e-6):
            return
        self._last_t = t

        if ROBOT_TOPIC in self.cfg.topics:
            st: RobotState = self.world.robot.robot_state(
                joint_names=self.controller.cfg.joint_names if self.controller else None,
                tau=self.controller.last_torques if self.controller else None)
            self.bus.publish(ROBOT_TOPIC, t, st.to_dict())

        if ENV_TOPIC in self.cfg.topics:
            env = EnvironmentState(sim_time=t, objects=self.world.object_poses())
            self.bus.publish(ENV_TOPIC, t, env.to_dict())

        if SCAN_TOPIC in self.cfg.topics and self.height_scanner is not None:
            scan = self.height_scanner.compute()
            sd = SensorData(height_scan=np.asarray(scan, dtype=float).tolist())
            self.bus.publish(SCAN_TOPIC, t, sd.to_dict())

    def close(self):
        if self.recorder is not None:
            self.recorder.close()
