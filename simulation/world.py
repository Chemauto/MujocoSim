"""world.py —— SimWorld:纯仿真世界。

职责边界(见 docs/simulation.md):
  只负责场景搭建结果的运行:物理步进、相机渲染接口、机器人访问接口(RobotAPI)、
  物体位姿查询、复位。不包含任何控制器/感知逻辑——RL 控制在 rl_control/,
  感知在 sensors/ 与 yolo/,由组合根(main/dashboard)组装。

  可选挂一个 common.data_bus.DataExporter:挂上后 step() 会按配置频率
  自动发布/导出 robot/env/sensor 快照(数据分类见 docs/common.md)。
"""
from __future__ import annotations

import mujoco

from .builder import build
from .robot_api import RobotAPI
from common.config import SimConfig


class SimWorld:
    """一个配置驱动的 MuJoCo 仿真世界(不含控制器与传感器实例)。

    相机等传感器实例由组合根经 camera_factory 注入(依赖方向:
    simulation 不 import sensors,见 docs/architecture.md)。
    """

    def __init__(self, cfg: SimConfig, camera_factory=None):
        """camera_factory(model, data, name, width, height) -> 相机实例,
        由组合根注入(如 sensors.SimCamera);None 则不创建相机。"""
        self.cfg = cfg
        self.model, self.robot_meta = build(cfg)
        self.data = mujoco.MjData(self.model)
        make_cam = camera_factory or (lambda *a: None)
        self.cameras: dict = {}
        for cc in cfg.cameras:
            cam = make_cam(self.model, self.data, cc.name, cc.width, cc.height)
            if cam is not None:
                self.cameras[cc.name] = cam
        self.robot = RobotAPI(self.model, self.data,
                              base_body=self.robot_meta["base_body"])
        self.spawn_pos = list(cfg.spawn if cfg.spawn
                              else self.robot_meta.get("default_spawn", [0, 0, 1.0]))
        self.exporter = None                 # common.data_bus.DataExporter,组合根挂载
        self.pre_step = None                 # 组合根可挂控制器:每步 mj_step 前调用
        self._viewer = None

    # ---- 步进 ----
    def step(self, n: int = 1) -> None:
        for _ in range(n):
            if self.pre_step is not None:
                self.pre_step()              # 控制器写 ctrl,随本次 mj_step 生效
            if self.exporter is not None:
                self.exporter.update()       # 数据导出(按配置频率节流)
            mujoco.mj_step(self.model, self.data)
        if self._viewer is not None:
            self._viewer.sync()

    def reset(self) -> None:
        """仿真整体清零(回 t=0)。机器人站姿复位用 world.robot.reset(...)。"""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    # ---- 查询 ----
    @property
    def sim_time(self) -> float:
        return self.data.time

    def robot_base_pose(self) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        pos, quat = self.robot.base_pose()
        return (tuple(float(v) for v in pos), tuple(float(v) for v in quat))

    def object_poses(self) -> dict:
        """场景物体位姿 {name: {pos:[3], quat:[4]}}(环境数据,进 env/state 话题)。"""
        out = {}
        for obj in self.cfg.objects:
            bid = self.model.body(obj.name).id
            out[obj.name] = {"pos": self.data.xpos[bid].tolist(),
                             "quat": self.data.xquat[bid].tolist()}
        return out

    # ---- viewer ----
    def open_viewer(self):
        """启动交互 viewer(passive 模式,本线程可以继续渲染相机/检测)。"""
        if self._viewer is None:
            import mujoco.viewer                 # 子模块,须显式导入
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        return self._viewer

    def viewer_running(self) -> bool:
        return self._viewer is not None and self._viewer.is_running()

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self.exporter is not None:
            self.exporter.close()
        for cam in self.cameras.values():
            cam.close()
        self.cameras.clear()

    # ---- 主循环 ----
    def run(self, on_update=None, realtime: bool | None = None) -> None:
        """步进直到 viewer 关闭。

        on_update(world) 在每个仿真步后被调用(做渲染/检测/打日志的地方)。
        realtime=True 时按墙钟时间节流,跑成实时。
        """
        realtime = self.cfg.realtime if realtime is None else realtime
        import time
        step_start = time.perf_counter()
        while self._viewer is None or self._viewer.is_running():
            if self.pre_step is not None:
                self.pre_step()
            if self.exporter is not None:
                self.exporter.update()
            mujoco.mj_step(self.model, self.data)
            if self._viewer is not None:
                self._viewer.sync()
            if on_update is not None:
                on_update(self)
            if realtime:
                step_start += self.model.opt.timestep
                delay = step_start - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    step_start = time.perf_counter()   # 落后太多就重新对齐

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
