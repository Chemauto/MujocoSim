"""world.py —— SimWorld:封装仿真世界。

职责:持有 MjModel/MjData、管理相机和 viewer、按真实时间步进。
主程序只需要:构建 -> 循环 step -> 随时从 world.cameras 抓图。
"""
from __future__ import annotations

import time

import mujoco
import numpy as np

from .builder import build
from .config import SimConfig
from .controller import RobotController
from sensors.camera import SimCamera


class SimWorld:
    """一个配置驱动的 MuJoCo 仿真世界。"""

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.model, self.robot_meta = build(cfg)
        self.data = mujoco.MjData(self.model)
        self.cameras: dict[str, SimCamera] = {}
        for cc in cfg.cameras:
            self.cameras[cc.name] = SimCamera(self.model, self.data, cc.name,
                                              width=cc.width, height=cc.height)
        self._viewer = None
        self._sim_time = 0.0
        self.controller: RobotController | None = None
        if cfg.controller is not None:
            self.controller = RobotController(self.model, self.data, cfg.controller,
                                              base_body=self.robot_meta["base_body"])
            self.controller.mode = cfg.controller.initial_mode
        self.spawn_pos = list(cfg.spawn if cfg.spawn
                              else self.robot_meta.get("default_spawn", [0, 0, 1.0]))

    # ---- 基本查询 ----
    @property
    def sim_time(self) -> float:
        return self.data.time

    def robot_base_pose(self) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        """机器人根 body 的世界位姿 (pos, quat[w,x,y,z])。"""
        bid = self.model.body(self.robot_meta["base_body"]).id
        return (tuple(float(v) for v in self.data.xpos[bid]),
                tuple(float(v) for v in self.data.xquat[bid]))

    # ---- 步进 ----
    def step(self, n: int = 1) -> None:
        for _ in range(n):
            if self.controller is not None:
                self.controller.compute()          # 力矩写在 ctrl,随本次 mj_step 生效
            mujoco.mj_step(self.model, self.data)
        if self._viewer is not None:
            self._viewer.sync()

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def reset_robot(self) -> None:
        """机器人回到默认站姿出生点(速度清零、控制器内部状态清零)。"""
        d, m = self.data, self.model
        mujoco.mj_resetData(m, d)
        if self.controller is not None:
            c = self.controller
            d.qpos[c.base_adr:c.base_adr + 3] = self.spawn_pos
            d.qpos[c.base_adr + 3:c.base_adr + 7] = [1.0, 0.0, 0.0, 0.0]
            d.qpos[c.qadr] = c.q_default            # 默认站姿(策略关节序散射到 qpos)
            d.qvel[:] = 0.0
            c.reset()
        mujoco.mj_forward(m, d)
        if self._viewer is not None:
            self._viewer.sync()

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
        step_start = time.perf_counter()
        while self._viewer is None or self._viewer.is_running():
            if self.controller is not None:
                self.controller.compute()
            mujoco.mj_step(self.model, self.data)
            self._sim_time = self.data.time
            if self._viewer is not None:
                self._viewer.sync()
            if on_update is not None:
                on_update(self)
            if realtime:
                # 按当前时间片补偿耗时,保持大约 1x 实时
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
