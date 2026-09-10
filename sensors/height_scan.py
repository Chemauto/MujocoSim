"""height_scan.py —— 高度扫描传感器(187 点相对高度,RL 策略的外感知)。

逐行对照 llm-legged-lab unitree_sdk2py_bridge._compute_height_map 实现:
  - 17×11 网格(GridPattern 1.6×1.0m @ 0.1m,y 在外层);
  - 网格随 base 偏航角旋转、平移到 base 位置,射线起点抬升 ray_offset_z(20m);
  - 射线竖直向下,命中高度换算 heights = base_z - hit_z - height_offset;
  - 只与 geomgroup (1,1,0,0,0,0)(地面/地形/物体)相交,不打机器人自身;
  - 未命中的点记 0。

属于"传感器"数据,归 SensorData/数据总线 sensor/height_scan 话题。
"""
from __future__ import annotations

import mujoco
import numpy as np


class HeightScanner:
    """绑定 (model, data) 的高度扫描仪;compute() 返回 float32[n_points]。"""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 base_body: str = "base_link",
                 size: tuple[float, float] = (1.6, 1.0),
                 resolution: float = 0.1,
                 ray_offset_z: float = 20.0,
                 height_offset: float = 0.5,
                 geomgroup=(1, 1, 0, 0, 0, 0),
                 robot_body_ids=None,
                 max_bounces: int = 32):
        self.ray_offset_z = ray_offset_z
        self.height_offset = height_offset
        self.geomgroup = np.array(geomgroup, dtype=np.uint8)
        self.max_bounces = int(max_bounces)
        bid = model.body(base_body).id
        self.base_adr = int(model.jnt_qposadr[model.body_jntadr[bid]])
        # 命中这些 body(机器人自身)时继续向下续射,排除自遮挡
        self.robot_body_ids = set(robot_body_ids) if robot_body_ids else set()

        x = np.arange(-size[0] / 2, size[0] / 2 + 1e-9, resolution)
        y = np.arange(-size[1] / 2, size[1] / 2 + 1e-9, resolution)
        gx, gy = np.meshgrid(x, y, indexing="xy")       # (11, 17),y 在外层
        self.grid = np.zeros((gx.size, 3))
        self.grid[:, 0] = gx.ravel()
        self.grid[:, 1] = gy.ravel()
        self.n_points = gx.size
        self._model = model
        self._data = data

    def compute(self) -> np.ndarray:
        """当前仿真状态下的高度扫描(float32[n_points])。"""
        model, data = self._model, self._data
        base_pos = data.qpos[self.base_adr:self.base_adr + 3]
        w, qx, qy, qz = data.qpos[self.base_adr + 3:self.base_adr + 7]
        yaw = np.arctan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

        starts = base_pos + self.grid @ rot.T
        starts[:, 2] += self.ray_offset_z
        dirs = np.zeros_like(starts)
        dirs[:, 2] = -1.0

        dist = np.full(self.n_points, -1.0)
        ray_start_z = starts[:, 2].copy()               # 每点最终射线起点 z(续射会更新)
        geomid = np.zeros(1, dtype=np.int32)
        for i in range(self.n_points):
            start = starts[i].copy()
            valid = False
            for _ in range(self.max_bounces):                          # 命中机器人 body 就越过它续射
                d = mujoco.mj_ray(model, data, start, dirs[i],
                                  self.geomgroup, 1, -1, geomid)
                if d < 0 or geomid[0] < 0:
                    break                               # 未命中
                hit_body = int(model.geom_bodyid[geomid[0]])
                if hit_body not in self.robot_body_ids:
                    valid = True
                    ray_start_z[i] = start[2]           # 记录有效命中时的起点 z
                    break
                start = start + dirs[i] * (d + 1e-4)    # 越过机器人,继续向下
                ray_start_z[i] = start[2]
            dist[i] = d if valid else -1.0

        heights = np.zeros(self.n_points, dtype=np.float32)
        hit = dist >= 0.0
        if hit.any():
            hit_z = ray_start_z[hit] - dist[hit]        # 射线向下:命中 z = 起点 - 距离
            heights[hit] = (base_pos[2] - hit_z - self.height_offset).astype(np.float32)
        return heights
