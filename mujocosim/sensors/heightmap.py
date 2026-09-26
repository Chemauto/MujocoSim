from __future__ import annotations

import numpy as np

from mujocosim.config import HeightmapDevice, MountCfg
from mujocosim.world import SimWorld

from .lidar import RayCaster


class HeightScanner:
    def __init__(self, world: SimWorld, mount: MountCfg, device: HeightmapDevice):
        self.name = mount.name
        self.device = device
        self.frame_id = mount.attach or mount.name
        self.caster = RayCaster(world, mount)
        length, width = device.size
        res = device.resolution
        xs = np.arange(-length / 2.0 + res / 2.0, length / 2.0, res)
        ys = np.arange(-width / 2.0 + res / 2.0, width / 2.0, res)
        self.nx, self.ny = len(xs), len(ys)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        self.grid_xy = np.stack([gx, gy], axis=-1).reshape(-1, 2)
        self._dirs = np.tile(np.array([0.0, 0.0, -1.0]), (len(self.grid_xy), 1))
        self._next_due = 0.0

    @property
    def fps(self) -> float:
        return self.device.fps

    @property
    def point_count(self) -> int:
        return len(self.grid_xy)

    def scan(self) -> dict:
        origin, rot = self.caster.mount_frame()
        offsets = np.zeros((len(self.grid_xy), 3))
        offsets[:, :2] = self.grid_xy
        offsets[:, 2] = 2.0
        hits, points_world = self.caster.cast(self._dirs, offsets_sensor=offsets)
        points_body = (points_world - origin) @ rot
        ok = hits & np.isfinite(points_body).all(axis=1)
        heights = np.full(len(self.grid_xy), np.nan, dtype=np.float32)
        heights[ok] = points_body[ok, 2]
        return {
            "points": points_body[ok].astype(np.float32),
            "heights": heights,
            "grid": (self.nx, self.ny),
        }
