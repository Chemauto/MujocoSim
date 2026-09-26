from __future__ import annotations

import math

import mujoco
import numpy as np

from mujocosim.config import CameraDevice, MountCfg
from mujocosim.world import SimWorld


class SimCamera:
    def __init__(self, world: SimWorld, mount: MountCfg, device: CameraDevice):
        self.name = mount.name
        self.device = device
        self.world = world
        self.cam_id = mujoco.mj_name2id(
            world.model, mujoco.mjtObj.mjOBJ_CAMERA, mount.name
        )
        if self.cam_id < 0:
            raise ValueError(f"模型里没有相机 '{mount.name}'（mounts 未生效？）")
        self.width = device.width
        self.height = device.height
        self.frame_id = mount.name
        self._renderer = mujoco.Renderer(
            world.model, height=device.height, width=device.width
        )
        self._next_due = 0.0

    @property
    def fps(self) -> float:
        return self.device.fps

    def intrinsics(self) -> tuple[float, float, float, float]:
        fovy = float(self.world.model.cam_fovy[self.cam_id])
        fy = (self.height / 2.0) / math.tan(math.radians(fovy) / 2.0)
        return fy, fy, self.width / 2.0, self.height / 2.0

    def render(self) -> dict:
        out: dict = {}
        with self.world.lock:
            if self.device.rgb:
                self._renderer.update_scene(self.world.data, camera=self.cam_id)
                out["rgb"] = self._renderer.render().copy()
            if self.device.depth:
                self._renderer.enable_depth_rendering()
                self._renderer.update_scene(self.world.data, camera=self.cam_id)
                depth = self._renderer.render().astype(np.float32).copy()
                self._renderer.disable_depth_rendering()
                depth[~np.isfinite(depth)] = np.nan
                depth[depth > 1e9] = np.nan
                far = float(self.world.model.vis.map.zfar)
                depth[depth >= far] = np.nan
                out["depth"] = depth
        return out
