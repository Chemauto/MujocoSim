from __future__ import annotations

import threading

from mujocosim.config import CameraDevice, HeightmapDevice, LidarDevice, SimConfig
from mujocosim.world import SimWorld

from .camera import SimCamera
from .heightmap import HeightScanner
from .lidar import SimLidar

__all__ = ["SimCamera", "SimLidar", "HeightScanner", "SensorSuite"]


class SensorSuite:
    def __init__(self, cfg: SimConfig, world: SimWorld):
        self.cameras: list[SimCamera] = []
        self.lidars: list[SimLidar] = []
        self.heightmaps: list[HeightScanner] = []
        for mount in cfg.mounts:
            device = cfg.devices[mount.device]
            if isinstance(device, CameraDevice):
                self.cameras.append(SimCamera(world, mount, device))
            elif isinstance(device, LidarDevice):
                self.lidars.append(SimLidar(world, mount, device))
            elif isinstance(device, HeightmapDevice):
                self.heightmaps.append(HeightScanner(world, mount, device))
        self._lock = threading.Lock()
        self._pending: dict[str, dict] = {}
        self._latest: dict[str, dict] = {}
        self._due: dict[str, float] = {}
        for sensor in [*self.cameras, *self.lidars, *self.heightmaps]:
            self._due[sensor.name] = 0.0

    def step(self, sim_time: float) -> None:
        for sensor in [*self.cameras, *self.lidars, *self.heightmaps]:
            if sim_time < self._due[sensor.name]:
                continue
            self._due[sensor.name] = sim_time + 1.0 / sensor.fps
            payload = (
                sensor.render() if isinstance(sensor, SimCamera) else sensor.scan()
            )
            payload["kind"] = (
                "camera"
                if isinstance(sensor, SimCamera)
                else "lidar"
                if isinstance(sensor, SimLidar)
                else "heightmap"
            )
            payload["name"] = sensor.name
            payload["frame_id"] = sensor.frame_id
            if isinstance(sensor, SimCamera):
                payload["intrinsics"] = sensor.intrinsics()
            with self._lock:
                self._pending[sensor.name] = payload
                self._latest[sensor.name] = payload

    def drain(self) -> dict[str, dict]:
        with self._lock:
            pending = dict(self._pending)
            self._pending.clear()
        return pending

    def latest(self) -> dict[str, dict]:
        with self._lock:
            return dict(self._latest)
