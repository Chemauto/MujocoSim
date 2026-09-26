from __future__ import annotations

import mujoco
import numpy as np

from mujocosim.config import LidarDevice, MountCfg
from mujocosim.world import SimWorld

_MAX_BOUNCES = 8


def _rpy_deg_to_mat(rpy: list[float]) -> np.ndarray:
    r, p, y = np.radians(np.asarray(rpy, dtype=float))
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _quat_wxyz_to_mat(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _mount_rotation(mount: MountCfg) -> np.ndarray:
    if mount.rpy is not None:
        return _rpy_deg_to_mat(mount.rpy)
    if mount.quat is not None:
        return _quat_wxyz_to_mat(np.asarray(mount.quat, dtype=float))
    return np.eye(3)


def robot_body_ids(world: SimWorld) -> set[int]:
    base = world._base_body_id
    ids = {int(base)}
    for body_id in range(world.model.nbody):
        parent = int(world.model.body_parentid[body_id])
        while parent != 0:
            if parent == base:
                ids.add(body_id)
                break
            parent = int(world.model.body_parentid[parent])
    return ids


class RayCaster:
    def __init__(self, world: SimWorld, mount: MountCfg):
        self.world = world
        self.mount = mount
        self._robot_bodies = robot_body_ids(world)
        self._body_id = (
            mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_BODY, mount.attach)
            if mount.attach
            else 0
        )
        if self._body_id < 0:
            raise ValueError(f"挂载 body '{mount.attach}' 不存在")
        self._mount_pos = np.asarray(mount.pos, dtype=float)
        self._mount_rot = _mount_rotation(mount)

    def mount_frame(self) -> tuple[np.ndarray, np.ndarray]:
        model, data = self.world.model, self.world.data
        if self._body_id == 0:
            return self._mount_pos.copy(), self._mount_rot.copy()
        pos = (
            data.xpos[self._body_id]
            + data.xmat[self._body_id].reshape(3, 3) @ self._mount_pos
        )
        rot = data.xmat[self._body_id].reshape(3, 3) @ self._mount_rot
        return pos, rot

    def cast(
        self, dirs_sensor: np.ndarray, offsets_sensor: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        base_origin, rot = self.mount_frame()
        dirs_world = dirs_sensor @ rot.T
        if offsets_sensor is None:
            origins = np.tile(base_origin, (len(dirs_sensor), 1))
        else:
            origins = base_origin + offsets_sensor @ rot.T
        points = np.full((len(dirs_sensor), 3), np.nan)
        hits = np.zeros(len(dirs_sensor), dtype=bool)
        geomgroup = np.array([1, 1, 1, 1, 1, 1], dtype=np.uint8)
        with self.world.lock:
            for i, direction in enumerate(dirs_world):
                start = origins[i].copy()
                for _ in range(_MAX_BOUNCES):
                    geomid = np.zeros(1, dtype=np.int32)
                    dist = mujoco.mj_ray(
                        self.world.model,
                        self.world.data,
                        start,
                        direction,
                        geomgroup,
                        True,
                        -1,
                        geomid,
                    )
                    if dist < 0 or geomid[0] < 0:
                        break
                    body_id = int(self.world.model.geom_bodyid[geomid[0]])
                    if body_id in self._robot_bodies:
                        start = start + (dist + 1e-4) * direction
                        continue
                    points[i] = start + dist * direction
                    hits[i] = True
                    break
        return hits, points


def _angles(spec: tuple[float, float, float]) -> np.ndarray:
    start, stop, step = spec
    count = int(round((stop - start) / step)) + 1
    return np.radians(start + step * np.arange(count))


class SimLidar:
    def __init__(self, world: SimWorld, mount: MountCfg, device: LidarDevice):
        self.name = mount.name
        self.device = device
        self.frame_id = mount.attach or mount.name
        self.caster = RayCaster(world, mount)
        self._h = _angles(device.h_angles)
        self._v = (
            _angles(device.v_angles) if device.v_angles is not None else np.zeros(1)
        )
        self._next_due = 0.0
        self._init_directions()

    def _init_directions(self) -> None:
        if self.device.lidar_type == "2d":
            dirs = np.stack(
                [np.cos(self._h), np.sin(self._h), np.zeros_like(self._h)], axis=1
            )
        else:
            vv, hh = np.meshgrid(self._v, self._h, indexing="ij")
            dirs = np.stack(
                [np.cos(vv) * np.cos(hh), np.cos(vv) * np.sin(hh), np.sin(vv)],
                axis=-1,
            ).reshape(-1, 3)
        self.directions = dirs

    @property
    def fps(self) -> float:
        return self.device.fps

    def scan(self) -> dict:
        hits, points_world = self.caster.cast(self.directions)
        origin, rot = self.caster.mount_frame()
        points_body = (points_world - origin) @ rot
        if self.device.lidar_type == "2d":
            dists = np.full(len(self.directions), np.inf)
            ok = hits & np.isfinite(points_world).all(axis=1)
            dists[ok] = np.linalg.norm(points_world[ok] - origin, axis=1)
            return {"ranges": dists, "hits": hits}
        pts = points_body[hits & np.isfinite(points_body).all(axis=1)]
        return {"points": pts.astype(np.float32), "hits": hits}
