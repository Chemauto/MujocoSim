from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

from .config import SimConfig

REPO_ROOT = Path(__file__).resolve().parent.parent


def _rpy_to_mat(rpy: list[float]) -> np.ndarray:
    r, p, y = rpy
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


def _lookat_quat(pos: list[float], lookat: list[float]) -> list[float]:
    pos_a = np.asarray(pos, dtype=float)
    forward = np.asarray(lookat, dtype=float) - pos_a
    norm = np.linalg.norm(forward)
    if norm < 1e-9:
        raise ValueError(f"lookat 与 pos 重合: {pos}")
    z = -forward / norm
    up = np.array([0.0, 0.0, 1.0])
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    mat = np.array([x[0], y[0], z[0], x[1], y[1], z[1], x[2], y[2], z[2]], dtype=float)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat)
    return [float(q) for q in quat]


class SimWorld:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.model = self._load_model(cfg)
        self.data = mujoco.MjData(self.model)
        self.lock = threading.Lock()
        self._base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, cfg.robot_meta.base_body
        )
        if self._base_body_id < 0:
            raise ValueError(f"场景里找不到机器人根 body '{cfg.robot_meta.base_body}'")
        self._joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in cfg.robot_meta.joint_names
        ]
        missing = [
            name
            for name, jid in zip(cfg.robot_meta.joint_names, self._joint_ids)
            if jid < 0
        ]
        if missing:
            raise ValueError(f"场景里找不到关节: {missing}")
        self._joint_qadr = [int(self.model.jnt_qposadr[j]) for j in self._joint_ids]
        self._joint_dofadr = [int(self.model.jnt_dofadr[j]) for j in self._joint_ids]
        self._free_jid = (
            self._find_free_joint() if cfg.robot_meta.floating_base else None
        )
        self.reset()

    @staticmethod
    def _load_model(cfg: SimConfig) -> mujoco.MjModel:
        scene_file = Path(cfg.scene)
        scene_path = scene_file if scene_file.is_absolute() else REPO_ROOT / scene_file
        robot_path = REPO_ROOT / "assets" / "robots" / cfg.robot / cfg.robot_meta.xml
        scene = mujoco.MjSpec.from_file(str(scene_path))
        robot = mujoco.MjSpec.from_file(str(robot_path))
        wb = robot.worldbody
        for geom in list(wb.geoms):
            robot.delete(geom)
        for light in list(wb.lights):
            robot.delete(light)
        scene.option.timestep = cfg.timing.sim_dt
        scene.option.cone = robot.option.cone
        scene.option.impratio = robot.option.impratio
        mount = scene.worldbody.add_site(name="robot_mount", pos=[0, 0, 0])
        scene.attach(robot, site=mount, prefix="")
        SimWorld._mount_cameras(scene, cfg)
        return scene.compile()

    @staticmethod
    def _mount_cameras(scene: mujoco.MjSpec, cfg: SimConfig) -> None:
        from .config import CameraDevice

        for cam_mount in cfg.mounts:
            device = cfg.devices[cam_mount.device]
            if not isinstance(device, CameraDevice):
                continue
            if scene.camera(cam_mount.name) is not None:
                raise ValueError(f"相机名 '{cam_mount.name}' 与场景/机器人自带相机重名")
            parent = (
                scene.body(cam_mount.attach) if cam_mount.attach else scene.worldbody
            )
            kwargs: dict = dict(
                name=cam_mount.name, pos=cam_mount.pos, fovy=device.fovy
            )
            forward = np.array(
                [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float
            )
            if cam_mount.rpy is not None:
                rpy = [math.radians(a) for a in cam_mount.rpy]
                rot = _rpy_to_mat(rpy) @ forward
                quat = np.zeros(4)
                mujoco.mju_mat2Quat(quat, rot.reshape(-1))
                kwargs["quat"] = [float(q) for q in quat]
            elif cam_mount.quat is not None:
                kwargs["quat"] = cam_mount.quat
            elif cam_mount.lookat is not None:
                kwargs["quat"] = _lookat_quat(cam_mount.pos, cam_mount.lookat)
            else:
                quat = np.zeros(4)
                mujoco.mju_mat2Quat(quat, forward.reshape(-1))
                kwargs["quat"] = [float(q) for q in quat]
            parent.add_camera(**kwargs)

    def _find_free_joint(self) -> int:
        for jid in range(self.model.njnt):
            if (
                self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE
                and self.model.jnt_bodyid[jid] == self._base_body_id
            ):
                return jid
        raise ValueError(
            f"floating_base=true 但 '{self.cfg.robot_meta.base_body}' 上没有 freejoint"
        )

    def reset(self) -> None:
        with self.lock:
            mujoco.mj_resetData(self.model, self.data)
            for i, adr in enumerate(self._joint_qadr):
                self.data.qpos[adr] = self.cfg.robot_meta.default_joint_pos[i]
            if self._free_jid is not None:
                adr = int(self.model.jnt_qposadr[self._free_jid])
                self.data.qpos[adr : adr + 3] = self.cfg.spawn
                self.data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]
            self.data.qvel[:] = 0.0
            self.data.ctrl[:] = 0.0
            mujoco.mj_forward(self.model, self.data)

    def step(self, n: int = 1) -> None:
        with self.lock:
            for _ in range(n):
                mujoco.mj_step(self.model, self.data)

    def run(
        self,
        gui: bool = True,
        max_sim_time: float | None = None,
        on_control_step: Callable[["SimWorld"], None] | None = None,
    ) -> None:
        decimation = self.cfg.timing.decimation
        realtime = self.cfg.timing.realtime
        wall_start = time.perf_counter()
        sim_start = self.data.time

        def period() -> None:
            self.step(decimation)
            if on_control_step is not None:
                on_control_step(self)

        def pace() -> None:
            if realtime:
                ahead = (
                    wall_start + (self.data.time - sim_start)
                ) - time.perf_counter()
                if ahead > 0:
                    time.sleep(ahead)

        if not gui:
            while max_sim_time is None or self.data.time < max_sim_time:
                period()
                pace()
            return

        import mujoco.viewer

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                period()
                viewer.sync()
                pace()
                if max_sim_time is not None and self.data.time >= max_sim_time:
                    break

    def sim_time(self) -> float:
        return float(self.data.time)

    def base_pose(self) -> tuple[np.ndarray, np.ndarray]:
        with self.lock:
            return (
                self.data.xpos[self._base_body_id].copy(),
                self.data.xquat[self._base_body_id].copy(),
            )

    def base_vel(self) -> tuple[np.ndarray, np.ndarray]:
        twist = np.zeros(6)
        with self.lock:
            mujoco.mj_objectVelocity(
                self.model,
                self.data,
                mujoco.mjtObj.mjOBJ_BODY,
                self._base_body_id,
                twist,
                0,
            )
        return twist[3:].copy(), twist[:3].copy()

    def joint_qpos(self) -> np.ndarray:
        with self.lock:
            return np.array([self.data.qpos[adr] for adr in self._joint_qadr])

    def joint_qvel(self) -> np.ndarray:
        with self.lock:
            return np.array([self.data.qvel[adr] for adr in self._joint_dofadr])

    def set_ctrl(self, ctrl) -> None:
        with self.lock:
            self.data.ctrl[:] = ctrl

    def camera_names(self) -> list[str]:
        return [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            for i in range(self.model.ncam)
        ]
