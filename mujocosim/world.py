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
        self.sim_dt = float(self.model.opt.timestep)
        self.decimation = max(1, round(cfg.timing.control_dt / self.sim_dt))
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
    def _explicit_timestep(xml_path: Path) -> float | None:
        """读取机器人 XML 里 <option> 显式声明的 timestep；未声明返回 None。"""
        import re

        text = xml_path.read_text(encoding="utf-8", errors="ignore")
        for tag in re.finditer(r"<option\b[^>]*>", text):
            m = re.search(r'timestep\s*=\s*"([^"]+)"', tag.group(0))
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    return None
        return None

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
        # 仅当机器人 XML 显式声明 timestep 时才取更小值（如 K1 的 0.001）；
        # go2 等 XML 未声明时 MjSpec 会填默认值 0.002，不应悄悄把物理步长
        # 加速到 500Hz——go2 训练与参考项目均为 0.005(200Hz)，500Hz 徒增一倍计算
        native_dt = SimWorld._explicit_timestep(robot_path)
        scene.option.timestep = (
            min(cfg.timing.sim_dt, native_dt) if native_dt else cfg.timing.sim_dt
        )
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
        decimation = self.decimation
        realtime = self.cfg.timing.realtime
        control_dt = self.cfg.timing.control_dt
        wall_start = time.perf_counter()
        sim_start = self.data.time
        stop_flag = threading.Event()

        def target_wall() -> float:
            return wall_start + (self.data.time - sim_start)

        def period() -> None:
            self.step(decimation)
            if on_control_step is not None:
                on_control_step(self)

        def pace() -> None:
            if not realtime:
                return
            ahead = target_wall() - time.perf_counter()
            if ahead > 0.002:
                time.sleep(ahead - 0.001)
            while target_wall() - time.perf_counter() > 0:
                pass

        def viewer_loop() -> None:
            import mujoco.viewer

            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                while viewer.is_running() and not stop_flag.is_set():
                    with self.lock:
                        viewer.sync()
                    time.sleep(control_dt)
            stop_flag.set()

        if gui:
            threading.Thread(target=viewer_loop, daemon=True).start()
            time.sleep(0.3)

        try:
            while max_sim_time is None or self.data.time < max_sim_time:
                period()
                pace()
                if stop_flag.is_set():
                    break
        finally:
            stop_flag.set()

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
