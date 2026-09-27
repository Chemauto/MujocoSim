from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from mujocosim.config import ConfigError, ObsPin, RobotMeta
from mujocosim.world import SimWorld


def _joint_indices(world: SimWorld, order: list[str] | None) -> list[int]:
    names = order or world.cfg.robot_meta.joint_names
    return [world.cfg.robot_meta.joint_names.index(n) for n in names]


def _joint_offset(
    pin: ObsPin, meta: RobotMeta, order: list[str] | None
) -> np.ndarray | float:
    offset = pin.params.get("offset")
    if offset is None:
        return 0.0
    if isinstance(offset, str):
        if offset != "zero_pose":
            raise ConfigError(f"{pin.source} 未知 offset 语义 '{offset}'")
        by_name = dict(zip(meta.joint_names, meta.default_joint_pos))
        return np.array([by_name[n] for n in (order or meta.joint_names)])
    return np.asarray(offset, dtype=float)


def _gravity_in_base(world: SimWorld) -> np.ndarray:
    rot = _base_rot(world)
    return rot.T @ np.array([0.0, 0.0, -1.0])


def _base_rot(world: SimWorld) -> np.ndarray:
    _, quat = world.base_pose()
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def sample_pin(
    pin: ObsPin,
    world: SimWorld,
    meta: RobotMeta,
    joint_order: list[str] | None,
    last_action: np.ndarray,
    command: np.ndarray,
    sensors: dict[str, Any] | None = None,
    motion: Any | None = None,
) -> np.ndarray:
    source = pin.source
    if source in ("gyro", "base_ang_vel"):
        _, ang = world.base_vel()
        return _base_rot(world).T @ ang
    if source == "accelerometer":
        acc = np.zeros(6)
        mujoco.mj_objectAcceleration(
            world.model,
            world.data,
            mujoco.mjtObj.mjOBJ_BODY,
            world._base_body_id,
            acc,
            0,
        )
        return _base_rot(world).T @ acc[3:].copy()
    if source == "projected_gravity":
        return _gravity_in_base(world)
    if source == "base_lin_vel":
        lin, _ = world.base_vel()
        return _base_rot(world).T @ lin
    if source == "base_orientation":
        _, quat = world.base_pose()
        return quat
    if source == "joint_qpos":
        idx = _joint_indices(world, joint_order)
        q = world.joint_qpos()[idx]
        return q - _joint_offset(pin, meta, joint_order)
    if source == "joint_qvel":
        idx = _joint_indices(world, joint_order)
        return world.joint_qvel()[idx]
    if source == "motor_target_cache":
        idx = _joint_indices(world, joint_order)
        q = world.data.ctrl.copy()[idx]
        return q - _joint_offset(pin, meta, joint_order)
    if source == "last_action":
        return np.asarray(last_action, dtype=float)
    if source == "command":
        return np.asarray(command, dtype=float)
    if source == "constant_zero":
        return np.zeros(int(pin.params["dim"]))
    if source == "motion_command":
        if motion is None:
            raise ConfigError("motion_command 引脚需要运动参考（policy.motion 未配置）")
        jp, jv = motion.clip.commands()
        return np.concatenate([jp, jv])
    if source == "motion_anchor_ori":
        if motion is None:
            raise ConfigError(
                "motion_anchor_ori 引脚需要运动参考（policy.motion 未配置）"
            )
        from .motion import matrix_from_quat, subtract_frame_transforms

        base_pos, base_quat = world.base_pose()
        mpos, mquat = motion.clip.anchor_pose()
        _, ori_b = subtract_frame_transforms(base_pos, base_quat, mpos, mquat)
        return matrix_from_quat(ori_b)[:, :2].reshape(-1)
    if source == "height_scan":
        if not sensors:
            raise ConfigError("height_scan 引脚需要传感器数据（SensorSuite 未接入）")
        name = pin.params.get("sensor")
        payload = (
            sensors.get(str(name))
            if name is not None
            else next(
                (v for v in sensors.values() if v.get("kind") == "heightmap"), None
            )
        )
        if payload is None:
            raise ConfigError(f"height_scan 找不到高度图传感器 '{name}'")
        heights = np.asarray(payload["heights"], dtype=float)
        return np.nan_to_num(heights, nan=0.0)
    raise ConfigError(f"引脚源 '{source}' 暂未实现采样")


def assemble_obs(
    pins: list[ObsPin],
    world: SimWorld,
    meta: RobotMeta,
    joint_order: list[str] | None,
    last_action: np.ndarray,
    command: np.ndarray,
    sensors: dict[str, Any] | None = None,
    motion: Any | None = None,
) -> np.ndarray:
    parts = []
    for pin in pins:
        value = np.atleast_1d(
            sample_pin(
                pin, world, meta, joint_order, last_action, command, sensors, motion
            )
        ).astype(float)
        if pin.history > 1:
            raise ConfigError("history>1 暂未实现（阶段6 后续）")
        scale = np.asarray(pin.scale, dtype=float)
        bias = np.asarray(pin.bias, dtype=float)
        value = value * scale + bias
        if pin.clip is not None:
            value = np.clip(value, pin.clip[0], pin.clip[1])
        parts.append(value)
    return np.concatenate(parts) if parts else np.zeros(0)
