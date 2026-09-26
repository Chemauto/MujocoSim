from __future__ import annotations

import numpy as np

from mujocosim.config import ActionsCfg, ActionPin, ConfigError, RobotMeta


def _per_joint(value: float | list[float] | None, n: int, default: float) -> np.ndarray:
    if value is None:
        return np.full(n, default, dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n, float(arr), dtype=float)
    if arr.shape != (n,):
        raise ConfigError(f"逐关节参数长度 {arr.shape} != {n}")
    return arr


def resolve_joints(
    actions: ActionsCfg, meta: RobotMeta, joint_order: list[str] | None
) -> list[str]:
    if actions.mapping == "auto":
        return list(joint_order or meta.joint_names)
    assert isinstance(actions.mapping, list)
    return [pin.joint for pin in actions.mapping]


def map_actions(
    action: np.ndarray,
    actions: ActionsCfg,
    meta: RobotMeta,
    joint_order: list[str] | None,
) -> np.ndarray:
    if actions.mode == "torque":
        raise ConfigError("actions.mode=torque 暂未实现")
    action = np.asarray(action, dtype=float)
    joints = resolve_joints(actions, meta, joint_order)
    n = len(joints)
    if action.shape != (n,):
        raise ConfigError(f"策略输出维度 {action.shape} != {n}")

    scale = _per_joint(actions.scale, n, 1.0)
    sign = _per_joint(actions.sign, n, 1.0)
    offset = _per_joint(actions.offset, n, np.nan)
    limit = _per_joint(actions.limit, n, np.inf)

    by_name = dict(zip(meta.joint_names, meta.default_joint_pos))
    for i, joint in enumerate(joints):
        if not np.isfinite(offset[i]):
            offset[i] = by_name[joint] if actions.mode == "position_offset" else 0.0

    if isinstance(actions.mapping, list):
        overrides = {pin.action: pin for pin in actions.mapping}
        for i, pin in overrides.items():
            if pin.scale is not None:
                scale[i] = pin.scale
            if pin.sign is not None:
                sign[i] = pin.sign
            if pin.offset is not None:
                offset[i] = pin.offset
            if pin.limit is not None:
                limit[i] = pin.limit

    clipped = np.clip(action, -limit, limit)
    targets = offset + sign * scale * clipped

    out = np.zeros(len(meta.joint_names), dtype=float)
    for joint, target in zip(joints, targets):
        out[meta.joint_names.index(joint)] = target
    return out


def gains_in_motor_order(
    actions: ActionsCfg,
    meta: RobotMeta,
    joint_order: list[str] | None,
    default_kp: float = 25.0,
    default_kd: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    joints = resolve_joints(actions, meta, joint_order)
    kp = _per_joint(actions.control.get("kp"), len(joints), default_kp)
    kd = _per_joint(actions.control.get("kd"), len(joints), default_kd)
    kp_out = np.zeros(len(meta.joint_names))
    kd_out = np.zeros(len(meta.joint_names))
    for joint, p, d in zip(joints, kp, kd):
        idx = meta.joint_names.index(joint)
        kp_out[idx] = p
        kd_out[idx] = d
    return kp_out, kd_out
