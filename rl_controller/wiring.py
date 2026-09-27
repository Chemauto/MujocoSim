from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from mujocosim.config import (
    ActionsCfg,
    ConfigError,
    ObsPin,
    RobotMeta,
    parse_actions,
    parse_obs_pin,
    pin_dim,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
POLICIES_ROOT = REPO_ROOT / "rl_controller" / "robot"

_FREE_DIM_SOURCES = frozenset(
    {"constant_zero", "extra_onnx_input", "motion_command", "motion_anchor_ori"}
)


@dataclass
class PolicyProfile:
    name: str
    model: str
    observations: list[ObsPin]
    actions: ActionsCfg
    joint_order: list[str] | None = None
    motion: str | None = None
    input_name: str = "obs"
    output_name: str = "actions"
    obs_dim: int = 0
    action_dim: int = 0


def policies_path(robot: str) -> Path:
    return POLICIES_ROOT / robot / "policies.yaml"


def _as_dict(value: Any, ctx: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{ctx} 应为映射(dict)")
    return value


def _as_list(value: Any, ctx: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ConfigError(f"{ctx} 应为列表(list)")
    return value


def _resolve_actions(actions: ActionsCfg, meta: RobotMeta, ctx: str) -> int:
    if actions.mapping == "auto":
        return len(meta.joint_names)
    mapping = actions.mapping
    assert isinstance(mapping, list)
    indices = sorted(pin.action for pin in mapping)
    if indices != list(range(len(mapping))):
        raise ConfigError(
            f"{ctx} actions.mapping 的 action 编号应覆盖 0..{len(mapping) - 1}，实际 {indices}"
        )
    for pin in mapping:
        if pin.joint not in meta.joint_names:
            raise ConfigError(f"{ctx} actions.mapping 引用未知关节 '{pin.joint}'")
    return len(mapping)


def _derive_observations(
    pins: list[ObsPin],
    meta: RobotMeta,
    action_dim: int,
    command_dim: int,
    ctx: str,
) -> int:
    cursor = 0
    for i, pin in enumerate(pins):
        if (
            pin.source in ("base_ang_vel", "base_lin_vel", "base_orientation")
            and not meta.floating_base
        ):
            raise ConfigError(
                f"{ctx} 第 {i} 根引脚 {pin.source} 仅适用于浮动基座机器人（{meta.name}）"
            )
        dim = pin_dim(pin, len(meta.joint_names), action_dim, command_dim)
        if dim is None:
            if pin.source not in _FREE_DIM_SOURCES:
                raise ConfigError(f"{ctx} 第 {i} 根引脚 {pin.source} 无法推导维度")
            dim = pin.params.get("dim")
            if dim is None:
                raise ConfigError(
                    f"{ctx} 第 {i} 根引脚 {pin.source} 是自由维，需要 params.dim"
                )
            dim = int(dim)
        expected = dim * pin.history
        start, end = cursor, cursor + expected
        if pin.span is None:
            pin.span = (start, end)
        elif pin.span != (start, end):
            raise ConfigError(
                f"{ctx} 第 {i} 根引脚 {pin.source} 标注 span={list(pin.span)}，"
                f"按顺序实际应为 [{start}, {end})"
            )
        cursor = end
    return cursor


def load_policies(
    path: Path,
    meta: RobotMeta,
    command_dim: int = 3,
) -> dict[str, PolicyProfile]:
    if not path.is_file():
        raise ConfigError(f"找不到接线配置: {path}")
    raw = _as_dict(yaml.safe_load(path.read_text(encoding="utf-8")), str(path))
    policies_raw = _as_dict(raw.get("policies"), f"{path.name}.policies")
    profiles: dict[str, PolicyProfile] = {}
    for name, item in policies_raw.items():
        ctx = f"policies.{name}"
        data = _as_dict(item, ctx)
        model = data.get("model")
        if not isinstance(model, str) or not model:
            raise ConfigError(f"{ctx} 缺少 model（策略模型路径）")
        joint_order = data.get("joint_order")
        if joint_order is not None:
            joint_order = [str(j) for j in _as_list(joint_order, f"{ctx}.joint_order")]
            if sorted(joint_order) != sorted(meta.joint_names):
                raise ConfigError(
                    f"{ctx}.joint_order 必须是 joint_names 的一个排列（策略关节序 ↔ 电机序）"
                )
        observations = [
            parse_obs_pin(p, i)
            for i, p in enumerate(
                _as_list(data.get("observations"), f"{ctx}.observations")
            )
        ]
        actions = parse_actions(_as_dict(data.get("actions"), f"{ctx}.actions"))
        action_dim = _resolve_actions(actions, meta, ctx)
        obs_dim = _derive_observations(observations, meta, action_dim, command_dim, ctx)
        profiles[name] = PolicyProfile(
            name=name,
            model=model,
            observations=observations,
            actions=actions,
            joint_order=joint_order,
            motion=str(data["motion"]) if data.get("motion") else None,
            input_name=str(data.get("input_name", "obs")),
            output_name=str(data.get("output_name", "actions")),
            obs_dim=obs_dim,
            action_dim=action_dim,
        )
    active = raw.get("active")
    if active is not None and str(active) not in profiles:
        raise ConfigError(
            f"{path.name}.active 指向不存在的策略 '{active}'，可选 {sorted(profiles)}"
        )
    return profiles


def print_wiring_table(profile: PolicyProfile, meta: RobotMeta) -> None:
    print(f"[wiring] policy={profile.name}  model={profile.model}")
    print(
        f"         input={profile.input_name}[{profile.obs_dim}]  "
        f"output={profile.output_name}[{profile.action_dim}]"
    )
    for pin in profile.observations:
        assert pin.span is not None
        start, end = pin.span
        extras = []
        if pin.history > 1:
            extras.append(f"history={pin.history}")
        extras.extend(f"{k}={v}" for k, v in pin.params.items())
        suffix = f"  ({', '.join(extras)})" if extras else ""
        print(f"  obs[{start:3d}:{end:3d}]  {pin.source}{suffix}")
    actions = profile.actions
    if actions.mapping == "auto":
        joints = profile.joint_order or meta.joint_names
    else:
        assert isinstance(actions.mapping, list)
        joints = [p.joint for p in actions.mapping]
    if profile.joint_order is not None:
        print(f"  joint_order: {', '.join(profile.joint_order)}")
    print(f"  actions[0..{profile.action_dim - 1}] -> {', '.join(joints)}")
    print(
        f"  mode={actions.mode} scale={actions.scale} sign={actions.sign} "
        f"limit={actions.limit} control={actions.control}"
    )
