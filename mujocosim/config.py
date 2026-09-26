from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ASSETS_ROOT = REPO_ROOT / "assets" / "robots"
DEFAULT_SENSORS_ROOT = REPO_ROOT / "assets" / "sensors"

_SOURCES_FIXED_DIM: dict[str, int] = {
    "gyro": 3,
    "accelerometer": 3,
    "projected_gravity": 3,
    "base_ang_vel": 3,
    "base_lin_vel": 3,
    "base_orientation": 4,
    "phase_clock": 2,
}

_SOURCES_FLOATING_ONLY = frozenset({"base_ang_vel", "base_lin_vel", "base_orientation"})

_SOURCE_TYPES = frozenset(
    {
        "joint_qpos",
        "joint_qvel",
        "motor_target_cache",
        "gyro",
        "accelerometer",
        "projected_gravity",
        "base_ang_vel",
        "base_lin_vel",
        "base_orientation",
        "command",
        "last_action",
        "action_history",
        "signal_history",
        "phase_clock",
        "foot_contact",
        "virtual_servo_state",
        "height_scan",
        "constant_zero",
        "extra_onnx_input",
        "reference_motion",
    }
)

_ACTION_MODES = frozenset({"position", "position_offset", "torque"})
_COMMAND_SOURCES = frozenset({"ros2", "gamepad"})


class ConfigError(ValueError):
    pass


@dataclass
class TimingCfg:
    sim_dt: float = 0.005
    control_dt: float = 0.02
    decimation: int = 4
    realtime: bool = True


CONTROLLER_MODES = frozenset({"damping", "pd_stand", "position", "motion"})


@dataclass
class ControllerCfg:
    mode: str = "pd_stand"
    kp: float = 25.0
    kd: float = 0.5


@dataclass
class ObsPin:
    source: str
    span: tuple[int, int] | None = None
    params: dict[str, Any] = field(default_factory=dict)
    scale: float | list[float] = 1.0
    bias: float | list[float] = 0.0
    clip: tuple[float, float] | None = None
    history: int = 1


@dataclass
class ActionPin:
    action: int
    joint: str
    sign: float = 1.0
    scale: float | None = None
    offset: float | None = None
    limit: float | None = None


@dataclass
class ActionsCfg:
    mode: str = "position_offset"
    scale: float | list[float] = 0.25
    sign: float | list[float] = 1.0
    offset: float | list[float] = 0.0
    limit: float | list[float] | None = None
    control: dict[str, Any] = field(default_factory=lambda: {"kp": 25.0, "kd": 0.5})
    mapping: str | list[ActionPin] = "auto"


@dataclass
class CommandCfg:
    components: list[str] = field(default_factory=lambda: ["vx", "vy", "yaw_rate"])
    source: str = "ros2"
    limits: list[tuple[float, float]] = field(
        default_factory=lambda: [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
    )
    defaults: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    smoothing: float = 0.0


@dataclass
class CameraDevice:
    name: str
    type: str = "camera"
    fovy: float = 75.0
    width: int = 640
    height: int = 480
    rgb: bool = True
    depth: bool = False
    fps: float = 30.0


@dataclass
class LidarDevice:
    name: str
    type: str = "lidar"
    lidar_type: str = "3d"
    h_angles: tuple[float, float, float] = (-135.0, 135.0, 5.0)
    v_angles: tuple[float, float, float] | None = None
    max_range: float = 30.0
    fps: float = 10.0


@dataclass
class HeightmapDevice:
    name: str
    type: str = "heightmap"
    size: tuple[float, float] = (1.6, 1.0)
    resolution: float = 0.1
    fps: float = 20.0


DeviceCfg = CameraDevice | LidarDevice | HeightmapDevice


@dataclass
class MountCfg:
    device: str
    name: str
    attach: str | None = None
    pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rpy: list[float] | None = None
    lookat: list[float] | None = None
    quat: list[float] | None = None


@dataclass
class Ros2Cfg:
    namespace: str = ""
    domain_id: int = 0


@dataclass
class RobotMeta:
    name: str
    xml: str
    base_body: str
    default_spawn: list[float]
    default_joint_pos: list[float]
    floating_base: bool
    control: str
    joint_names: list[str]
    description: str = ""


@dataclass
class SimConfig:
    robot: str
    scene: str
    spawn: list[float]
    timing: TimingCfg
    controller: ControllerCfg
    policy_name: str | None
    command: CommandCfg
    mounts: list[MountCfg]
    devices: dict[str, DeviceCfg]
    ros2: Ros2Cfg
    robot_meta: RobotMeta
    config_path: Path


def _known_keys(cls: type) -> set[str]:
    return {f.name for f in fields(cls)}


def _warn_unknown(data: dict[str, Any], cls: type, ctx: str) -> None:
    known = _known_keys(cls)
    for key in data:
        if key not in known:
            print(f"[config] 警告: {ctx} 未知键 '{key}'", file=sys.stderr)


def _require_mapping(data: Any, ctx: str) -> dict[str, Any]:
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{ctx} 应为映射(dict)，实际是 {type(data).__name__}")
    return data


def _require_list(data: Any, ctx: str) -> list[Any]:
    if data is None:
        return []
    if not isinstance(data, list):
        raise ConfigError(f"{ctx} 应为列表(list)，实际是 {type(data).__name__}")
    return data


def _float_list(value: Any, ctx: str, length: int | None = None) -> list[float]:
    items = _require_list(value, ctx)
    if length is not None and len(items) != length:
        raise ConfigError(f"{ctx} 长度应为 {length}，实际 {len(items)}")
    try:
        return [float(v) for v in items]
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{ctx} 含非数值项: {error}") from error


def _float_triple(value: Any, ctx: str) -> tuple[float, float, float]:
    return tuple(_float_list(value, ctx, 3))  # type: ignore[return-value]


def _span(value: Any, ctx: str) -> tuple[int, int]:
    items = _require_list(value, ctx)
    if len(items) != 2:
        raise ConfigError(f"{ctx} 应为 [起, 止) 两个整数")
    try:
        start, end = int(items[0]), int(items[1])
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{ctx} 应为整数: {error}") from error
    if start < 0 or end <= start:
        raise ConfigError(f"{ctx} 非法区间 [{start}, {end})")
    return start, end


def _float_pair(value: Any, ctx: str) -> tuple[float, float]:
    items = _float_list(value, ctx, 2)
    if items[0] > items[1]:
        raise ConfigError(f"{ctx} 下限大于上限: {items}")
    return items[0], items[1]


def _scale_bias(value: Any, ctx: str) -> float | list[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return _float_list(value, ctx)


def _parse_timing(data: dict[str, Any]) -> TimingCfg:
    _warn_unknown(data, TimingCfg, "timing")
    timing = TimingCfg(**data)
    if timing.sim_dt <= 0:
        raise ConfigError("timing.sim_dt 必须 > 0")
    if timing.control_dt <= 0:
        raise ConfigError("timing.control_dt 必须 > 0")
    if timing.decimation <= 0:
        raise ConfigError("timing.decimation 必须 > 0")
    expected = timing.sim_dt * timing.decimation
    if not math.isclose(expected, timing.control_dt, rel_tol=1e-6, abs_tol=1e-9):
        raise ConfigError(
            f"timing 不自洽: sim_dt({timing.sim_dt}) * decimation({timing.decimation}) "
            f"= {expected} != control_dt({timing.control_dt})"
        )
    return timing


def _parse_controller(data: dict[str, Any]) -> ControllerCfg:
    _warn_unknown(data, ControllerCfg, "controller")
    controller = ControllerCfg(**data)
    if controller.mode not in CONTROLLER_MODES:
        raise ConfigError(f"controller.mode 只能是 {sorted(CONTROLLER_MODES)}")
    if controller.kp < 0 or controller.kd < 0:
        raise ConfigError("controller.kp/kd 必须 >= 0")
    return controller


def parse_obs_pin(data: Any, index: int) -> ObsPin:
    ctx = f"observations[{index}]"
    mapping = _require_mapping(data, ctx)
    _warn_unknown(mapping, ObsPin, ctx)
    source = mapping.get("source")
    if not isinstance(source, str) or not source:
        raise ConfigError(f"{ctx} 缺少 source")
    if source not in _SOURCE_TYPES:
        raise ConfigError(f"{ctx} 未知引脚源 '{source}'")
    span = (
        _span(mapping["span"], f"{ctx}.span")
        if mapping.get("span") is not None
        else None
    )
    history = int(mapping.get("history", 1))
    if history < 1:
        raise ConfigError(f"{ctx}.history 必须 >= 1")
    clip = mapping.get("clip")
    clip_tuple = None
    if clip is not None:
        clip_tuple = _float_pair(clip, f"{ctx}.clip")
    params = _require_mapping(mapping.get("params"), f"{ctx}.params")
    return ObsPin(
        source=source,
        span=span,
        params=params,
        scale=_scale_bias(mapping.get("scale", 1.0), f"{ctx}.scale"),
        bias=_scale_bias(mapping.get("bias", 0.0), f"{ctx}.bias"),
        clip=clip_tuple,
        history=history,
    )


def _parse_action_pin(data: Any, index: int) -> ActionPin:
    ctx = f"actions.mapping[{index}]"
    mapping = _require_mapping(data, ctx)
    _warn_unknown(mapping, ActionPin, ctx)
    try:
        action = int(mapping["action"])
        joint = str(mapping["joint"])
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigError(f"{ctx} 需要整数 action 与字符串 joint: {error}") from error
    return ActionPin(
        action=action,
        joint=joint,
        sign=float(mapping.get("sign", 1.0)),
        scale=float(mapping["scale"]) if mapping.get("scale") is not None else None,
        offset=float(mapping["offset"]) if mapping.get("offset") is not None else None,
        limit=float(mapping["limit"]) if mapping.get("limit") is not None else None,
    )


def parse_actions(data: dict[str, Any]) -> ActionsCfg:
    _warn_unknown(data, ActionsCfg, "actions")
    payload = dict(data)
    mapping_raw = payload.pop("mapping", "auto")
    control = _require_mapping(payload.pop("control", None), "actions.control")
    if not control:
        control = {"kp": 25.0, "kd": 0.5}
    parsed_control: dict[str, Any] = {}
    for key, value in control.items():
        parsed_control[str(key)] = (
            _float_list(value, f"actions.control.{key}")
            if isinstance(value, list)
            else float(value)
        )
    for key in ("scale", "sign", "offset", "limit"):
        if key in payload:
            payload[key] = (
                _float_list(payload[key], f"actions.{key}")
                if isinstance(payload[key], list)
                else float(payload[key])
            )
    actions = ActionsCfg(control=parsed_control, mapping="auto", **payload)
    if isinstance(mapping_raw, str):
        if mapping_raw != "auto":
            raise ConfigError("actions.mapping 只能是 'auto' 或逐根接线列表")
        actions.mapping = "auto"
    else:
        items = _require_list(mapping_raw, "actions.mapping")
        actions.mapping = [_parse_action_pin(item, i) for i, item in enumerate(items)]
    return actions


def _parse_command(data: dict[str, Any]) -> CommandCfg:
    _warn_unknown(data, CommandCfg, "command")
    payload = dict(data)
    components = payload.pop("components", None) or ["vx", "vy", "yaw_rate"]
    if not isinstance(components, list) or not all(
        isinstance(c, str) for c in components
    ):
        raise ConfigError("command.components 应为字符串列表")
    limits_raw = payload.pop("limits", None)
    defaults_raw = payload.pop("defaults", None)
    n = len(components)
    limits = (
        [
            _float_pair(item, f"command.limits[{i}]")
            for i, item in enumerate(_require_list(limits_raw, "command.limits"))
        ]
        if limits_raw is not None
        else [(-1.0, 1.0)] * n
    )
    defaults = (
        _float_list(defaults_raw, "command.defaults")
        if defaults_raw is not None
        else [0.0] * n
    )
    command = CommandCfg(
        components=components, limits=limits, defaults=defaults, **payload
    )
    if len(command.limits) != n:
        raise ConfigError(
            f"command.limits 长度 {len(command.limits)} != components 长度 {n}"
        )
    if len(command.defaults) != n:
        raise ConfigError(
            f"command.defaults 长度 {len(command.defaults)} != components 长度 {n}"
        )
    if command.source not in _COMMAND_SOURCES:
        raise ConfigError(f"command.source 只能是 {sorted(_COMMAND_SOURCES)}")
    if not (0.0 <= command.smoothing <= 1.0):
        raise ConfigError("command.smoothing 必须在 [0, 1]")
    return command


def load_device(name: str, sensors_root: Path | None = None) -> DeviceCfg:
    root = sensors_root if sensors_root is not None else DEFAULT_SENSORS_ROOT
    path = root / f"{name}.yaml"
    if not path.is_file():
        raise ConfigError(f"找不到传感器设备定义: {path}")
    raw = _require_mapping(yaml.safe_load(path.read_text(encoding="utf-8")), str(path))
    kind = str(raw.get("type", ""))
    if kind == "camera":
        _warn_unknown(raw, CameraDevice, f"device({name})")
        device: DeviceCfg = CameraDevice(
            name=name,
            fovy=float(raw.get("fovy", 75.0)),
            width=int(raw.get("width", 640)),
            height=int(raw.get("height", 480)),
            rgb=bool(raw.get("rgb", True)),
            depth=bool(raw.get("depth", False)),
            fps=float(raw.get("fps", 30.0)),
        )
        if (
            device.fovy <= 0
            or device.width <= 0
            or device.height <= 0
            or device.fps <= 0
        ):
            raise ConfigError(f"device({name}) fovy/width/height/fps 必须 > 0")
        return device
    if kind == "lidar":
        _warn_unknown(raw, LidarDevice, f"device({name})")
        lidar_type = str(raw.get("lidar_type", "3d"))
        if lidar_type not in ("2d", "3d"):
            raise ConfigError(f"device({name}) lidar_type 只能是 2d 或 3d")
        v_angles = (
            _float_triple(raw["v_angles"], f"device({name}).v_angles")
            if raw.get("v_angles") is not None
            else None
        )
        if lidar_type == "3d" and v_angles is None:
            raise ConfigError(f"device({name}) lidar_type=3d 需要 v_angles")
        device = LidarDevice(
            name=name,
            lidar_type=lidar_type,
            h_angles=_float_triple(
                raw.get("h_angles", [-135, 135, 5.0]), f"device({name}).h_angles"
            ),
            v_angles=v_angles,
            max_range=float(raw.get("max_range", 30.0)),
            fps=float(raw.get("fps", 10.0)),
        )
        if device.h_angles[2] <= 0 or device.max_range <= 0 or device.fps <= 0:
            raise ConfigError(f"device({name}) 步长/max_range/fps 必须 > 0")
        if device.v_angles is not None and device.v_angles[2] <= 0:
            raise ConfigError(f"device({name}).v_angles 步长必须 > 0")
        return device
    if kind == "heightmap":
        _warn_unknown(raw, HeightmapDevice, f"device({name})")
        size = raw.get("size", [1.6, 1.0])
        device = HeightmapDevice(
            name=name,
            size=tuple(_float_list(size, f"device({name}).size", 2)),  # type: ignore[arg-type]
            resolution=float(raw.get("resolution", 0.1)),
            fps=float(raw.get("fps", 20.0)),
        )
        if (
            device.size[0] <= 0
            or device.size[1] <= 0
            or device.resolution <= 0
            or device.fps <= 0
        ):
            raise ConfigError(f"device({name}) size/resolution/fps 必须 > 0")
        return device
    raise ConfigError(
        f"device({name}) 未知 type '{kind}'（camera / lidar / heightmap）"
    )


def _parse_mounts(
    data: list[Any], sensors_root: Path | None = None
) -> tuple[list[MountCfg], dict[str, DeviceCfg]]:
    mounts: list[MountCfg] = []
    devices: dict[str, DeviceCfg] = {}
    for i, item in enumerate(data):
        ctx = f"mounts[{i}]"
        mapping = _require_mapping(item, ctx)
        _warn_unknown(mapping, MountCfg, ctx)
        device = mapping.get("device")
        if not isinstance(device, str) or not device:
            raise ConfigError(f"{ctx} 缺少 device（assets/sensors/ 下的设备名）")
        if device not in devices:
            devices[device] = load_device(device, sensors_root)
        orientations = [
            k for k in ("rpy", "lookat", "quat") if mapping.get(k) is not None
        ]
        if len(orientations) > 1:
            raise ConfigError(
                f"{ctx} 姿态来源只能选一个: rpy/lookat/quat，实际 {orientations}"
            )
        mount = MountCfg(
            device=device,
            name=str(mapping.get("name", device)),
            attach=str(mapping["attach"])
            if mapping.get("attach") is not None
            else None,
            pos=_float_list(mapping.get("pos", [0, 0, 0]), f"{ctx}.pos", 3),
            rpy=_float_list(mapping["rpy"], f"{ctx}.rpy", 3)
            if mapping.get("rpy") is not None
            else None,
            lookat=_float_list(mapping["lookat"], f"{ctx}.lookat", 3)
            if mapping.get("lookat") is not None
            else None,
            quat=_float_list(mapping["quat"], f"{ctx}.quat", 4)
            if mapping.get("quat") is not None
            else None,
        )
        mounts.append(mount)
    names = [m.name for m in mounts]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ConfigError(f"mounts name 重复: {sorted(duplicates)}")
    return mounts, devices


def _parse_ros2(data: dict[str, Any]) -> Ros2Cfg:
    _warn_unknown(data, Ros2Cfg, "ros2")
    return Ros2Cfg(**data)


def _load_robot_meta(robot: str, assets_root: Path) -> RobotMeta:
    robot_dir = assets_root / robot
    meta_path = robot_dir / "robot.yaml"
    if not meta_path.is_file():
        raise ConfigError(f"找不到机器人元数据: {meta_path}")
    raw = _require_mapping(
        yaml.safe_load(meta_path.read_text(encoding="utf-8")), str(meta_path)
    )
    _warn_unknown(raw, RobotMeta, f"robot.yaml({robot})")
    try:
        meta = RobotMeta(
            name=str(raw["name"]),
            xml=str(raw["xml"]),
            base_body=str(raw["base_body"]),
            default_spawn=_float_list(
                raw["default_spawn"], "robot.yaml.default_spawn", 3
            ),
            default_joint_pos=_float_list(
                raw["default_joint_pos"], "robot.yaml.default_joint_pos"
            ),
            floating_base=bool(raw["floating_base"]),
            control=str(raw["control"]),
            joint_names=[
                str(j)
                for j in _require_list(raw["joint_names"], "robot.yaml.joint_names")
            ],
            description=str(raw.get("description", "")),
        )
    except KeyError as error:
        raise ConfigError(f"robot.yaml({robot}) 缺少字段 {error}") from error
    if meta.control not in _ACTION_MODES and meta.control != "torque_pd":
        raise ConfigError(f"robot.yaml({robot}).control 非法: {meta.control}")
    if not meta.joint_names:
        raise ConfigError(f"robot.yaml({robot}).joint_names 不能为空")
    if len(set(meta.joint_names)) != len(meta.joint_names):
        raise ConfigError(f"robot.yaml({robot}).joint_names 有重复")
    if len(meta.default_joint_pos) != len(meta.joint_names):
        raise ConfigError(
            f"robot.yaml({robot}).default_joint_pos 长度 {len(meta.default_joint_pos)}"
            f" != joint_names 长度 {len(meta.joint_names)}"
        )
    return meta


def pin_dim(
    pin: ObsPin, n_joints: int, action_dim: int, command_dim: int
) -> int | None:
    source = pin.source
    if source in _SOURCES_FIXED_DIM:
        return _SOURCES_FIXED_DIM[source]
    if source in ("joint_qpos", "joint_qvel", "motor_target_cache"):
        return n_joints
    if source == "virtual_servo_state":
        return 2 * n_joints
    if source == "last_action":
        return action_dim
    if source == "action_history":
        return action_dim
    if source == "command":
        components = pin.params.get("components")
        if components is not None:
            return len(_require_list(components, "observations params.components"))
        return command_dim
    if source == "foot_contact":
        geoms = pin.params.get("foot_geoms")
        if geoms is None:
            raise ConfigError("foot_contact 引脚需要 params.foot_geoms 列表")
        return len(_require_list(geoms, "observations params.foot_geoms"))
    if source in ("signal_history", "height_scan", "reference_motion"):
        dim = pin.params.get("dim")
        if dim is None:
            raise ConfigError(f"{source} 引脚需要 params.dim")
        return int(dim)
    return None


def load_config(
    path: str | Path,
    assets_root: Path | None = None,
    robot: str | None = None,
) -> SimConfig:
    config_path = Path(path).resolve()
    raw = _require_mapping(
        yaml.safe_load(config_path.read_text(encoding="utf-8")), str(config_path)
    )
    top_known = {
        "robot",
        "spawn",
        "timing",
        "controller",
        "policy",
        "command",
        "scene",
        "mounts",
        "ros2",
    }
    _warn_unknown_keys = {k for k in raw if k not in top_known}
    for key in sorted(_warn_unknown_keys):
        print(f"[config] 警告: 顶层未知键 '{key}'", file=sys.stderr)

    robot = robot or raw.get("robot")
    if not isinstance(robot, str) or not robot:
        raise ConfigError("缺少顶层 robot 字段")

    root = Path(assets_root) if assets_root is not None else DEFAULT_ASSETS_ROOT
    meta = _load_robot_meta(robot, root)

    spawn_raw = raw.get("spawn")
    spawn = (
        _float_list(spawn_raw, "spawn", 3)
        if spawn_raw is not None
        else list(meta.default_spawn)
    )

    timing = _parse_timing(_require_mapping(raw.get("timing"), "timing"))
    controller = _parse_controller(
        _require_mapping(raw.get("controller"), "controller")
    )
    command = _parse_command(_require_mapping(raw.get("command"), "command"))
    mounts, devices = _parse_mounts(_require_list(raw.get("mounts"), "mounts"))
    ros2 = _parse_ros2(_require_mapping(raw.get("ros2"), "ros2"))

    policy_name = raw.get("policy")
    if policy_name is not None and (
        not isinstance(policy_name, str) or not policy_name
    ):
        raise ConfigError(
            "policy 应为策略名字符串（rl_controller/robot/<robot>/policies.yaml 里的名字）"
        )

    scene = raw.get("scene")
    if not isinstance(scene, str) or not scene:
        raise ConfigError("缺少顶层 scene 字段（场景 XML 路径，如 scenes/flat.xml）")
    scene_path = Path(scene)
    scene_full = scene_path if scene_path.is_absolute() else REPO_ROOT / scene_path
    if not scene_full.is_file():
        raise ConfigError(f"场景文件不存在: {scene_full}")

    return SimConfig(
        robot=robot,
        scene=scene,
        spawn=spawn,
        timing=timing,
        controller=controller,
        policy_name=policy_name,
        command=command,
        mounts=mounts,
        devices=devices,
        ros2=ros2,
        robot_meta=meta,
        config_path=config_path,
    )
