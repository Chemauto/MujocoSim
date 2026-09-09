"""config.py —— 场景配置:加载 YAML,补全默认值。

配置是平台的唯一"可变项"来源:换机器人、换相机位置、改物体,
都只改 YAML,不改代码。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSET_ROBOTS_DIR = PROJECT_ROOT / "assets" / "robots"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "go2.yaml"


@dataclass
class ObjectCfg:
    """场景中的一个可检测物体。size 用"完整尺寸"(米),内部转 MJCF 半尺寸。"""
    name: str
    type: str = "box"                       # box / cylinder / sphere / capsule
    size: list[float] = field(default_factory=lambda: [0.08, 0.08, 0.08])
    pos: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.2])
    rgba: list[float] = field(default_factory=lambda: [0.85, 0.2, 0.2, 1.0])
    euler: list[float] | None = None        # 度,ZYX;None = 不旋转
    static: bool = False                    # True = 固定不动物体(障碍),不加 freejoint
    mass: float | None = None               # 千克;None = 由尺寸/密度推算
    friction: list[float] | None = None     # [滑动, 扭转, 滚动]

    def geom_size(self) -> list[float]:
        """MJCF 的 size 语义:box=半边长, cylinder/capsule=[半径, 半长], sphere=半径。"""
        if self.type == "box":
            return [s / 2.0 for s in self.size[:3]]
        if self.type in ("cylinder", "capsule"):
            return [self.size[0], self.size[1] / 2.0]
        if self.type == "sphere":
            return [self.size[0]]
        raise ValueError(f"未知物体类型: {self.type}")


@dataclass
class CameraCfg:
    """一台相机。attach 省略 = 世界固定相机;否则挂在指定 body 上跟着动。

    姿态(优先级从高到低):track(始终看向某 body)> lookat(看向一个点,
    构建时解析为固定姿态)> euler(度)> quat(w,x,y,z)> 默认朝 -z。
    """
    name: str
    attach: str | None = None               # 挂载到的 body 名;None = 世界固定
    pos: list[float] = field(default_factory=lambda: [0.0, -2.0, 1.5])
    track: str | None = None                # mjCAM_TARGETBODY 的目标 body
    lookat: list[float] | None = None
    euler: list[float] | None = None        # 度
    quat: list[float] | None = None         # w, x, y, z
    fovy: float = 60.0                      # 垂直视场角(度)
    width: int = 640
    height: int = 480


@dataclass
class YoloCfg:
    enabled: bool = False
    camera: str = ""                        # 用哪台相机做检测
    weights: str = "yolov8s-worldv2.pt"     # 本地路径或 ultralytics 自动下载的名字
    conf: float = 0.2
    classes: list[str] | None = None        # None = 自动用场景物体名
    interval: float = 0.5                   # 检测周期(仿真时间秒)
    save_dir: str = "outputs"               # 标注图保存目录(相对项目根)


@dataclass
class ControllerCfg:
    """控制器配置。关节相关的量全部按 joint_names("策略关节序")排列。

    joint_names 决定了策略观测/动作的关节排列,例如 go2 walk 策略是
    [FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, ..., RL_calf](见 configs/go2.yaml)。
    policy 指向导出的 TorchScript 模型;留空则 RL行走 模式不可用。
    """
    joint_names: list[str] = field(default_factory=list)
    step_dt: float = 0.02                   # 控制周期(策略 50Hz)
    stiffness: float | list[float] = 25.0   # PD 的 kp(标量 = 全关节一样)
    damping: float | list[float] = 0.5      # PD 的 kd
    default_joint_pos: list[float] = field(default_factory=list)   # 默认站姿
    action_scale: float | list[float] = 0.25
    action_clip: float = 100.0
    obs_clip: float = 100.0
    policy: str = ""                        # TorchScript 路径;空 = 无策略
    obs_order: str = ""                     # "yaml" = deploy.yaml 文件序(推荐);空 = 字母序
    command_smoothing: float = 0.2          # 速度指令一阶平滑系数(与部署端键盘 EMA 一致)
    height_scan: dict = field(default_factory=lambda: {
        "size": [1.6, 1.0], "resolution": 0.1,
        "ray_offset_z": 20.0, "height_offset": 0.5})
    commands: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # vx, vy, wz
    initial_mode: str = "pd_stand"          # damping/force/pd_stand/position/policy


@dataclass
class SimConfig:
    robot: str = "go2"
    spawn: list[float] | None = None        # None = 用 robot.yaml 的 default_spawn
    timestep: float = 0.002
    realtime: bool = True                   # 主循环是否按真实时间步进
    objects: list[ObjectCfg] = field(default_factory=list)
    terrain: list[dict] = field(default_factory=list)   # 地形元素,见 simulation/terrain.py
    cameras: list[CameraCfg] = field(default_factory=list)
    yolo: YoloCfg = field(default_factory=YoloCfg)
    controller: ControllerCfg | None = None             # 控制器;无 = 自由落体演示

    # ---- 机器人资产路径 ----
    @property
    def robot_dir(self) -> Path:
        d = ASSET_ROBOTS_DIR / self.robot
        if not d.is_dir():
            avail = ", ".join(sorted(p.name for p in ASSET_ROBOTS_DIR.iterdir() if p.is_dir()))
            raise FileNotFoundError(f"assets/robots 下没有 '{self.robot}',可用: {avail}")
        return d

    def load_robot_meta(self) -> dict[str, Any]:
        meta_path = self.robot_dir / "robot.yaml"
        if not meta_path.is_file():
            raise FileNotFoundError(f"机器人 {self.robot} 缺少 robot.yaml: {meta_path}")
        with open(meta_path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def yolo_classes(self) -> list[str]:
        if self.yolo.classes:
            return list(self.yolo.classes)
        return [o.name for o in self.objects] if self.objects else ["box"]


def _objects(raw: list[dict]) -> list[ObjectCfg]:
    return [ObjectCfg(**{k: v for k, v in it.items() if k in ObjectCfg.__dataclass_fields__})
            for it in raw]


def _cameras(raw: list[dict]) -> list[CameraCfg]:
    return [CameraCfg(**{k: v for k, v in it.items() if k in CameraCfg.__dataclass_fields__})
            for it in raw]


def load_config(path: str | Path | None = None, robot: str | None = None) -> SimConfig:
    """读 YAML -> SimConfig。robot 参数用于命令行覆盖配置里的机器人。"""
    p = Path(path) if path else DEFAULT_CONFIG
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    with open(p, encoding="utf-8") as f:
        raw: dict = yaml.safe_load(f) or {}

    yolo_raw = raw.get("yolo") or {}
    yolo = YoloCfg(**{k: v for k, v in yolo_raw.items() if k in YoloCfg.__dataclass_fields__})

    controller = None
    if raw.get("controller"):
        c_raw = dict(raw["controller"])
        fields = ControllerCfg.__dataclass_fields__
        controller = ControllerCfg(**{k: v for k, v in c_raw.items() if k in fields})

    cfg = SimConfig(
        robot=robot or raw.get("robot", SimConfig.robot),
        spawn=raw.get("spawn"),
        timestep=float(raw.get("timestep", SimConfig.timestep)),
        realtime=bool(raw.get("realtime", SimConfig.realtime)),
        objects=_objects(raw.get("objects") or []),
        terrain=list(raw.get("terrain") or []),
        cameras=_cameras(raw.get("cameras") or []),
        yolo=yolo,
        controller=controller,
    )
    if cfg.yolo.enabled and cfg.cameras and not cfg.yolo.camera:
        cfg.yolo.camera = cfg.cameras[0].name   # 默认用第一台相机
    return cfg
