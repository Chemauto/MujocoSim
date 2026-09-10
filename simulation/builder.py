"""builder.py —— 运行时组装 MuJoCo 场景(MjSpec)。

机器人 XML + 场景模板 + 配置里的物体/相机 -> 一个完整 MjModel。
全流程不写临时 XML、不硬编码任何机器人/相机信息。

为什么用 from_string + meshdir 而不是 from_file:
  MuJoCo 的 C 层 fopen 在 Windows 上打不开含中文/非 ASCII 的路径,
  Python 读文本 + 指定绝对 meshdir 可以绕开这个限制,项目放哪都能跑。
"""
from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from common.config import CameraCfg, SimConfig
from .terrain import add_terrain, euler_deg_to_quat


_ASSET_EXTS = {".obj", ".stl", ".msh", ".png", ".jpg"}


def _load_spec(xml_path: Path) -> mujoco.MjSpec:
    """读 MJCF 文本 -> MjSpec,mesh/纹理以字节流喂进 spec.assets。

    不能依赖 MuJoCo 自己去 open 文件:它的 C 层 fopen 在 Windows 上
    打不开含中文/非 ASCII 的路径;Python 读字节 + assets 字典可以绕开,
    项目放在任何目录都能跑。
    """
    xml_path = xml_path.resolve()
    spec = mujoco.MjSpec.from_string(xml_path.read_text(encoding="utf-8"))
    meshdir = spec.meshdir
    if meshdir:
        asset_dir = (xml_path.parent / meshdir).resolve()
        if asset_dir.is_dir():
            assets: dict[str, bytes] = {}
            for f in sorted(asset_dir.rglob("*")):
                if f.suffix.lower() in _ASSET_EXTS:
                    key = (Path(meshdir) / f.relative_to(asset_dir)).as_posix()
                    assets[key] = f.read_bytes()
            spec.assets = assets
    return spec


def _strip_worldbody_extras(spec: mujoco.MjSpec) -> None:
    """剥掉机器人 MJCF worldbody 层的 floor/light 等"场景元素",
    这些由我们的场景模板统一提供,留着会重名冲突。"""
    wb = spec.worldbody
    for g in list(wb.geoms):
        spec.delete(g)
    for l in list(wb.lights):
        spec.delete(l)


def _ensure_freejoint(root: mujoco.MjsBody) -> None:
    """根 body 没有 freejoint 就补一个(有的资产是固定基座格式)。"""
    has_free = any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in root.joints)
    if not has_free:
        root.add_freejoint()


def lookat_quat(pos: list[float], lookat: list[float]) -> np.ndarray:
    """相机在 pos、看向 lookat 时的四元数(w,x,y,z)。

    MuJoCo 相机沿自身 -z 看、+y 朝上;这里构造一个满足该约定的姿态。
    """
    pos, lookat = np.asarray(pos, float), np.asarray(lookat, float)
    forward = lookat - pos
    n = np.linalg.norm(forward)
    if n < 1e-9:
        raise ValueError(f"lookat 与 pos 重合,无法确定朝向: {pos}")
    forward /= n
    z = -forward                                   # 相机 +z 指向身后
    up = np.array([0.0, 0.0, 1.0])
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-6:                   # 垂直朝上/朝下时选一个参考轴
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    # 旋转矩阵列 = 相机本体轴(x 右, y 上, z 朝后)在父坐标系中的方向
    mat = np.array([x[0], y[0], z[0],
                    x[1], y[1], z[1],
                    x[2], y[2], z[2]])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat)
    return quat


def _add_camera(spec: mujoco.MjSpec, cfg: CameraCfg, base_body: str) -> None:
    """按配置加一台相机:决定挂载 body 和姿态。

    attach/track 写 "BASE" 是别名 = 当前机器人的根 body,换机器人不用改这行。
    """
    resolve = lambda name: base_body if name == "BASE" else name
    parent = spec.body(resolve(cfg.attach)) if cfg.attach else spec.worldbody
    if parent is None:
        raise ValueError(f"相机 '{cfg.name}' 的挂载 body '{cfg.attach}' 不存在")
    if spec.camera(cfg.name) is not None:
        raise ValueError(f"相机名 '{cfg.name}' 与机器人自带相机重名,请换个名字")
    kwargs: dict = dict(name=cfg.name, pos=cfg.pos, fovy=cfg.fovy)
    if cfg.track:                                  # 始终看向某个 body(原生跟踪)
        target = resolve(cfg.track)
        if spec.body(target) is None:
            raise ValueError(f"相机 '{cfg.name}' track 的 body '{target}' 不存在")
        kwargs["mode"] = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
        kwargs["targetbody"] = target
    elif cfg.lookat:                               # 构建时把 lookat 解析成固定姿态
        kwargs["quat"] = lookat_quat(cfg.pos, cfg.lookat)
    elif cfg.euler is not None:                    # euler 是角度制,MJCF 需要弧度
        kwargs["euler"] = [math.radians(a) for a in cfg.euler]
    elif cfg.quat is not None:
        kwargs["quat"] = cfg.quat
    parent.add_camera(**kwargs)


_GEOM_TYPES = {
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
}


def _add_object(spec: mujoco.MjSpec, obj) -> None:
    """加一个可被 YOLO 检测的物体(命名 body + geom,默认带 freejoint)。"""
    if obj.type not in _GEOM_TYPES:
        raise ValueError(f"物体 '{obj.name}' 类型 '{obj.type}' 不支持,"
                         f"可选: {sorted(_GEOM_TYPES)}")
    body = spec.worldbody.add_body(name=obj.name, pos=obj.pos)
    if not obj.static:
        body.add_freejoint()
    kw: dict = dict(name=f"{obj.name}_geom", type=_GEOM_TYPES[obj.type],
                    size=obj.geom_size(), rgba=obj.rgba)
    if obj.euler:                              # 度制 ZYX 欧拉角
        kw["quat"] = euler_deg_to_quat(obj.euler)
    if obj.mass is not None:
        kw["mass"] = obj.mass
    if obj.friction is not None:
        kw["friction"] = obj.friction
    body.add_geom(**kw)


def build(cfg: SimConfig) -> tuple[mujoco.MjModel, dict]:
    """配置 -> (MjModel, robot_meta)。"""
    meta = cfg.load_robot_meta()
    base_body = meta["base_body"]
    spawn = list(cfg.spawn if cfg.spawn else meta.get("default_spawn", [0, 0, 1.0]))

    scene = mujoco.MjSpec.from_string(_scene_template())

    robot = _load_spec(cfg.robot_dir / meta["xml"])
    _strip_worldbody_extras(robot)
    # 机器人的求解器设置(cone/impratio 等)并入场景,避免 attach 冲突告警
    scene.option.cone = robot.option.cone
    scene.option.impratio = robot.option.impratio
    scene.option.timestep = cfg.timestep

    site = scene.worldbody.add_site(name="robot_mount", pos=[0, 0, 0])
    scene.attach(robot, site=site, prefix="")      # prefix="" 保持原始 body 名

    root = scene.body(base_body)
    if root is None:
        raise ValueError(f"机器人 {cfg.robot} 没有根 body '{base_body}',"
                         f"请检查 {cfg.robot_dir / 'robot.yaml'}")
    _ensure_freejoint(root)
    root.pos = spawn                               # 出生点由配置决定

    for obj in cfg.objects:
        _add_object(scene, obj)
    for feature in cfg.terrain:                # 地形(楼梯/乱石地/高程图…)
        scene.assets = {**scene.assets, **add_terrain(scene, feature)}
    for cam in cfg.cameras:
        _add_camera(scene, cam, base_body)

    # viewer 初始视野,随机器人尺寸走
    extent = float(meta.get("view_extent", 1.0))
    scene.stat.center = [0.0, 0.0, spawn[2] * 0.5]
    scene.stat.extent = extent

    # 离屏渲染缓冲 >= 最大相机分辨率(默认 640x480 会限制大分辨率相机)
    widths = [c.width for c in cfg.cameras] or [640]
    heights = [c.height for c in cfg.cameras] or [480]
    scene.visual.global_.offwidth = max(640, *widths)
    scene.visual.global_.offheight = max(480, *heights)

    model = scene.compile()
    return model, meta


_SCENE_TEMPLATE = """
<mujoco model="platform_scene">
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <global azimuth="-120" elevation="-25"/>
  </visual>
  <asset>
    <!-- 场景自有资产统一用 scene_ 前缀,避免和机器人自带的资产重名 -->
    <texture type="skybox" builtin="gradient" rgb1="0.35 0.5 0.7" rgb2="0 0 0"
             width="512" height="3072"/>
    <texture type="2d" name="scene_ground" builtin="checker" mark="edge"
             rgb1="0.25 0.3 0.35" rgb2="0.15 0.2 0.25" markrgb="0.85 0.85 0.85"
             width="300" height="300"/>
    <material name="scene_ground" texture="scene_ground" texuniform="true"
              texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light name="scene_top_light" pos="0 0 3.5" dir="0 0 -1" directional="true"/>
    <geom name="scene_floor" type="plane" size="10 10 0.05" material="scene_ground"/>
  </worldbody>
</mujoco>
"""


def _scene_template() -> str:
    return _SCENE_TEMPLATE
