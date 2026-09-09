"""terrain.py —— 程序化地形生成(移植自 llm-legged-lab/Mujoco/terrain_tool)。

原工具是离线用 ElementTree 改 XML 再存盘;这里改成运行时直接往 MjSpec 里加,
参数名与原版 Add* 系列保持一致,方便对照 readme_zh.md:
  box            静态障碍箱            (AddBox / AddGeometry)
  stairs         楼梯                  (AddStairs)
  suspend_stairs 悬浮楼梯              (AddSuspendStairs)
  rough_ground   乱石地(随机方块阵)  (AddRoughGround)
  perlin_hfield  Perlin 噪声高程图地形 (AddPerlinHeighField)
  image_hfield   图片生成高程图地形    (AddHeighFieldFromImage)

与原版的差异:
  - euler 一律用"度"(YAML 友好),内部再转四元数;
  - Perlin 噪声用 numpy 自实现(原版的 `noise` 是 C 扩展,Windows/新 Python
    装不上),支持 seed 复现;
  - 高程图 PNG 通过 spec.assets 从内存喂给 MuJoCo,不落盘,
    因此项目放在中文路径下也能跑。

可推箱子(push_box 的 AddMovableBox)由场景配置的 objects 承担
(支持 mass/friction),不在这里重复。
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np

from sensors.camera import imread_u

# ---------------------------------------------------------------- 基础工具

def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX 欧拉角(弧度)-> 四元数 [w, x, y, z],与原版一致。"""
    cx, sx = np.cos(roll / 2), np.sin(roll / 2)
    cy, sy = np.cos(pitch / 2), np.sin(pitch / 2)
    cz, sz = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([cx * cy * cz + sx * sy * sz,
                     sx * cy * cz - cx * sy * sz,
                     cx * sy * cz + sx * cy * sz,
                     cx * cy * sz - sx * sy * cz])


def _quat_deg(euler_deg) -> np.ndarray:
    """度制 ZYX 欧拉角 -> 四元数。"""
    r = np.deg2rad(np.asarray(euler_deg, float))
    return euler_to_quat(*r)


# 对外别名:builder 组装 objects 时也用这个约定
euler_deg_to_quat = _quat_deg


def rot2d(x: float, y: float, yaw: float):
    """平面内绕 z 旋转(弧度)。"""
    return x * np.cos(yaw) - y * np.sin(yaw), x * np.sin(yaw) + y * np.cos(yaw)


def _add_box_geom(spec: mujoco.MjSpec, pos, euler_deg, size_full, rgba=None, name=None):
    """加静态 box,size 给完整尺寸(内部转 MuJoCo 半尺寸)。"""
    kw: dict = dict(type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=[s / 2.0 for s in size_full[:3]])
    if name:
        kw["name"] = name
    if rgba is not None:
        kw["rgba"] = rgba
    if any(euler_deg):
        kw["quat"] = _quat_deg(euler_deg)
    spec.worldbody.add_geom(pos=list(pos), **kw)


# ---------------------------------------------------------------- 各类地形

def _add_box(spec: mujoco.MjSpec, p: dict) -> None:
    _add_box_geom(spec, p.get("pos", [1.0, 0.0, 0.0]),
                  p.get("euler", [0, 0, 0]), p.get("size", [0.5, 0.5, 0.5]),
                  rgba=p.get("rgba"), name=p.get("name"))


def _add_stairs(spec: mujoco.MjSpec, p: dict) -> None:
    """楼梯:width 进深 / height 每级高 / length 横向长 / stair_nums 级数。"""
    init = np.asarray(p.get("init_pos", [1.0, 0.0, 0.0]), float)
    yaw = np.deg2rad(p.get("yaw", 0.0))
    width, height = p.get("width", 0.2), p.get("height", 0.15)
    length, n = p.get("length", 1.5), int(p.get("stair_nums", 10))
    lx, ly, lz = 0.0, 0.0, -0.5 * height
    for _ in range(n):
        lx += width
        lz += height
        x, y = rot2d(lx, ly, yaw)
        _add_box_geom(spec, [x + init[0], y + init[1], lz], [0, 0, 0],
                      [width, length, height], rgba=p.get("rgba"))


def _add_suspend_stairs(spec: mujoco.MjSpec, p: dict) -> None:
    """悬浮楼梯:每级板厚 |height-gap|,与下一级之间留 gap 空隙。"""
    init = np.asarray(p.get("init_pos", [1.0, 0.0, 0.0]), float)
    yaw = np.deg2rad(p.get("yaw", 0.0))
    width, height = p.get("width", 0.2), p.get("height", 0.15)
    length, gap = p.get("length", 1.5), p.get("gap", 0.1)
    n = int(p.get("stair_nums", 10))
    lx, ly, lz = 0.0, 0.0, -0.5 * height
    for _ in range(n):
        lx += width
        lz += height
        x, y = rot2d(lx, ly, yaw)
        _add_box_geom(spec, [x + init[0], y + init[1], lz], [0, 0, 0],
                      [width, length, abs(height - gap)], rgba=p.get("rgba"))


def _add_rough_ground(spec: mujoco.MjSpec, p: dict) -> None:
    """乱石地:网格排布的方块,尺寸/姿态/间隔带随机扰动。"""
    init = np.asarray(p.get("init_pos", [1.0, 0.0, 0.0]), float)
    yaw = np.deg2rad(p.get("yaw", 0.0))
    nx, ny = (p.get("nums") or [10, 10])
    box_size = np.asarray(p.get("box_size", [0.5, 0.5, 0.5]), float)
    box_euler = np.asarray(p.get("box_euler", [0.0, 0.0, 0.0]), float)
    sep = np.asarray(p.get("separation", [0.2, 0.2]), float)
    size_rand = np.asarray(p.get("box_size_rand", [0.05, 0.05, 0.05]), float)
    euler_rand = np.asarray(p.get("box_euler_rand", [0.2, 0.2, 0.2]), float)
    sep_rand = np.asarray(p.get("separation_rand", [0.05, 0.05]), float)
    rng = np.random.default_rng(p.get("seed", 0))

    def rand3(base, amp):
        return base + amp * rng.uniform(-1.0, 1.0, len(base))

    lx, ly, lz = 0.0, 0.0, -0.5 * box_size[2]
    cur_sep = rand3(sep, sep_rand)
    for _ in range(int(nx)):
        lx += cur_sep[0]
        ly = 0.0
        for _ in range(int(ny)):
            size = np.maximum(rand3(box_size, size_rand), 0.01)
            euler = rand3(box_euler, euler_rand)
            cur_sep = rand3(sep, sep_rand)
            ly += cur_sep[1]
            x, y = rot2d(lx, ly, yaw)
            pos = np.array([x + init[0], y + init[1], lz + size[2] - box_size[2]])
            _add_box_geom(spec, pos, euler, size, rgba=p.get("rgba"))


# ---- Perlin 噪声(numpy 自实现,免 C 扩展依赖) ----

def _hash_gradients(xi: np.ndarray, yi: np.ndarray, seed: int) -> np.ndarray:
    """整数格点 -> 伪随机单位梯度向量(角度),确定性可复现。"""
    h = (xi.astype(np.int64) * 374761393 + yi.astype(np.int64) * 668265263
         + seed * 1442695040888963407 % (1 << 62)) % 360
    ang = np.deg2rad(h.astype(float))
    return np.stack([np.cos(ang), np.sin(ang)], axis=-1)


def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def perlin2d(w: int, h: int, scale: float = 100.0, octaves: int = 6,
             persistence: float = 0.5, lacunarity: float = 2.0,
             seed: int = 0) -> np.ndarray:
    """经典 Perlin 梯度噪声 + fBm 多倍频,输出 [-1, 1] 的 (h, w) 数组。"""
    out = np.zeros((h, w), dtype=np.float64)
    amp, freq, total_amp = 1.0, 1.0 / max(scale, 1e-9), 0.0
    for o in range(octaves):
        xs = np.arange(w) * freq
        ys = np.arange(h) * freq
        xi, yi = np.floor(xs).astype(int), np.floor(ys).astype(int)
        xf, yf = xs - xi, ys - yi
        g = _hash_gradients(xi[None, :] + 0, yi[:, None] + 0, seed + o)
        g_x1 = _hash_gradients(xi[None, :] + 1, yi[:, None] + 0, seed + o)
        g_y1 = _hash_gradients(xi[None, :] + 0, yi[:, None] + 1, seed + o)
        g_xy = _hash_gradients(xi[None, :] + 1, yi[:, None] + 1, seed + o)

        def dot(grid, dx, dy):
            return grid[..., 0] * (xf[None, :] + dx) + grid[..., 1] * (yf[:, None] + dy)

        n00, n10 = dot(g, 0, 0), dot(g_x1, -1, 0)
        n01, n11 = dot(g_y1, 0, -1), dot(g_xy, -1, -1)
        u, v = _fade(xf)[None, :], _fade(yf)[:, None]
        out += amp * ((n00 * (1 - u) + n10 * u) * (1 - v)
                      + (n01 * (1 - u) + n11 * u) * v)
        total_amp += amp
        amp *= persistence
        freq *= lacunarity
    out /= max(total_amp, 1e-9)
    return np.clip(out, -1.0, 1.0)


# ---- 高程图地形 ----

def _hfield_geom(spec: mujoco.MjSpec, name: str, pos, euler_deg) -> None:
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=name,
                            pos=list(pos), quat=_quat_deg(euler_deg))


def _register_hfield(spec: mujoco.MjSpec, name: str, gray: np.ndarray,
                     hsize, pos, euler_deg, assets: dict) -> None:
    """灰度图 -> hfield 资产 + geom;PNG 以字节进 assets,不写盘。"""
    ok, buf = cv2.imencode(".png", gray)
    if not ok:
        raise RuntimeError("高程图 PNG 编码失败")
    assets[f"{name}.png"] = buf.tobytes()
    spec.add_hfield(name=name, size=list(hsize), file=f"{name}.png")
    _hfield_geom(spec, name, pos, euler_deg)


def _add_perlin_hfield(spec: mujoco.MjSpec, p: dict, assets: dict) -> None:
    name = p.get("name", "terrain_perlin")
    w = int(p.get("image_width", 128))
    h = int(p.get("image_height", p.get("img_height", w)))
    noise = perlin2d(w, h, scale=p.get("smooth", 100.0),
                     octaves=int(p.get("octaves", p.get("perlin_octaves", 6))),
                     persistence=p.get("persistence", p.get("perlin_persistence", 0.5)),
                     lacunarity=p.get("lacunarity", p.get("perlin_lacunarity", 2.0)),
                     seed=int(p.get("seed", 0)))
    gray = ((noise + 1) / 2 * 255).astype(np.uint8)
    size = list(p.get("size", [2.0, 1.5]))
    hsize = [size[0], size[1], p.get("height_scale", 0.2), p.get("negative_height", 0.2)]
    _register_hfield(spec, name, gray, hsize,
                     p.get("pos", [-1.5, 4.0, 0.0]), p.get("euler", [0, 0, 0]), assets)


def _add_image_hfield(spec: mujoco.MjSpec, p: dict, assets: dict) -> None:
    name = p.get("name", "terrain_image")
    img_path = Path(p["input_img"])
    if not img_path.is_absolute():
        img_path = Path(__file__).resolve().parent.parent / img_path
    img = imread_u(str(img_path))
    if img is None:
        raise FileNotFoundError(f"image_hfield 读不到图片: {img_path}")
    sy, sx = p.get("image_scale", [1.0, 1.0])
    img = cv2.resize(img, (int(img.shape[1] * sx), int(img.shape[0] * sy)),
                     interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if p.get("invert_gray", False):
        gray = 255 - gray
    size = list(p.get("size", [2.0, 2.0]))
    hsize = [size[0], size[1], p.get("height_scale", 0.02), p.get("negative_height", 0.1)]
    _register_hfield(spec, name, gray, hsize,
                     p.get("pos", [-1.5, 4.0, 0.0]), p.get("euler", [0, 0, 0]), assets)


# ---------------------------------------------------------------- 入口

_BUILDERS = {
    "box": lambda spec, p, a: _add_box(spec, p),
    "stairs": lambda spec, p, a: _add_stairs(spec, p),
    "suspend_stairs": lambda spec, p, a: _add_suspend_stairs(spec, p),
    "rough_ground": lambda spec, p, a: _add_rough_ground(spec, p),
    "perlin_hfield": _add_perlin_hfield,
    "image_hfield": _add_image_hfield,
}


def add_terrain(spec: mujoco.MjSpec, feature: dict) -> dict[str, bytes]:
    """按配置字典往场景 spec 加一个地形元素,返回需要注册的资源(PNG 字节)。"""
    ftype = feature.get("type")
    builder = _BUILDERS.get(ftype)
    if builder is None:
        raise ValueError(f"未知地形类型 '{ftype}',可选: {sorted(_BUILDERS)}")
    assets: dict[str, bytes] = {}
    builder(spec, feature, assets)
    return assets
