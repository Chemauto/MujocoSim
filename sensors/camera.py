"""camera.py —— SimCamera:场景里任意一台相机的渲染与几何。

相机本体在 simulation/builder.py 里按配置挂进模型;这个类只负责:
  - 用配置的分辨率渲染 RGB / 深度
  - 计算针孔内参(K)
  - 提供相机外参(世界位姿),用于把相机系坐标换到世界系
分辨率/FOV/挂载位置全部来自配置,这里没有任何硬编码。
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import mujoco
import numpy as np


class SimCamera:
    """绑定 (model, data, 相机名) 的渲染器。一台相机一个 Renderer(分辨率独立)。"""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 name: str, width: int = 640, height: int = 480):
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if cam_id < 0:
            raise ValueError(f"模型里没有相机 '{name}',检查配置里的 cameras 段")
        self.name = name
        self.cam_id = cam_id
        self.model = model
        self.data = data
        self.width = width
        self.height = height
        self._renderer = mujoco.Renderer(model, height, width)

    # ---- 渲染 ----
    def render_rgb(self) -> np.ndarray:
        """RGB uint8 [H, W, 3]。"""
        self._renderer.update_scene(self.data, camera=self.cam_id)
        return self._renderer.render()

    def render_depth(self) -> np.ndarray:
        """深度 float [H, W],单位米;无效区域可能是 0/inf。"""
        self._renderer.update_scene(self.data, camera=self.cam_id)
        self._renderer.enable_depth_rendering()
        depth = self._renderer.render()
        self._renderer.disable_depth_rendering()
        return depth

    def render_rgbd(self) -> tuple[np.ndarray, np.ndarray]:
        self._renderer.update_scene(self.data, camera=self.cam_id)
        rgb = self._renderer.render()
        self._renderer.enable_depth_rendering()
        depth = self._renderer.render()
        self._renderer.disable_depth_rendering()
        return rgb, depth

    # ---- 几何 ----
    def intrinsics(self) -> dict[str, float]:
        """针孔内参 {fx, fy, cx, cy}(MuJoCo 由 fovy 推出,像素为方形)。"""
        return get_intrinsics(self.model, self.cam_id, self.width, self.height)

    def pose(self) -> tuple[np.ndarray, np.ndarray]:
        """相机当前世界位姿:(pos[3], R[3,3])(取自 mj_step/mj_forward 后的 cam_xpos/cam_xmat)。"""
        mujoco.mj_camlight(self.model, self.data)      # 确保相机位姿是最新
        pos = self.data.cam_xpos[self.cam_id].copy()
        R = self.data.cam_xmat[self.cam_id].reshape(3, 3).copy()
        return pos, R

    def cam_to_world(self, p_cam: np.ndarray) -> np.ndarray:
        """相机系点(N,3) -> 世界系(MuJoCo 原生约定:相机沿 -z 看)。"""
        pos, R = self.pose()
        return (np.atleast_2d(p_cam) @ R.T) + pos

    def cv_to_world(self, p_cv: np.ndarray) -> np.ndarray:
        """OpenCV 相机系点(N,3)(x 右, y 下, z 向前为正)-> 世界系。"""
        p_cv = np.atleast_2d(np.asarray(p_cv, float))
        p_mj = p_cv * np.array([1.0, -1.0, -1.0])   # OpenCV -> MuJoCo 相机系
        return self.cam_to_world(p_mj)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def get_intrinsics(model: mujoco.MjModel, cam_id: int, width: int, height: int) -> dict[str, float]:
    """针孔内参 {fx, fy, cx, cy}:MuJoCo 定义垂直视场角 fovy,fx=fy。"""
    fovy = math.radians(float(model.cam_fovy[cam_id]))
    f = (height / 2.0) / math.tan(fovy / 2.0)
    return {"fx": f, "fy": f, "cx": width / 2.0, "cy": height / 2.0}


def deproject_pixel(depth: np.ndarray, u: int, v: int, K: dict[str, float]) -> tuple[float, float, float] | None:
    """像素 (u, v) + 深度 -> 相机系 3D 点 (x, y, z),无效深度返回 None。"""
    if not (0 <= v < depth.shape[0] and 0 <= u < depth.shape[1]):
        return None
    z = float(depth[v, u])
    if not np.isfinite(z) or z <= 0:
        return None
    x = (u - K["cx"]) * z / K["fx"]
    y = (v - K["cy"]) * z / K["fy"]
    return (x, y, z)


def depth_colormap(depth: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """浮点深度(米)-> BGR 伪彩图,方便保存查看。"""
    d = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    vis = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.applyColorMap(vis, colormap)


# ---- 中文/非 ASCII 路径安全的图片读写 ----
# OpenCV 在 Windows 上对含中文的绝对路径行为不稳定(imread 直接失败),
# 这里用 Python 文件 IO + 编解码,任何路径都可靠。

def imread_u(path: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(buf, flags)


def imwrite_u(path: str, img: np.ndarray, params: list[int] | None = None) -> bool:
    ext = Path(path).suffix or ".png"
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        return False
    buf.tofile(path)                            # numpy 的 tofile 走 Python IO,中文路径没问题
    return True
