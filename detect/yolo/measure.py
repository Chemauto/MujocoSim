"""measure.py —— 用深度 + 内参把检测框变成 3D 距离/坐标。

bbox 中央 50% 区域深度取中位数（物体占框不足一半时全框中位数会偏向背景），
再按针孔模型反投影：
  - distance R：相机到物体的直线距离
  - xyz_cam：OpenCV 相机系坐标（x 右, y 下, z 前）
  - xyz_world：世界系坐标（需提供相机位姿；MuJoCo 相机系 p_cv = p_mj * (1,-1,-1)）
"""

from __future__ import annotations

import math

import numpy as np

from .detector import Detection


def bbox_depth_median(depth: np.ndarray, xyxy, max_range: float = 20.0) -> float | None:
    h, w = depth.shape
    x1, y1 = max(0, int(round(xyxy[0]))), max(0, int(round(xyxy[1])))
    x2, y2 = min(w, int(round(xyxy[2]))), min(h, int(round(xyxy[3])))
    if x2 <= x1 or y2 <= y1:
        return None
    hpad, wpad = (y2 - y1) // 4, (x2 - x1) // 4
    roi = depth[y1 + hpad : y2 - hpad, x1 + wpad : x2 - wpad]
    if roi.size == 0:
        roi = depth[y1:y2, x1:x2]
    roi = roi[np.isfinite(roi) & (roi > 0) & (roi < max_range)]
    if roi.size == 0:
        return None
    return float(np.median(roi))


def deproject(u: float, v: float, z: float, K: dict) -> np.ndarray:
    """像素 + 深度 -> OpenCV 相机系 3D（x 右, y 下, z 前）。"""
    return np.array([(u - K["cx"]) * z / K["fx"], (v - K["cy"]) * z / K["fy"], z])


def measure_detections(
    detections: list[Detection],
    depth: np.ndarray | None,
    K: dict,
    max_range: float = 20.0,
    cam_pos_w: np.ndarray | None = None,
    cam_R_mj: np.ndarray | None = None,
) -> list[Detection]:
    """给每个 Detection 填 distance / xyz_cam（就地修改并返回）。

    传 cam_pos_w（相机原点世界坐标）和 cam_R_mj（MuJoCo 相机轴的世界姿态，
    列向量为相机 x/y/z 轴）时同时填 xyz_world。
    """
    if depth is None:
        return detections
    for det in detections:
        z = bbox_depth_median(depth, det.xyxy, max_range)
        if z is None:
            continue
        cx = (det.xyxy[0] + det.xyxy[2]) / 2.0
        cy = (det.xyxy[1] + det.xyxy[3]) / 2.0
        p_cv = deproject(cx, cy, z, K)
        det.xyz_cam = (float(p_cv[0]), float(p_cv[1]), float(p_cv[2]))
        det.distance = math.sqrt(float(p_cv @ p_cv))
        if cam_pos_w is not None and cam_R_mj is not None:
            p_mj = p_cv * np.array([1.0, -1.0, -1.0])
            p_w = np.asarray(cam_pos_w) + np.asarray(cam_R_mj) @ p_mj
            det.xyz_world = (float(p_w[0]), float(p_w[1]), float(p_w[2]))
    return detections
