"""measure.py —— 用深度 + 内参把检测框变成 3D 距离/坐标。

bbox 区域深度取中位数(抗边缘噪声),再按针孔模型反投影:
  - 距离 R:相机到物体的直线距离
  - xyz_cam:相机系坐标(x 右, y 下, z 前)
  - xyz_world:世界系坐标(用相机外参变换)
"""
from __future__ import annotations

import math

import numpy as np

from sensors.camera import SimCamera
from .detector import Detection


def _bbox_depth_median(depth: np.ndarray, xyxy, max_range: float = 20.0) -> float | None:
    h, w = depth.shape
    x1, y1 = max(0, int(round(xyxy[0]))), max(0, int(round(xyxy[1])))
    x2, y2 = min(w, int(round(xyxy[2]))), min(h, int(round(xyxy[3])))
    if x2 <= x1 or y2 <= y1:
        return None
    roi = depth[y1:y2, x1:x2]
    roi = roi[np.isfinite(roi) & (roi > 0) & (roi < max_range)]
    if roi.size == 0:
        return None
    return float(np.median(roi))


def measure_detections(detections: list[Detection], camera: SimCamera,
                       depth: np.ndarray | None = None) -> list[Detection]:
    """给每个 Detection 填上 distance / xyz_cam / xyz_world(就地修改并返回)。

    depth 不传就现场渲染一张。
    """
    if depth is None:
        depth = camera.render_depth()
    K = camera.intrinsics()
    for det in detections:
        z = _bbox_depth_median(depth, det.xyxy)
        if z is None:
            continue
        cx = (det.xyxy[0] + det.xyxy[2]) / 2.0
        cy = (det.xyxy[1] + det.xyxy[3]) / 2.0
        x = (cx - K["cx"]) * z / K["fx"]
        y = (cy - K["cy"]) * z / K["fy"]
        det.xyz_cam = (x, y, z)                    # OpenCV 惯例:z 向前为正
        det.distance = math.sqrt(x * x + y * y + z * z)
        det.xyz_world = tuple(float(v) for v in camera.cv_to_world([x, y, z])[0])
    return detections
