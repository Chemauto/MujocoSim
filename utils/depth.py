"""Depth utilities for the MuJoCo camera.

Mirrors astra_camera/depth_processor.py, but works on the float depth (meters)
returned by mujoco.Renderer instead of uint16-millimeter data from a real sensor.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def get_intrinsics(model, cam_id: int, width: int, height: int) -> dict:
    """Pinhole intrinsics {fx, fy, cx, cy} for a MuJoCo camera at WxH.

    MuJoCo defines the vertical field of view (fovy); pixels are square so
    fx = fy, and the principal point sits at the image center.
    """
    fovy = math.radians(float(model.cam_fovy[cam_id]))
    f = (height / 2.0) / math.tan(fovy / 2.0)
    return {"fx": f, "fy": f, "cx": width / 2.0, "cy": height / 2.0}


def deproject_pixel(depth: np.ndarray, u: int, v: int, K: dict):
    """3D point (x, y, z) in the camera frame [m] for pixel (u, v).

    z is the depth value (forward axis); x, y follow the pinhole model.
    Returns None if (u, v) is out of bounds or has invalid depth (<=0 / non-finite).
    """
    if not (0 <= v < depth.shape[0] and 0 <= u < depth.shape[1]):
        return None
    z = float(depth[v, u])
    if not np.isfinite(z) or z <= 0:
        return None
    x = (u - K["cx"]) * z / K["fx"]
    y = (v - K["cy"]) * z / K["fy"]
    return (x, y, z)


def pixel_distance(depth: np.ndarray, u: int, v: int, K: dict):
    """Euclidean distance [m] from the camera to the surface at pixel (u, v)."""
    p = deproject_pixel(depth, u, v, K)
    if p is None:
        return None
    x, y, z = p
    return math.sqrt(x * x + y * y + z * z)


def depth_to_colormap(depth: np.ndarray, colormap=int(cv2.COLORMAP_JET)) -> np.ndarray:
    """Float depth (m) -> BGR uint8 colormap, like depth_to_colormap in Carema."""
    d = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    vis = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.applyColorMap(vis, colormap)
