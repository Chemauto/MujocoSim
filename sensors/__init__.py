"""sensors —— 传感器(目前是相机):渲染 RGB/深度、内参、坐标换算。"""
from .camera import SimCamera, get_intrinsics, deproject_pixel, depth_colormap

__all__ = ["SimCamera", "get_intrinsics", "deproject_pixel", "depth_colormap"]
