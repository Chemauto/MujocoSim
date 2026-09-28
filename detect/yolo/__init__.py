"""detect.yolo —— YOLOv8s 目标检测 + 深度测距。"""

from .detector import Detection, YoloDetector, annotate, resolve_weights
from .measure import deproject, measure_detections

__all__ = [
    "YoloDetector",
    "Detection",
    "resolve_weights",
    "annotate",
    "measure_detections",
    "deproject",
]
