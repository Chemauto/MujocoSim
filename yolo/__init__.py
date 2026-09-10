"""yolo —— 感知:YOLO-World 开放词表检测 + 深度测距(见 docs/yolo.md)。"""
from .detector import Detection, YoloWorldDetector, resolve_weights
from .measure import measure_detections

__all__ = ["YoloWorldDetector", "Detection", "resolve_weights", "measure_detections"]
