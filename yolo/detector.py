"""detector.py —— YOLO-World 开放词表检测器。

权重、类别、置信度全部来自配置;模型在第一次 detect 时才加载,
不用 YOLO 时零开销。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    name: str
    conf: float
    xyxy: tuple[float, float, float, float]        # 像素坐标 x1, y1, x2, y2
    distance: float | None = None                  # 到相机直线距离(米),measure 后填
    xyz_cam: tuple[float, float, float] | None = None   # 相机系坐标(米)
    xyz_world: tuple[float, float, float] | None = None  # 世界系坐标(米)


class YoloWorldDetector:
    """ultralytics YOLOWorld 封装:set_classes 支持任意文本类别。"""

    def __init__(self, weights: str, classes: list[str], conf: float = 0.2):
        self.weights = weights
        self.classes = list(classes)
        self.conf = float(conf)
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from ultralytics import YOLOWorld          # 延迟导入,不用 YOLO 不加载
            self._model = YOLOWorld(self.weights)
            if self.classes:
                self._model.set_classes(self.classes)  # 开放词表:类别随便定
        return self._model

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        """RGB uint8 图 -> Detection 列表(内部转 BGR,ultralytics 约定)。"""
        model = self._ensure_model()
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        result = model.predict(bgr, conf=self.conf, verbose=False)[0]
        out: list[Detection] = []
        for b in result.boxes:
            out.append(Detection(
                name=model.names[int(b.cls)],
                conf=float(b.conf),
                xyxy=tuple(float(v) for v in b.xyxy[0].tolist()),
            ))
        return out

    def annotate(self, rgb: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """RGB -> 画好框和距离标签的 RGB(无检测器依赖,纯 OpenCV)。"""
        import cv2
        vis = rgb[..., ::-1].copy()
        for det in detections:
            x1, y1, x2, y2 = (int(round(v)) for v in det.xyxy)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det.name} {det.conf:.2f}"
            if det.distance is not None:
                label += f"  R={det.distance:.2f}m"
            cv2.putText(vis, label, (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        return vis[..., ::-1]
