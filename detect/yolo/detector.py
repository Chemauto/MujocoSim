"""detector.py —— YOLOv8s 检测器（ultralytics）。

权重、置信度来自构造参数；默认权重 detect/yolo/yolov8s.pt（COCO 80 类）。
设备自动选择 CUDA，GPU 无可用 kernel（如 sm_120）自动回退 CPU。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_YOLO_DIR = Path(__file__).resolve().parent


def resolve_weights(weights: str = "yolov8s.pt") -> str:
    """权重路径解析：绝对路径原样；单文件名若 detect/yolo/ 下不存在则交给
    ultralytics 自动下载；其余相对路径按 detect/yolo/ 解析。"""
    p = Path(weights)
    if p.is_absolute():
        return str(p)
    if len(p.parts) == 1 and not (_YOLO_DIR / p).exists():
        return weights
    return str(_YOLO_DIR / p)


@dataclass
class Detection:
    name: str
    conf: float
    xyxy: tuple[float, float, float, float]  # 像素坐标 x1, y1, x2, y2
    distance: float | None = None  # 到相机直线距离（米），measure 后填
    xyz_cam: tuple[float, float, float] | None = None  # 相机系坐标（米）
    xyz_world: tuple[float, float, float] | None = None  # 世界系坐标（米）


class YoloDetector:
    def __init__(
        self,
        weights: str = "yolov8s.pt",
        conf: float = 0.25,
        imgsz: int = 640,
        device: str | None = None,
        classes: list[str] | None = None,
    ):
        import torch
        from ultralytics import YOLO

        self.weights = resolve_weights(weights)
        self.model = YOLO(self.weights)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                try:
                    self.model.to("cuda")
                    self.model.predict(
                        np.zeros((64, 64, 3), np.uint8),
                        imgsz=64,
                        verbose=False,
                        device="cuda",
                    )
                except Exception:
                    device = "cpu"
        self.device = device
        self.model.to(self.device)
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.names: dict[int, str] = dict(self.model.names or {})
        self.class_filter = set(classes) if classes else None

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        """RGB uint8 图 -> Detection 列表（内部转 BGR，ultralytics 约定）。"""
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        result = self.model.predict(
            bgr, conf=self.conf, imgsz=self.imgsz, verbose=False
        )[0]
        out: list[Detection] = []
        for b in result.boxes or []:
            name = self.names[int(b.cls)]
            if self.class_filter and name not in self.class_filter:
                continue
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            out.append(
                Detection(
                    name=name,
                    conf=float(b.conf),
                    xyxy=(x1, y1, x2, y2),
                )
            )
        return out


def annotate(rgb: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """RGB -> 画好框和标签的 RGB（纯 OpenCV，无检测器依赖）。"""
    import cv2

    vis = rgb[..., ::-1].copy()
    for det in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in det.xyxy)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{det.name} {det.conf:.2f}"
        if det.distance is not None:
            label += f"  R={det.distance:.2f}m"
        cv2.putText(
            vis,
            label,
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )
    return vis[..., ::-1]
