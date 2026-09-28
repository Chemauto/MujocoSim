from __future__ import annotations

import argparse
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from detect.yolo import YoloDetector, annotate, measure_detections

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QGridLayout,
    QGroupBox,
    QLabel,
    QMainWindow,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

TOPIC_RE = re.compile(
    r"^(?:/(?P<ns>[^/]+))?/camera/(?P<cam>[^/]+)/(?P<kind>rgb|depth)/image_raw$"
)
INFO_RE = re.compile(r"^(?:/(?P<ns>[^/]+))?/camera/(?P<cam>[^/]+)/rgb/camera_info$")


@dataclass
class Stream:
    cam: str
    kind: str
    frame: np.ndarray | None = None
    seq: int = 0
    last_time: float = 0.0
    active: bool = False
    label: QLabel | None = field(default=None, repr=False)
    dets: list = field(default_factory=list, repr=False)
    dets_seq: int = -1


class YoloWorker(threading.Thread):
    """后台检测线程：对每路 RGB 新帧跑 YOLO，结果写回 Stream.dets。"""

    def __init__(self, node: "CameraMonitor", detector: YoloDetector, interval: float):
        super().__init__(daemon=True)
        self.node = node
        self.detector = detector
        self.interval = interval
        self._done: dict[tuple[str, str], int] = {}

    def run(self) -> None:
        while True:
            time.sleep(self.interval)
            for key, s in self.node.snapshot().items():
                if s.kind != "rgb" or s.frame is None or not s.active:
                    continue
                if self._done.get(key) == s.seq:
                    continue
                self._done[key] = s.seq
                frame = s.frame
                try:
                    dets = self.detector.detect(frame)
                    depth = self.node.frame_of(s.cam, "depth")
                    k = self.node.intrinsics(s.cam)
                    if depth is not None and k is not None:
                        measure_detections(dets, depth.astype(np.float32), k)
                    s.dets = dets
                    s.dets_seq = s.seq
                except Exception as e:
                    print(f"[camera_viewer] yolo 检测失败({s.cam}): {e!r}")


def image_to_np(msg: Image) -> np.ndarray | None:
    if msg.encoding in ("rgb8", "8UC3"):
        return np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
    if msg.encoding == "bgr8":
        return np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)[
            :, :, ::-1
        ]
    if msg.encoding in ("32FC1",):
        return np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width)
    if msg.encoding in ("16UC1",):
        return np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.width)
    return None


def depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    import cv2

    valid = np.isfinite(depth) & (depth > 0)
    out = np.full((*depth.shape, 3), 30, dtype=np.uint8)
    if valid.any():
        lo = float(np.percentile(depth[valid], 2))
        hi = float(np.percentile(depth[valid], 98))
        norm = np.zeros_like(depth, dtype=np.float32)
        norm[valid] = np.clip((depth[valid] - lo) / max(hi - lo, 1e-6), 0, 1)
        near_hot = ((1.0 - norm) * 255).astype(np.uint8)
        out = cv2.applyColorMap(near_hot, cv2.COLORMAP_TURBO)[:, :, ::-1]
        out[~valid] = 30
    return np.ascontiguousarray(out)


class CameraMonitor(Node):
    def __init__(self, namespace: str):
        super().__init__("mujocosim_camera_viewer")
        self.ns = namespace
        self.streams: dict[tuple[str, str], Stream] = {}
        self._subs: dict[str, object] = {}
        self._k: dict[str, dict] = {}
        self._lock = threading.Lock()
        self.create_timer(1.0, self._discover)

    def frame_of(self, cam: str, kind: str) -> np.ndarray | None:
        with self._lock:
            s = self.streams.get((cam, kind))
            return None if s is None else s.frame

    def intrinsics(self, cam: str) -> dict | None:
        with self._lock:
            return self._k.get(cam)

    def _discover(self) -> None:
        names = {
            t
            for t, types in self.get_topic_names_and_types()
            if "sensor_msgs/msg/Image" in types
        }
        for topic in names:
            m = TOPIC_RE.match(topic)
            if not m:
                continue
            if self.ns and m.group("ns") != self.ns:
                continue
            if topic in self._subs:
                continue
            self._subs[topic] = self.create_subscription(
                Image, topic, self._make_cb(m.group("cam"), m.group("kind")), 10
            )
            key = (m.group("cam"), m.group("kind"))
            with self._lock:
                self.streams.setdefault(
                    key, Stream(cam=m.group("cam"), kind=m.group("kind"))
                )
        info_names = {
            t
            for t, types in self.get_topic_names_and_types()
            if "sensor_msgs/msg/CameraInfo" in types
        }
        for topic in info_names:
            m = INFO_RE.match(topic)
            if not m or topic in self._subs:
                continue
            if self.ns and m.group("ns") != self.ns:
                continue
            self._subs[topic] = self.create_subscription(
                CameraInfo, topic, self._make_info_cb(m.group("cam")), 10
            )
        for key, stream in self.streams.items():
            stream.active = self.count_publishers(self._topic_for(key)) > 0

    def _topic_for(self, key: tuple[str, str]) -> str:
        cam, kind = key
        prefix = f"/{self.ns}" if self.ns else ""
        return f"{prefix}/camera/{cam}/{kind}/image_raw"

    def _make_cb(self, cam: str, kind: str):
        def cb(msg: Image) -> None:
            frame = image_to_np(msg)
            if frame is None:
                return
            with self._lock:
                s = self.streams.get((cam, kind))
                if s is not None:
                    s.frame = frame
                    s.seq += 1
                    s.last_time = time.time()
                    s.active = True

        return cb

    def _make_info_cb(self, cam: str):
        def cb(msg) -> None:
            k = msg.k
            with self._lock:
                self._k[cam] = {
                    "fx": float(k[0]),
                    "fy": float(k[4]),
                    "cx": float(k[2]),
                    "cy": float(k[5]),
                }

        return cb

    def snapshot(self) -> dict[tuple[str, str], Stream]:
        with self._lock:
            return dict(self.streams)


class CameraWindow(QMainWindow):
    def __init__(self, node: CameraMonitor):
        super().__init__()
        self.node = node
        self.setWindowTitle("MujocoSim 相机监视")
        self._panels: dict[tuple[str, str], QGroupBox] = {}
        self._imgs: dict[tuple[str, str], QLabel] = {}
        central = QWidget()
        self._layout = QVBoxLayout(central)
        self._status = QLabel("等待相机话题…（启动仿真器后自动接入）")
        self._layout.addWidget(self._status)
        scroll = QScrollArea()
        scroll.setWidget(central)
        scroll.setWidgetResizable(True)
        self.setCentralWidget(scroll)
        self.resize(720, 640)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(33)

    def _ensure_panel(self, key: tuple[str, str]) -> None:
        if key in self._imgs:
            return
        cam, kind = key
        box = QGroupBox(f"{cam} · {'RGB' if kind == 'rgb' else '深度'}")
        lay = QVBoxLayout(box)
        img = QLabel("…")
        img.setMinimumSize(320, 240)
        img.setAlignment(Qt.AlignCenter)
        lay.addWidget(img)
        grid_index = len(self._imgs)
        self._layout.addWidget(box)
        self._panels[key] = box
        self._imgs[key] = img
        self._status.setVisible(len(self._imgs) == 0)

    def _refresh(self) -> None:
        streams = self.node.snapshot()
        live = 0
        for key, s in streams.items():
            if s.frame is None:
                continue
            if s.last_time and time.time() - s.last_time > 3.0:
                s.active = False
            if not s.active:
                continue
            live += 1
            self._ensure_panel(key)
            frame = s.frame
            if s.kind == "depth":
                frame = (
                    depth_to_rgb(frame.astype(np.float32))
                    if frame.dtype != np.uint8
                    else frame
                )
            elif s.dets:
                frame = annotate(frame, s.dets)
            h, w = frame.shape[:2]
            qimg = QImage(
                np.ascontiguousarray(frame).tobytes(), w, h, w * 3, QImage.Format_RGB888
            ).copy()
            self._imgs[key].setPixmap(QPixmap.fromImage(qimg))
        want = live > 0
        if want != self.isVisible():
            self.setVisible(want)
            print(
                f"[camera_viewer] {'打开画面（检测到相机话题）' if want else '窗口隐藏（无相机话题）'}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="相机画面实时监视（自动检测话题并开窗，含 YOLO 检测框）"
    )
    parser.add_argument("--namespace", default="", help="ROS2 命名空间（默认根）")
    parser.add_argument("--no-yolo", action="store_true", help="不运行 YOLO 检测")
    parser.add_argument(
        "--yolo-weights",
        default="yolov8s.pt",
        help="YOLO 权重（默认 detect/yolo/yolov8s.pt）",
    )
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument(
        "--yolo-classes", default="", help="只保留这些类别（逗号分隔，空 = 全部）"
    )
    parser.add_argument(
        "--yolo-interval", type=float, default=0.1, help="检测间隔（秒）"
    )
    args = parser.parse_args()

    detector = None
    if not args.no_yolo:
        try:
            classes = [c.strip() for c in args.yolo_classes.split(",") if c.strip()]
            detector = YoloDetector(
                args.yolo_weights, conf=args.yolo_conf, classes=classes or None
            )
            print(
                f"[camera_viewer] yolo 权重={detector.weights} 设备={detector.device}"
            )
        except Exception as e:
            print(f"[camera_viewer] yolo 初始化失败({e!r})，本次运行不启用检测")

    rclpy.init()
    node = CameraMonitor(args.namespace)
    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()
    if detector is not None:
        YoloWorker(node, detector, args.yolo_interval).start()
    app = QApplication(sys.argv)
    win = CameraWindow(node)
    print("[camera_viewer] 监测中：无相机话题不显示窗口，检测到画面自动打开")
    app.exec()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
