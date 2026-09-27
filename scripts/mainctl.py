from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, String

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from mujocosim.config import load_config
from mujocosim.ros2_bridge import topic_name

MODES = [
    ("damping", "阻尼/无力"),
    ("pd_stand", "PD 站立"),
    ("position", "位控"),
    ("motion", "策略运动"),
]


def _slider(lo: float, hi: float) -> QSlider:
    slider = QSlider(Qt.Horizontal)
    steps = 200
    slider.setRange(-steps, steps)
    slider.setValue(0)
    slider.setProperty("hi", hi)
    return slider


def _slider_value(slider: QSlider) -> float:
    return slider.value() / slider.maximum() * float(slider.property("hi"))


class ControlNode(Node):
    def __init__(self, namespace: str, joint_names: list[str]):
        super().__init__("mujocosim_mainctl")
        self.mode_pub = self.create_publisher(
            String, topic_name(namespace, "control_mode"), 10
        )
        self.cmd_pub = self.create_publisher(
            Twist, topic_name(namespace, "cmd_vel"), 10
        )
        self.joint_pub = self.create_publisher(
            JointState, topic_name(namespace, "joint_command"), 10
        )
        self.reset_pub = self.create_publisher(
            Empty, topic_name(namespace, "reset"), 10
        )
        self.joint_names = joint_names
        self.latest_joints: JointState | None = None
        self.latest_odom: Odometry | None = None
        self.create_subscription(
            JointState, topic_name(namespace, "joint_states"), self._on_joints, 10
        )
        self.create_subscription(
            Odometry, topic_name(namespace, "odom"), self._on_odom, 10
        )

    def _on_joints(self, msg: JointState) -> None:
        self.latest_joints = msg

    def _on_odom(self, msg: Odometry) -> None:
        self.latest_odom = msg


class MainCtl(QMainWindow):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.node = ControlNode(cfg.ros2.namespace, cfg.robot_meta.joint_names)
        self._spin = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self._spin.start()
        self.setWindowTitle(f"MujocoSim 控制器 · {cfg.robot}")
        self._build_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(200)

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)

        mode_box = QGroupBox("控制模式（状态机）")
        mode_layout = QHBoxLayout(mode_box)
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        for mode, label in MODES:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.clicked.connect(lambda _=False, m=mode: self._set_mode(m))
            self._mode_group.addButton(btn)
            mode_layout.addWidget(btn)
        estop = QPushButton("急停")
        estop.setStyleSheet("background-color: #b33; color: white; font-weight: bold;")
        estop.clicked.connect(lambda: self._set_mode("damping", estop=True))
        mode_layout.addWidget(estop)
        layout.addWidget(mode_box)

        cmd_box = QGroupBox("速度指令（vx / vy / yaw_rate）")
        cmd_layout = QGridLayout(cmd_box)
        self._cmd_sliders = []
        for i, name in enumerate(["vx", "vy", "yaw_rate"]):
            cmd_layout.addWidget(QLabel(name), i, 0)
            slider = _slider(-1.0, 1.0)
            slider.valueChanged.connect(self._send_command)
            cmd_layout.addWidget(slider, i, 1)
            value = QLabel("0.00")
            cmd_layout.addWidget(value, i, 2)
            self._cmd_sliders.append((slider, value))
        layout.addWidget(cmd_box)

        joint_box = QGroupBox("关节目标（position 模式）")
        joint_layout = QGridLayout(joint_box)
        self._joint_sliders = []
        for i, name in enumerate(self.cfg.robot_meta.joint_names):
            joint_layout.addWidget(QLabel(name), i, 0)
            slider = _slider(-3.14, 3.14)
            default = self.cfg.robot_meta.default_joint_pos[i]
            slider.setValue(int(default / 3.14 * slider.maximum()))
            joint_layout.addWidget(slider, i, 1)
            value = QLabel(f"{default:.2f}")
            joint_layout.addWidget(value, i, 2)
            self._joint_sliders.append((slider, value))
        apply_btn = QPushButton("下发关节目标")
        apply_btn.clicked.connect(self._apply_joints)
        joint_layout.addWidget(apply_btn, len(self._joint_sliders), 0, 1, 3)
        scroll = QScrollArea()
        scroll.setWidget(joint_box)
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(260)
        layout.addWidget(scroll)

        ctrl_layout = QHBoxLayout()
        reset_btn = QPushButton("重置场景")
        reset_btn.clicked.connect(self._reset)
        ctrl_layout.addWidget(reset_btn)
        layout.addLayout(ctrl_layout)

        self._status = QLabel("等待仿真器…")
        layout.addWidget(self._status)
        root.setLayout(layout)
        self.setCentralWidget(root)
        self.resize(560, 720)

    def _set_mode(self, mode: str, estop: bool = False) -> None:
        msg = String()
        msg.data = mode
        self.node.mode_pub.publish(msg)
        if estop:
            for btn in self._mode_group.buttons():
                btn.setChecked(False)

    def _send_command(self) -> None:
        msg = Twist()
        values = []
        for slider, label in self._cmd_sliders:
            v = _slider_value(slider)
            values.append(v)
            label.setText(f"{v:.2f}")
        msg.linear.x = values[0]
        msg.linear.y = values[1]
        msg.angular.z = values[2]
        self.node.cmd_pub.publish(msg)

    def _apply_joints(self) -> None:
        msg = JointState()
        msg.name = self.cfg.robot_meta.joint_names
        msg.position = [_slider_value(s) for s, _ in self._joint_sliders]
        self.node.joint_pub.publish(msg)
        self._set_mode("position")

    def _reset(self) -> None:
        self.node.reset_pub.publish(Empty())

    def _refresh(self) -> None:
        js = self.node.latest_joints
        odom = self.node.latest_odom
        if odom is None and js is None:
            self._status.setText("等待仿真器…（先运行 python3 sim_entry.py）")
            return
        lines = []
        if odom is not None:
            p = odom.pose.pose.position
            lines.append(f"base: [{p.x:.2f}, {p.y:.2f}, {p.z:.2f}]")
        if js is not None:
            q = js.position[:3] if js.position else []
            rounded = ", ".join(f"{v:.2f}" for v in q)
            lines.append(f"joint_states 收到 {len(js.position)} 关节（前3: {rounded}）")
        self._status.setText("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="MujocoSim 控制器（Qt 面板）")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "config.yaml"))
    parser.add_argument("--robot", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, robot=args.robot)
    rclpy.init()
    app = QApplication(sys.argv)
    window = MainCtl(cfg)
    window.show()
    app.exec()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
