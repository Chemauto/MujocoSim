from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Empty, String

from mujocosim.ros2_bridge import topic_name
from simutils.Joystick import make_joystick

BUTTON_MODES = {
    "a": "damping",
    "b": "pd_stand",
    "y": "motion",
    "x": "position",
}


class JoystickNode(Node):
    def __init__(self, namespace: str):
        super().__init__("mujocosim_joystick")
        self.cmd_pub = self.create_publisher(
            Twist, topic_name(namespace, "cmd_vel"), 10
        )
        self.mode_pub = self.create_publisher(
            String, topic_name(namespace, "control_mode"), 10
        )
        self.reset_pub = self.create_publisher(
            Empty, topic_name(namespace, "reset"), 10
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="手柄 → /cmd_vel（虚拟 Xbox 或真实手柄）"
    )
    parser.add_argument(
        "--kind",
        default="sim",
        choices=["sim", "xbox"],
        help="sim=虚拟手柄窗口, xbox=真实手柄",
    )
    parser.add_argument("--namespace", default="", help="ROS2 命名空间（默认根）")
    parser.add_argument("--vx-max", type=float, default=1.0, help="前后最大速度 m/s")
    parser.add_argument("--vy-max", type=float, default=0.5, help="左右最大速度 m/s")
    parser.add_argument("--yaw-max", type=float, default=1.0, help="最大转向 rad/s")
    args = parser.parse_args()

    js = make_joystick(args.kind)
    rclpy.init()
    node = JoystickNode(args.namespace)
    print(
        "[joystick] 运行中：左摇杆=平移，右摇杆X=转向；A=急停 B=站立 Y=策略 X=位控 start=重置"
    )

    prev_buttons: dict[str, bool] = {}
    try:
        while rclpy.ok():
            js.update()
            state = js.get_state()
            twist = Twist()
            twist.linear.x = -state.axes["left_y"] * args.vx_max
            twist.linear.y = -state.axes["left_x"] * args.vy_max
            twist.angular.z = -state.axes["right_x"] * args.yaw_max
            node.cmd_pub.publish(twist)

            for button, mode in BUTTON_MODES.items():
                if state.buttons.get(button) and not prev_buttons.get(button):
                    msg = String()
                    msg.data = mode
                    node.mode_pub.publish(msg)
                    print(f"[joystick] {button} -> control_mode: {mode}")
            if state.buttons.get("start") and not prev_buttons.get("start"):
                node.reset_pub.publish(Empty())
                print("[joystick] start -> reset")
            prev_buttons = dict(state.buttons)
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        js.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
