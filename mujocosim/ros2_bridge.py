from __future__ import annotations

from typing import Callable

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import (
    CameraInfo,
    Image,
    Imu,
    JointState,
    LaserScan,
    PointCloud2,
    PointField,
)
from std_msgs.msg import Empty, String

from .config import SimConfig
from .controller import RobotController
from .world import SimWorld


def topic_name(namespace: str, name: str) -> str:
    return f"/{namespace}/{name}" if namespace else f"/{name}"


def _image_msg(array: np.ndarray, stamp, frame_id: str, encoding: str) -> Image:
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = int(array.shape[0])
    msg.width = int(array.shape[1])
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = int(array.strides[0])
    msg.data = array.tobytes()
    return msg


def _camera_info_msg(
    width: int,
    height: int,
    intrinsics: tuple[float, float, float, float],
    stamp,
    frame_id: str,
) -> CameraInfo:
    fx, fy, cx, cy = intrinsics
    msg = CameraInfo()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.width = width
    msg.height = height
    msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    msg.distortion_model = "plumb_bob"
    msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    return msg


def _scan_msg(payload: dict, device, stamp) -> LaserScan:
    msg = LaserScan()
    msg.header.stamp = stamp
    msg.header.frame_id = payload["frame_id"]
    start, stop, step = device.h_angles
    msg.angle_min = float(np.radians(start))
    msg.angle_max = float(np.radians(stop))
    msg.angle_increment = float(np.radians(step))
    msg.time_increment = 0.0
    msg.scan_time = 1.0 / device.fps
    msg.range_min = 0.02
    msg.range_max = float(device.max_range)
    ranges = np.where(np.isfinite(payload["ranges"]), payload["ranges"], np.inf)
    msg.ranges = [float(v) for v in ranges]
    return msg


def _cloud_msg(points: np.ndarray, stamp, frame_id: str) -> PointCloud2:
    msg = PointCloud2()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = int(points.shape[0])
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 12 * msg.width
    msg.is_dense = True
    msg.data = np.ascontiguousarray(points, dtype=np.float32).tobytes()
    return msg


class SimRos2Bridge(Node):
    def __init__(
        self,
        cfg: SimConfig,
        world: SimWorld,
        controller: RobotController,
        on_cmd_vel: Callable[[np.ndarray], None] | None = None,
        suite=None,
    ):
        super().__init__("mujocosim_sim")
        self.cfg = cfg
        self.world = world
        self.controller = controller
        self.on_cmd_vel = on_cmd_vel
        self.suite = suite
        ns = cfg.ros2.namespace
        names = cfg.robot_meta.joint_names

        self.create_subscription(
            String, topic_name(ns, "control_mode"), self._on_mode, 10
        )
        self.create_subscription(Twist, topic_name(ns, "cmd_vel"), self._on_cmd_vel, 10)
        self.create_subscription(
            JointState, topic_name(ns, "joint_command"), self._on_joint_command, 10
        )
        self.create_subscription(Empty, topic_name(ns, "reset"), self._on_reset, 10)

        self._joint_pub = self.create_publisher(
            JointState, topic_name(ns, "joint_states"), 10
        )
        self._odom_pub = self.create_publisher(Odometry, topic_name(ns, "odom"), 10)
        self._imu_pub = self.create_publisher(Imu, topic_name(ns, "imu"), 10)
        self._joint_names = list(names)
        self._sensor_pubs = self._make_sensor_publishers(ns)
        self.create_timer(1.0 / 50.0, self._publish)

    def _make_sensor_publishers(self, ns: str) -> dict:
        pubs: dict[str, dict] = {}
        from .config import CameraDevice, HeightmapDevice, LidarDevice

        for mount in self.cfg.mounts:
            device = self.cfg.devices[mount.device]
            kind = device.type
            if isinstance(device, CameraDevice):
                base = topic_name(ns, f"camera/{mount.name}")
                entry = {}
                if device.rgb:
                    entry["rgb"] = self.create_publisher(
                        Image, f"{base}/rgb/image_raw", 10
                    )
                    entry["info"] = self.create_publisher(
                        CameraInfo, f"{base}/rgb/camera_info", 10
                    )
                if device.depth:
                    entry["depth"] = self.create_publisher(
                        Image, f"{base}/depth/image_raw", 10
                    )
                pubs[mount.name] = {"kind": kind, "device": device, "pubs": entry}
            elif isinstance(device, LidarDevice):
                if device.lidar_type == "2d":
                    pub = self.create_publisher(LaserScan, topic_name(ns, "scan"), 10)
                else:
                    pub = self.create_publisher(
                        PointCloud2, topic_name(ns, "points"), 10
                    )
                pubs[mount.name] = {
                    "kind": kind,
                    "device": device,
                    "pubs": {"data": pub},
                }
            elif isinstance(device, HeightmapDevice):
                pub = self.create_publisher(
                    PointCloud2, topic_name(ns, "heightmap"), 10
                )
                pubs[mount.name] = {
                    "kind": kind,
                    "device": device,
                    "pubs": {"points": pub},
                }
        return pubs

    def _on_mode(self, msg: String) -> None:
        mode = msg.data.strip()
        try:
            self.controller.set_mode(mode)
        except ValueError as error:
            self.get_logger().warn(str(error))
            return
        if mode == "motion" and self.controller.action_fn is not None:
            gains = getattr(self.controller.action_fn, "kp", None)
            kd = getattr(self.controller.action_fn, "kd", None)
            if gains is not None and kd is not None:
                self.controller.set_gains(gains, kd)
        self.get_logger().info(f"mode -> {mode}")

    def _on_cmd_vel(self, msg: Twist) -> None:
        if self.on_cmd_vel is not None:
            self.on_cmd_vel(np.array([msg.linear.x, msg.linear.y, msg.angular.z]))

    def _on_joint_command(self, msg: JointState) -> None:
        if len(msg.position) != len(self._joint_names):
            self.get_logger().warn("joint_command 维度不符，忽略")
            return
        self.controller.apply_joint_target(msg.position)

    def _on_reset(self, _msg: Empty) -> None:
        self.world.reset()
        reset_fn = (
            getattr(self.controller.action_fn, "reset", None)
            if self.controller.action_fn
            else None
        )
        if reset_fn is not None:
            reset_fn()
        self.get_logger().info("reset")

    def _publish(self) -> None:
        stamp = self.get_clock().now().to_msg()
        q = self.world.joint_qpos()
        qd = self.world.joint_qvel()

        js = JointState()
        js.header.stamp = stamp
        js.name = self._joint_names
        js.position = [float(v) for v in q]
        js.velocity = [float(v) for v in qd]
        js.effort = [float(v) for v in self.world.data.ctrl]
        self._joint_pub.publish(js)

        pos, quat_wxyz = self.world.base_pose()
        lin, ang = self.world.base_vel()
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base"
        odom.pose.pose.position.x = float(pos[0])
        odom.pose.pose.position.y = float(pos[1])
        odom.pose.pose.position.z = float(pos[2])
        odom.pose.pose.orientation.w = float(quat_wxyz[0])
        odom.pose.pose.orientation.x = float(quat_wxyz[1])
        odom.pose.pose.orientation.y = float(quat_wxyz[2])
        odom.pose.pose.orientation.z = float(quat_wxyz[3])
        odom.twist.twist.linear.x = float(lin[0])
        odom.twist.twist.linear.y = float(lin[1])
        odom.twist.twist.linear.z = float(lin[2])
        odom.twist.twist.angular.x = float(ang[0])
        odom.twist.twist.angular.y = float(ang[1])
        odom.twist.twist.angular.z = float(ang[2])
        self._odom_pub.publish(odom)

        imu = Imu()
        imu.header.stamp = stamp
        imu.header.frame_id = "base"
        imu.orientation.w = float(quat_wxyz[0])
        imu.orientation.x = float(quat_wxyz[1])
        imu.orientation.y = float(quat_wxyz[2])
        imu.orientation.z = float(quat_wxyz[3])
        imu.angular_velocity.x = float(ang[0])
        imu.angular_velocity.y = float(ang[1])
        imu.angular_velocity.z = float(ang[2])
        self._imu_pub.publish(imu)
        self._publish_sensors(stamp)

    def _publish_sensors(self, stamp) -> None:
        if self.suite is None:
            return
        for name, payload in self.suite.drain().items():
            info = self._sensor_pubs.get(name)
            if info is None:
                continue
            pubs = info["pubs"]
            device = info["device"]
            if info["kind"] == "camera":
                frame = payload["frame_id"]
                if "rgb" in pubs and "rgb" in payload:
                    pubs["rgb"].publish(
                        _image_msg(
                            np.ascontiguousarray(payload["rgb"]), stamp, frame, "rgb8"
                        )
                    )
                    w, h = payload["rgb"].shape[1], payload["rgb"].shape[0]
                    pubs["info"].publish(
                        _camera_info_msg(w, h, payload["intrinsics"], stamp, frame)
                    )
                if "depth" in pubs and "depth" in payload:
                    depth = np.ascontiguousarray(payload["depth"], dtype=np.float32)
                    pubs["depth"].publish(_image_msg(depth, stamp, frame, "32FC1"))
            elif info["kind"] == "lidar":
                if "ranges" in payload:
                    pubs["data"].publish(_scan_msg(payload, device, stamp))
                else:
                    pubs["data"].publish(
                        _cloud_msg(payload["points"], stamp, payload["frame_id"])
                    )
            elif info["kind"] == "heightmap":
                pubs["points"].publish(
                    _cloud_msg(payload["points"], stamp, payload["frame_id"])
                )
