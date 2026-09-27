from __future__ import annotations

from pathlib import Path

import numpy as np


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_inv(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]]) / max(float(np.dot(q, q)), 1e-12)


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qw = q[0]
    qv = q[1:4]
    a = v * (2.0 * qw * qw - 1.0)
    b = np.cross(qv, v) * qw * 2.0
    c = qv * np.dot(qv, v) * 2.0
    return a - b + c


def matrix_from_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def subtract_frame_transforms(t01, q01, t02, q02):
    q_12 = quat_mul(quat_inv(q01), q02)
    t_12 = quat_rotate_inverse(q01, np.asarray(t02) - np.asarray(t01))
    return t_12, q_12


def yaw_quat(q: np.ndarray) -> np.ndarray:
    yaw = np.arctan2(
        2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3])
    )
    half = yaw / 2.0
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


class MotionClip:
    def __init__(self, path: Path, anchor_body: str | int = 0):
        data = np.load(path)
        self.joint_pos = np.asarray(data["joint_pos"], dtype=float)
        self.joint_vel = np.asarray(data["joint_vel"], dtype=float)
        self.body_pos_w = np.asarray(data["body_pos_w"], dtype=float)
        self.body_quat_w = np.asarray(data["body_quat_w"], dtype=float)
        self.fps = (
            float(np.asarray(data["fps"]).reshape(-1)[0])
            if "fps" in data.files
            else 30.0
        )
        if "body_names" in data.files:
            names = [str(n) for n in data["body_names"]]
            self.anchor = (
                names.index(str(anchor_body))
                if isinstance(anchor_body, str)
                else int(anchor_body)
            )
        else:
            self.anchor = int(anchor_body) if not isinstance(anchor_body, str) else 0
        self.frame = 0

    @property
    def num_frames(self) -> int:
        return len(self.joint_pos)

    @property
    def num_joints(self) -> int:
        return self.joint_pos.shape[1]

    def reset(self) -> None:
        self.frame = 0

    def advance(self) -> None:
        self.frame = (self.frame + 1) % self.num_frames

    def commands(self) -> tuple[np.ndarray, np.ndarray]:
        return self.joint_pos[self.frame].copy(), self.joint_vel[self.frame].copy()

    def anchor_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.body_pos_w[self.frame, self.anchor].copy(), self.body_quat_w[
            self.frame, self.anchor
        ].copy()


class MotionContext:
    def __init__(self, clip: MotionClip):
        self.clip = clip
        self.init_yaw_inv = np.array([1.0, 0.0, 0.0, 0.0])

    def reset(self) -> None:
        self.clip.reset()
        _, anchor_quat = self.clip.anchor_pose()
        self.init_yaw_inv = quat_inv(yaw_quat(anchor_quat))
