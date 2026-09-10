"""robot_api.py —— 机器人访问接口:rl_control 与仿真之间的唯一通道。

把 MuJoCo 细节(执行器表、qpos 地址、freejoint 角速度坐标系)全部封装在这里,
rl_control 层只面向"按名字取关节状态 / 按名字写力矩"的接口,不 import mujoco。

坐标系与单位约定:
  关节角 rad、关节速 rad/s、力矩 N·m;
  base 位姿为世界系,角速度为体系(MuJoCo freejoint 的 qvel 角速度在
  "父子正中帧",这里已用 mj_objectVelocity 转成真正的体系);
  projected_gravity = 重力单位向量(0,0,-1)在体系的投影。
"""
from __future__ import annotations

import mujoco
import numpy as np

from common.data_model import RobotState


class RobotAPI:
    """绑定 (model, data, base_body) 的机器人状态/执行器接口。"""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, base_body: str):
        self.model = model
        self.data = data
        bid = model.body(base_body).id
        self.base_body_id = bid
        self.base_adr = int(model.jnt_qposadr[model.body_jntadr[bid]])
        self._cache: dict[str, tuple[int, int, int]] = {}   # 关节名 -> (qadr, dofadr, act_idx)

    # ---- 关节(按名字) ----
    def joint_id(self, joint_name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            raise ValueError(f"机器人模型里没有关节 '{joint_name}'")
        return jid

    def joint_adr(self, joint_name: str) -> tuple[int, int, int]:
        """关节名 -> (qpos 地址, dof 地址, 执行器下标)。结果缓存。"""
        if joint_name not in self._cache:
            jid = self.joint_id(joint_name)
            hits = np.nonzero(self.model.actuator_trnid[:, 0] == jid)[0]
            if hits.size == 0:
                raise ValueError(f"关节 '{joint_name}' 没有对应的执行器(motor)")
            self._cache[joint_name] = (int(self.model.jnt_qposadr[jid]),
                                       int(self.model.jnt_dofadr[jid]),
                                       int(hits[0]))
        return self._cache[joint_name]

    def actuator_ids(self, joint_names: list[str]) -> np.ndarray:
        return np.array([self.joint_adr(n)[2] for n in joint_names], dtype=int)

    def joint_q(self, joint_names: list[str]) -> np.ndarray:
        return np.array([self.data.qpos[self.joint_adr(n)[0]] for n in joint_names])

    def joint_dq(self, joint_names: list[str]) -> np.ndarray:
        return np.array([self.data.qvel[self.joint_adr(n)[1]] for n in joint_names])

    def joint_range_rad(self, joint_name: str) -> tuple[float, float]:
        """关节限位(弧度,MuJoCo jnt_range 原值)。"""
        lo, hi = self.model.jnt_range[self.joint_id(joint_name)]
        return float(lo), float(hi)

    @property
    def sim_time(self) -> float:
        return self.data.time

    def body_ids(self) -> list[int]:
        """机器人全部 body id(从根 body 向下遍历),供高度扫描排除自身。"""
        out = [self.base_body_id]
        changed = True
        while changed:
            changed = False
            for i in range(self.model.nbody):
                if i in out:
                    continue
                if self.model.body_parentid[i] in out:
                    out.append(i)
                    changed = True
        return out

    def set_torques(self, actuator_ids: np.ndarray, torques) -> None:
        """把(与 actuator_ids 同序的)力矩写入 ctrl。"""
        self.data.ctrl[np.asarray(actuator_ids, dtype=int)] = torques

    # ---- base ----
    def base_pose(self) -> tuple[np.ndarray, np.ndarray]:
        adr = self.base_adr
        return (self.data.qpos[adr:adr + 3].copy(),
                self.data.qpos[adr + 3:adr + 7].copy())          # wxyz

    def base_attitude(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(体系角速度 rad/s, 重力投影, 旋转矩阵 R 世界<-体系)。"""
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data,
                                 mujoco.mjtObj.mjOBJ_BODY, self.base_body_id, vel, 1)
        quat = self.data.qpos[self.base_adr + 3:self.base_adr + 7].copy()
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, quat)
        R = R.reshape(3, 3)
        proj_grav = R.T @ np.array([0.0, 0.0, -1.0])
        return vel[0:3].copy(), proj_grav, R

    # ---- 复位 ----
    def reset(self, base_pos, base_quat=(1.0, 0.0, 0.0, 0.0),
              joint_names: list[str] | None = None, joint_pos=None) -> None:
        """整体复位:base 位姿 + 指定关节角(其余速度清零)。"""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.base_adr:self.base_adr + 3] = base_pos
        self.data.qpos[self.base_adr + 3:self.base_adr + 7] = base_quat
        if joint_names and joint_pos is not None:
            for name, q in zip(joint_names, joint_pos):
                qadr, _, _ = self.joint_adr(name)
                self.data.qpos[qadr] = q
        mujoco.mj_forward(self.model, self.data)

    # ---- 给数据总线用的快照 ----
    def robot_state(self, joint_names: list[str] | None = None,
                    tau=None) -> RobotState:
        """默认导出"执行器序"全关节;给 joint_names 则按该顺序(策略序)。"""
        if joint_names is None:
            joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                                            self.model.actuator_trnid[i, 0])
                           for i in range(self.model.nu)]
            q = np.array([self.data.qpos[self.joint_adr(n)[0]] for n in joint_names])
            dq = np.array([self.data.qvel[self.joint_adr(n)[1]] for n in joint_names])
        else:
            q, dq = self.joint_q(joint_names), self.joint_dq(joint_names)
        tau = np.zeros(len(joint_names)) if tau is None else np.asarray(tau, dtype=float)
        pos, quat = self.base_pose()
        ang_vel_b, proj_grav, _ = self.base_attitude()
        return RobotState(
            joint_names=list(joint_names),
            q=q.tolist(), dq=dq.tolist(), tau=np.asarray(tau).tolist(),
            base_pos=pos.tolist(), base_quat=quat.tolist(),
            base_ang_vel_b=ang_vel_b.tolist(),
            projected_gravity_b=proj_grav.tolist())
