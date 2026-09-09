"""controller.py —— 控制层:多模式控制 + RL 行走策略接入。

模式(上位机 dashboard.py 可实时切换):
  damping   阻尼模式    tau = -kd * dq(关节软,机器人瘫软)
  force     力控模式    tau = 上位机滑条给定的力矩
  pd_stand  PD 站立     tau = kp*(q_default - q) - kd*dq,锁默认站姿
  position  位控模式    tau = kp*(q_des_ui - q) - kd*dq,滑条摆姿势
  policy    RL 行走     训练好的策略(TorchScript)50Hz 出目标角,PD 跟踪

策略接入完全对齐 llm-legged-lab 的部署链路(deploy.yaml + unitree_sdk2py_bridge):
  - 观测 232 维 = [ang_vel 3, projected_gravity 3, cmd 3,
                   joint_pos_rel 12, joint_vel_rel 12, last_action 12, height_scan 187]
  - 关节按"策略关节序"(joint_names 配置)排布,内部自动映射到 MuJoCo 执行器
  - 高度扫描 17x11 网格(1.6x1.0m @0.1),yaw 对齐,射线 +20m 向下,
    heights = base_z - hit_z - 0.5,geomgroup 只打 (1,1,0,0,0,0)
  - 动作:q_des = default + scale * clip(action, ±action_clip),50Hz(step_dt 0.02)
"""
from __future__ import annotations

import io
from pathlib import Path

import mujoco
import numpy as np

MODES = ["damping", "force", "pd_stand", "position", "policy"]
MODE_LABELS = {"damping": "阻尼模式", "force": "力控模式", "pd_stand": "PD站立",
               "position": "位控模式", "policy": "RL行走"}


class HeightScanner:
    """187 点高度扫描,逐行对照 unitree_sdk2py_bridge._compute_height_map 移植。"""

    def __init__(self, model: mujoco.MjModel, size=(1.6, 1.0), resolution: float = 0.1,
                 ray_offset_z: float = 20.0, height_offset: float = 0.5,
                 geomgroup=(1, 1, 0, 0, 0, 0), base_body: str = "base_link"):
        self.ray_offset_z = ray_offset_z
        self.height_offset = height_offset
        self.geomgroup = np.array(geomgroup, dtype=np.uint8)
        bid = model.body(base_body).id
        self.base_adr = model.jnt_qposadr[model.body_jntadr[bid]]

        x = np.arange(-size[0] / 2, size[0] / 2 + 1e-9, resolution)
        y = np.arange(-size[1] / 2, size[1] / 2 + 1e-9, resolution)
        gx, gy = np.meshgrid(x, y, indexing="xy")       # (11, 17),y 在外层
        self.grid = np.zeros((gx.size, 3))
        self.grid[:, 0] = gx.ravel()
        self.grid[:, 1] = gy.ravel()
        self.n_points = gx.size

    def compute(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        """返回 float32[n_points],没有命中高度的射线记 0。"""
        base_pos = data.qpos[self.base_adr:self.base_adr + 3]
        w, qx, qy, qz = data.qpos[self.base_adr + 3:self.base_adr + 7]
        yaw = np.arctan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

        starts = base_pos + self.grid @ rot.T
        starts[:, 2] += self.ray_offset_z
        dirs = np.zeros_like(starts)
        dirs[:, 2] = -1.0

        dist = np.full(self.n_points, -1.0)
        geomid = np.zeros(1, dtype=np.int32)
        for i in range(self.n_points):
            dist[i] = mujoco.mj_ray(model, data, starts[i], dirs[i],
                                    self.geomgroup, 1, -1, geomid)

        heights = np.zeros(self.n_points, dtype=np.float32)
        hit = dist > 0.0
        if hit.any():
            hit_z = starts[hit, 2] - dist[hit]          # 射线向下:命中 z = 起点 - 距离
            heights[hit] = (base_pos[2] - hit_z - self.height_offset).astype(np.float32)
        return heights


class RobotController:
    """按模式算力矩并写 data.ctrl。所有关节序对外都是"策略关节序"(cfg.joint_names)。"""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, cfg, base_body: str):
        self.model, self.data, self.cfg = model, data, cfg
        self.mode = "damping"

        # 策略关节序 -> MuJoCo (执行器下标, qposadr, dofadr)
        self.joint_names = list(cfg.joint_names)
        self.act_idx, self.qadr, self.dofadr = [], [], []
        for name in self.joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"控制器配置的关节 '{name}' 在机器人模型里不存在")
            aid = int(np.nonzero(model.actuator_trnid[:, 0] == jid)[0][0]) \
                if np.any(model.actuator_trnid[:, 0] == jid) else -1
            if aid < 0:
                raise ValueError(f"关节 '{name}' 没有对应的执行器(motor)")
            self.act_idx.append(aid)
            self.qadr.append(model.jnt_qposadr[jid])
            self.dofadr.append(model.jnt_dofadr[jid])
        self.act_idx = np.array(self.act_idx)
        self.qadr = np.array(self.qadr)
        self.dofadr = np.array(self.dofadr)

        bid = model.body(base_body).id
        self._base_bid = bid
        self.base_adr = model.jnt_qposadr[model.body_jntadr[bid]]
        self.base_dofadr = model.jnt_dofadr[model.body_jntadr[bid]]

        kp, kd = _expand(cfg.stiffness, len(self.joint_names)), _expand(cfg.damping, len(self.joint_names))
        self.kp = np.array(kp, dtype=float)
        self.kd = np.array(kd, dtype=float)
        self.q_default = np.array(cfg.default_joint_pos, dtype=float)   # 策略关节序
        self.q_des_ui = self.q_default.copy()          # 位控模式的目标(上位机改)
        self.tau_ui = np.zeros(len(self.joint_names))  # 力控模式的力矩(上位机改)
        self.commands = np.array(getattr(cfg, "commands", [0.0, 0.0, 0.0]), dtype=float)
        self.cmd_target = self.commands.copy()      # 上位机写这里,平滑后进观测

        # RL 策略
        self.policy = None
        self.last_action = np.zeros(len(self.joint_names))
        self.action_scale = np.array(_expand(cfg.action_scale, len(self.joint_names)), dtype=float)
        self.step_dt = float(cfg.step_dt)
        self._last_policy_t = -1e9
        self.q_des_policy = self.q_default.copy()       # 策略关节序
        if cfg.policy:
            self._load_policy(cfg.policy)
            self.scanner = HeightScanner(
                model, size=tuple(cfg.height_scan["size"]),
                resolution=cfg.height_scan["resolution"],
                ray_offset_z=cfg.height_scan["ray_offset_z"],
                height_offset=cfg.height_scan["height_offset"],
                base_body=base_body)
        else:
            self.scanner = None

    # ---- 坐标/顺序换算 ----
    def _to_mj(self, v_policy_order) -> np.ndarray:
        """策略关节序 -> MuJoCo 执行器序。"""
        out = np.zeros(self.model.nu)
        out[self.act_idx] = v_policy_order
        return out

    def _from_mj(self, v_mj, kind: str) -> np.ndarray:
        """MuJoCo 序(qposadr/dofadr/actuator) -> 策略关节序。"""
        src = {"q": self.qadr, "dq": self.dofadr, "act": self.act_idx}[kind]
        return np.asarray(v_mj)[src]

    # ---- 状态读取 ----
    def base_state(self):
        d = self.data
        adr = self.base_adr
        pos = d.qpos[adr:adr + 3].copy()
        quat = d.qpos[adr + 3:adr + 7].copy()               # wxyz
        # MuJoCo free joint 的 qvel 角速度在"父子正中帧",不是体系!
        # 用 mj_objectVelocity 取真正的体系速度
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, d, mujoco.mjtObj.mjOBJ_BODY,
                                 self._base_bid, vel, 1)
        ang_vel_b = vel[0:3].copy()
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, quat)
        R = R.reshape(3, 3)
        proj_grav_b = R.T @ np.array([0.0, 0.0, -1.0])
        return pos, quat, ang_vel_b, proj_grav_b, R

    def joint_state(self):
        q = self._from_mj(self.data.qpos, "q")
        dq = self._from_mj(self.data.qvel, "dq")
        return q, dq

    # ---- 策略 ----
    def _load_policy(self, path: str):
        import torch                                        # 延迟导入
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"策略文件不存在: {p}")
        buf = io.BytesIO(p.read_bytes())                    # 绕开 torch 的中文路径 fopen
        self.policy = torch.jit.load(buf, map_location="cpu")
        self.policy.eval()
        self._torch = torch
        print(f"[policy] 已加载: {p.name}")

    def _build_obs(self) -> np.ndarray:
        pos, quat, ang_vel_b, proj_grav_b, _ = self.base_state()
        q, dq = self.joint_state()
        clip = self.cfg.obs_clip
        terms = {
            "base_ang_vel": np.clip(ang_vel_b, -clip, clip),
            "projected_gravity": np.clip(proj_grav_b, -clip, clip),
            # 注意用训练时的原始项名:它决定字母序的位置(k < l,排在 last_action 前)
            "keyboard_velocity_commands": np.clip(self.commands, -clip, clip),
            "joint_pos_rel": np.clip(q - self.q_default, -clip, clip),
            "joint_vel_rel": np.clip(dq, -clip, clip),
            "last_action": np.clip(self.last_action, -clip, clip),
            "height_scan": np.clip(self.scanner.compute(self.model, self.data), -1.0, 1.0),
        }
        # 训练时 IsaacLab 的观测项按名字字母序拼接(deploy.yaml 的文件序可能不同)
        order = self.cfg.obs_order or sorted(terms)
        if self.cfg.obs_order == "yaml":
            order = ["base_ang_vel", "projected_gravity", "keyboard_velocity_commands",
                     "joint_pos_rel", "joint_vel_rel", "last_action", "height_scan"]
        obs = np.concatenate([terms[k] for k in order]).astype(np.float32)
        return obs

    def _policy_infer(self) -> np.ndarray:
        obs = self._build_obs()
        torch = self._torch
        with torch.no_grad():
            action = self.policy(torch.from_numpy(obs).unsqueeze(0))
        action = action.squeeze(0).cpu().numpy().astype(np.float64)
        self.last_action = action.copy()
        clip = self.cfg.action_clip
        return self.q_default + self.action_scale * np.clip(action, -clip, clip)

    def update_commands(self, target):
        """上位机设置速度指令(内部按部署端同款 EMA 平滑)。"""
        self.cmd_target = np.asarray(target, dtype=float)

    def _smooth_commands(self):
        a = float(getattr(self.cfg, "command_smoothing", 0.2))
        self.commands += a * (self.cmd_target - self.commands)

    # ---- 主入口:按当前模式算力矩,写 data.ctrl ----
    def compute(self) -> np.ndarray:
        """每个物理步调用一次;策略模式内部按 step_dt 节流推理。

        tau 全程是"策略关节序",写入 ctrl 时按 act_idx 散射。
        """
        q, dq = self.joint_state()
        if self.mode == "damping":
            tau = -self.kd * dq
        elif self.mode == "force":
            tau = self.tau_ui
        elif self.mode == "pd_stand":
            tau = self.kp * (self.q_default - q) - self.kd * dq
        elif self.mode == "position":
            tau = self.kp * (self.q_des_ui - q) - self.kd * dq
        elif self.mode == "policy":
            if self.policy is None:
                raise RuntimeError("当前配置没有 policy,无法进入 RL行走 模式")
            if self.data.time - self._last_policy_t >= self.step_dt - 1e-9:
                self._smooth_commands()
                self.q_des_policy = self._policy_infer()     # 策略关节序
                self._last_policy_t = self.data.time
            tau = self.kp * (self.q_des_policy - q) - self.kd * dq
        else:
            tau = np.zeros(len(self.joint_names))
        self.data.ctrl[self.act_idx] = tau
        return tau

    def reset(self):
        """复位:清策略内部状态(机器人 qpos 由外部复位)。"""
        self.last_action = np.zeros(len(self.joint_names))
        self._last_policy_t = -1e9
        self.q_des_policy = self.q_default.copy()


def _expand(v, n: int) -> list[float]:
    if isinstance(v, (int, float)):
        return [float(v)] * n
    v = list(v)
    if len(v) != n:
        raise ValueError(f"控制器参数长度 {len(v)} != 关节数 {n}")
    return v
