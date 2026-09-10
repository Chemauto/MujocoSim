"""controller.py —— 多模式机器人控制器(见 docs/rl_control.md)。

五种模式(dashboard/main 可实时切换):
  damping   阻尼模式    tau = -kd·dq(关节软,机器人瘫软)
  force     力控模式    tau = 上位机滑条给定的力矩
  pd_stand  PD 站立     tau = kp*(q_default - q) - kd*dq,锁默认站姿
  position  位控模式    tau = kp*(q_des_ui - q) - kd*dq,滑条摆姿势
  policy    RL 行走     训练好的策略(TorchScript)按 step_dt 出目标角,PD 跟踪

与 llm-legged-lab 部署链路逐项对齐(deploy.yaml + unitree_sdk2py_bridge):
  观测 232 维 = [ang_vel 3, projected_gravity 3, cmd 3,
                 joint_pos_rel 12, joint_vel_rel 12, last_action 12, height_scan 187];
  关节按"策略关节序"(cfg.joint_names)排布,经 RobotAPI 映射到执行器;
  动作 q_des = default + scale·clip(action, ±action_clip),50Hz,指令 EMA 平滑。

本层只依赖 common(配置)与注入的 RobotAPI/HeightScanner/PolicyRunner,
不 import simulation / mujoco——控制逻辑可直接搬去实机部署侧。
"""
from __future__ import annotations

import numpy as np

from common.config import ControllerCfg

MODES = ["damping", "force", "pd_stand", "position", "policy"]
MODE_LABELS = {"damping": "阻尼模式", "force": "力控模式", "pd_stand": "PD站立",
               "position": "位控模式", "policy": "RL行走"}

# deploy.yaml 文件序 = 训练序(IsaacLab 生成器 "order preserved")
YAML_OBS_ORDER = ["base_ang_vel", "projected_gravity", "keyboard_velocity_commands",
                  "joint_pos_rel", "joint_vel_rel", "last_action", "height_scan"]


def _expand(v, n: int) -> list[float]:
    if isinstance(v, (int, float)):
        return [float(v)] * n
    v = list(v)
    if len(v) != n:
        raise ValueError(f"控制器参数长度 {len(v)} != 关节数 {n}")
    return v


class RobotController:
    """按模式算力矩。对外关节序一律为 cfg.joint_names("策略关节序")。"""

    def __init__(self, robot, cfg: ControllerCfg, height_scanner=None):
        """robot: simulation.robot_api.RobotAPI(组合根注入);
        height_scanner: sensors.height_scan.HeightScanner(policy 模式必需)。"""
        self.robot = robot
        self.cfg = cfg
        if cfg.initial_mode not in MODES:
            print(f"[controller] 未知初始模式 '{cfg.initial_mode}',回退 pd_stand")
            self.mode = "pd_stand"
        else:
            self.mode = cfg.initial_mode

        if not cfg.joint_names:
            raise ValueError("controller.joint_names 不能为空(策略关节序)")
        self.joint_names = list(cfg.joint_names)
        self.act_ids = robot.actuator_ids(self.joint_names)      # 策略序 -> 执行器下标

        self.kp = np.array(_expand(cfg.stiffness, len(self.joint_names)), dtype=float)
        self.kd = np.array(_expand(cfg.damping, len(self.joint_names)), dtype=float)
        self.q_default = np.array(_expand(cfg.default_joint_pos, len(self.joint_names)),
                                  dtype=float)
        self.commands = np.array(_expand(cfg.commands, 3), dtype=float)
        self.q_des_ui = self.q_default.copy()          # 位控模式目标(上位机改)
        self.tau_ui = np.zeros(len(self.joint_names))  # 力控模式力矩(上位机改)
        self.cmd_target = self.commands.copy()                # 上位机写这里

        self.action_scale = np.array(_expand(cfg.action_scale, len(self.joint_names)), dtype=float)
        self.step_dt = float(cfg.step_dt)
        self._last_policy_t = -1e9
        self.last_action = np.zeros(len(self.joint_names))
        self.last_torques = np.zeros(len(self.joint_names))
        self.q_des_policy = self.q_default.copy()

        self.runner = None
        self.scanner = height_scanner
        if cfg.policy:
            # 策略文件缺失/损坏不阻断启动:非 policy 模式(站立/位控等)仍可用
            try:
                from .policy import PolicyRunner
                self.runner = PolicyRunner(cfg.policy)
            except Exception as e:
                print(f"[controller] 策略加载失败({e}),RL行走 模式不可用")
        if self.runner is None and self.mode == "policy":
            print("[controller] 初始模式 policy 不可用,回退为 pd_stand")
            self.mode = "pd_stand"
        if cfg.obs_order not in ("", "yaml"):
            raise ValueError(f"obs_order 只支持 ''(字母序)或 'yaml',收到: {cfg.obs_order!r}")

    # ---- 状态(策略关节序) ----
    def joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        return self.robot.joint_q(self.joint_names), self.robot.joint_dq(self.joint_names)

    def base_state(self):
        """(pos, quat[wxyz], ang_vel_b, projected_gravity_b, R)。"""
        pos, quat = self.robot.base_pose()
        ang_vel_b, proj_grav, R = self.robot.base_attitude()
        return pos, quat, ang_vel_b, proj_grav, R

    # ---- RL 观测/推理 ----
    def _build_obs(self) -> np.ndarray:
        if self.scanner is None:
            raise RuntimeError("policy 模式需要高度扫描仪(组合根未注入 HeightScanner)")
        pos, quat, ang_vel_b, proj_grav_b, _ = self.base_state()
        q, dq = self.joint_state()
        clip = self.cfg.obs_clip
        terms = {
            "base_ang_vel": np.clip(ang_vel_b, -clip, clip),
            "projected_gravity": np.clip(proj_grav_b, -clip, clip),
            # 训练时的原始项名:它决定字母序的位置(k < l,排在 last_action 前)
            "keyboard_velocity_commands": np.clip(self.commands, -clip, clip),
            "joint_pos_rel": np.clip(q - self.q_default, -clip, clip),
            "joint_vel_rel": np.clip(dq, -clip, clip),
            "last_action": np.clip(self.last_action, -clip, clip),
            "height_scan": np.clip(self.scanner.compute(), -1.0, 1.0),
        }
        # 训练时 IsaacLab 按名字字母序拼接;deploy.yaml 文件序 = 训练序(推荐)
        order = self.cfg.obs_order or sorted(terms)
        if order == "yaml":
            order = YAML_OBS_ORDER
        return np.concatenate([terms[k] for k in order]).astype(np.float32)

    def _policy_step(self) -> np.ndarray:
        self._smooth_commands()
        action = self.runner.infer(self._build_obs())
        self.last_action = action.copy()
        return self.q_default + self.action_scale * np.clip(action, -self.cfg.action_clip,
                                                            self.cfg.action_clip)

    def update_commands(self, target):
        """上位机设置速度指令(内部按部署端同款 EMA 平滑)。"""
        self.cmd_target = np.asarray(target, dtype=float)

    def _smooth_commands(self):
        a = float(self.cfg.command_smoothing)
        self.commands += a * (self.cmd_target - self.commands)

    # ---- 主入口 ----
    def compute(self) -> np.ndarray:
        """每个物理步调用一次;policy 模式内部按 step_dt 节流推理。

        tau 全程是"策略关节序",写 ctrl 时经 RobotAPI 散射到执行器。
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
            if self.runner is None:
                raise RuntimeError("当前配置没有 policy(或加载失败),无法进入 RL行走 模式")
            if self.robot.sim_time - self._last_policy_t >= self.step_dt - 1e-9:
                self.q_des_policy = self._policy_step()
                self._last_policy_t = self.robot.sim_time
            tau = self.kp * (self.q_des_policy - q) - self.kd * dq
        else:
            tau = np.zeros(len(self.joint_names))
        self.last_torques = tau.copy()
        self.robot.set_torques(self.act_ids, tau)
        return tau

    def reset(self):
        """清控制器内部状态(机器人 qpos 由 world.robot.reset 复位)。"""
        self.last_action = np.zeros(len(self.joint_names))
        self.last_torques = np.zeros(len(self.joint_names))
        self._last_policy_t = -1e9
        self.q_des_policy = self.q_default.copy()

    def reset_robot(self, world) -> None:
        """机器人回到默认站姿出生点,并清控制器内部状态。"""
        world.robot.reset(base_pos=world.spawn_pos,
                          joint_names=self.joint_names,
                          joint_pos=self.q_default)
        self.reset()

