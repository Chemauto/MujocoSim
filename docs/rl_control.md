# rl_control/ —— RL 控制层

控制器的所有变体都在这一层:五种模式 + 策略推理 + 高度扫描观测。
**只依赖 common(配置),经组合根注入 RobotAPI 与 HeightScanner,
不 import simulation / mujoco——控制逻辑可以直接搬去实机部署侧。**

## 文件

| 文件 | 职责 |
|---|---|
| `controller.py` | `RobotController`:五种模式的力矩计算、RL 观测构建、指令平滑、复位 |
| `policy.py` | `PolicyRunner`:TorchScript 策略加载与推理(torch 字节流加载,中文路径可用) |

## 五种模式

| 模式 | 力矩 |
|---|---|
| `damping` 阻尼模式 | tau = -kd·dq(关节软,机器人瘫软) |
| `force` 力控模式 | tau = tau_ui(上位机 12 个力矩滑条) |
| `pd_stand` PD站立 | tau = kp(q_default - q) - kd·dq,锁默认站姿 |
| `position` 位控模式 | tau = kp(q_des_ui - q) - kd·dq,12 个角度滑条 |
| `policy` RL行走 | 策略按 step_dt(50Hz)出目标角 + PD 跟踪 |

## RL 策略接入(以 go2 walk 为例)

已对接 llm-legged-lab 训练的 walk 策略
(`ModelBackup/TransPolicy/WalkFlatLowHeightTransfer.pt`,配置可换任意 TorchScript)。
与部署链路(deploy.yaml + unitree_sdk2py_bridge)逐项对齐:

### 观测(232 维)

```
[base_ang_vel 3, projected_gravity 3, 速度指令 3,
 joint_pos_rel 12, joint_vel_rel 12, last_action 12, height_scan 187]
```

- 拼接顺序:`obs_order: yaml` = deploy.yaml 文件序 = 训练序
  (IsaacLab 生成器 "order preserved");留空则按字母序
- 每项按 `obs_clip`(±100)截断;高度扫描单独 ±1

### 关节序

`controller.joint_names` 就是策略关节序:
`[FL_hip_joint, FR_hip_joint, RL_hip_joint, RR_hip_joint, FL_thigh_joint, …]`
(与 deploy.yaml 的 joint_ids_map 等价,但按"名字"配置,可读且机器人无关)。
内部经 `RobotAPI.actuator_ids` 映射到 MuJoCo 执行器。

### 动作

```
q_des = default_joint_pos + action_scale · clip(action, ±action_clip)
tau   = kp·(q_des - q) - kd·dq          # kp/kd 来自配置(25 / 0.5)
```

策略 50Hz(step_dt 0.02)推理一次,PD 每个物理步执行;
速度指令经 EMA(command_smoothing 0.2,与部署端键盘一致)平滑。

### 高度扫描(187 点)

见 [sensors.md](sensors.md) HeightScanner:17×11 网格,yaw 对齐,射线 +20m 向下,
`heights = base_z - hit_z - 0.5`,只打 geomgroup (1,1,0,0,0,0),
命中机器人自身自动越过续射。

## 换策略 / 换机器人

- 换策略:改 `controller.policy` 指向另一个 TorchScript;`joint_names`/
  `default_joint_pos`/`action_scale`/`stiffness/damping` 按新策略的部署参数填
- 换机器人:configs 里加对应机器人的 controller 段即可;g1 已有模板
  (12 腿关节,未填 policy 时支持阻尼/力控/PD站立/位控)

## 踩过的坑(换策略时注意)

1. **MuJoCo freejoint 的 qvel 角速度在"父子正中帧"**,不是体系——RobotAPI 已用
   `mj_objectVelocity` 处理;自己写部署代码时别直接用 `data.qvel[3:6]`。
2. **观测顺序**:deploy.yaml 生成器直接取自训练环境管理器("order preserved"),
   用 `obs_order: yaml`;字母序是另一套(名字变了排序会变,如 keyboard_velocity_commands)。
3. **出生高度**:从高处落下会产生分布外的高度扫描瞬态,策略会锁死站立;
   `spawn` 设为站立高度(0.30)即可正常起步。
4. 策略行走带缓慢偏航属正常(无 heading 指令项),用 wz 修正。
