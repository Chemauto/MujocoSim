<div align="center">

# 🤖 MujocoSim

**面向 Sim2Real 的容器化 MuJoCo 仿真平台**

[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![MuJoCo](https://img.shields.io/badge/Physics-MuJoCo-red.svg)](https://github.com/google-deepmind/mujoco)
[![ROS2](https://img.shields.io/badge/ROS2-Jazzy-blue.svg)](https://docs.ros.org/)
[![Docker](https://img.shields.io/badge/Run-Docker-2496ED.svg)](https://www.docker.com/)

</div>

---

## 🎯 这是什么？

一个开箱即用的 MuJoCo 仿真平台：多机器人（Go2 / G1 / SO-ARM101）、场景随手改、
策略直接跑、状态与传感器走 ROS2 话题，全部在容器里运行。

- 配置文件即工程：机器人、场景、策略映射、传感器开关都写在一份 YAML 里
- 策略的输入输出与仿真量（关节/IMU/指令…）的对应关系写在配置中，加载时自动校验

## ✨ 核心特性

- 🦿 **多机器人一键切换** —— Go2 / G1 / SO-ARM101，换一行 YAML
- 🌍 **场景 XML 手改** —— 场景文件机器人无关，改完立即生效
- 🧠 **策略映射可配置** —— 观测/动作与策略输入输出的对应关系写在 YAML，加载即校验
- 📡 **ROS2 话题接口** —— 指令进（cmd_vel / joint_command），状态与传感器出
- 📷 **传感器一键开关** —— RGB / 深度 / 2D&3D Lidar / 高度图
- 🐳 **容器化运行** —— build.sh + enter.sh 双入口，VSCode Dev Containers 开箱即用

## 🚀 快速开始

```bash
# 1. 构建镜像（首次；构建完成后自动打开 VSCode Dev Container）
./build.sh

# 2. 在 VSCode 里打开容器终端（或不开 VSCode 时用 ./enter.sh 进容器）
./enter.sh

# 3. 运行程序
python3 sim_entry.py            # 终端1：打开仿真器（MuJoCo 3D 窗口）
python3 scripts/mainctl.py      # 终端2：打开控制器（Qt 面板：模式切换 / 遥控 / 状态）
python3 scripts/joystick_node.py  # 终端3（可选）：虚拟手柄（鼠标拖摇杆控速度）
```

仿真器与控制器是**独立进程**，通过 ROS2 话题通信（`/cmd_vel`、`/control_mode`、`/joint_states`…），
所以也可以只开仿真器用 `ros2 topic` 手动发指令，或让控制器对接真机。

仿真器参数：

```bash
python3 sim_entry.py --mode motion --command "0.3 0 0"   # 策略行走
python3 sim_entry.py --mode damping                      # 零力矩（软倒调试）
python3 sim_entry.py --robot so101 --mode position       # 换机器人 / 换模式
python3 sim_entry.py --no-gui                            # 无界面（纯后台）
```

国内网络构建可换源：

```bash
BASE_IMAGE=docker.m.daocloud.io/library/ros:jazzy-ros-base ./build.sh
```

## 🔧 改场景 / 换机器人

**加物体、加相机** —— 直接编辑 `scenes/flat.xml`，抄一行：

```xml
<body name="blue ball" pos="0.8 0.5 0.3">
  <freejoint/><geom type="sphere" size="0.1" rgba="0.1 0.3 0.9 1"/>
</body>
```

**换场景 / 换机器人 / 换策略** —— 改 `configs/config.yaml` 三行：

```yaml
robot: g1                            # go2 | g1 | so101
scene: scenes/push_box.xml           # flat.xml | push_box.xml | 自己写的
policy: walk                         # rl_controller/robot/<robot>/policies.yaml 里的策略名
```

## 📂 目录结构

```
MujocoSim/
├── scenes/                 # 🌍 场景 XML（手改这里，机器人无关）
├── assets/
│   ├── robots/             # 🦿 机器人本体 MJCF + robot.yaml 元数据
│   └── sensors/            # 📷 传感器模块库（做好即插：相机/雷达/高度图）
├── configs/
│   └── config.yaml         # ⚙️ 唯一入口：机器人/场景/策略/模式 + mounts 传感器接入
├── rl_controller/          # 🧠 策略推理与接线
│   └── robot/
│       └── go2/
│           ├── policy/     #    该机器人的策略模型（ONNX）
│           └── policies.yaml  #  多策略接线（观测输入按顺序 / 动作输出标注）
├── simutils/               # 🕹️ 虚拟手柄（pygame 画 Xbox 手柄，鼠标操控）
├── scripts/                # 🛠️ mainctl.py 控制器 / joystick_node.py 手柄 / 策略工具
├── mujocosim/              # 📦 仿真侧
│   ├── config.py           #    YAML → 强类型配置（加载即校验）
│   ├── world.py            #    场景加载 + 步进循环 + viewer
│   ├── controller.py       #    控制模式：damping / pd_stand / position / motion
│   └── sensors/            #    相机 / 雷达 / 高度图
├── docker/                 # 🐳 镜像定义
├── sim_entry.py            # ▶️ 仿真器入口
├── build.sh  enter.sh      # 一键构建 / 进容器
└── .devcontainer/          # VSCode 容器开发
```

## 🗺️ 开发路线

| 阶段 | 内容 | 状态 |
|------|------|:----:|
| 1 | 配置系统 + 容器基座 | ✅ |
| 2 | 三机器人资产导入 | ✅ |
| 3 | 仿真核心（场景 XML + world） | ✅ |
| 4 | 控制层（damping / pd_stand / position / motion） | ✅ |
| 5 | 传感器（相机 / 深度 / 雷达 / 高度图） | ✅ |
| 6 | 策略接线与 ONNX 推理（go2 velocity 已跑通） | 🚧 |
| 7 | ROS2 桥（含手柄 / 控制器接入） | ✅ |
| 8 | 入口与预设配置 | ⬜ |
| 9 | 容器化收尾（GPU / GUI） | ⬜ |
| 10 | 端到端联调 | ⬜ |

## 🙏 致谢

- **[MuJoCo](https://github.com/google-deepmind/mujoco)** — 物理引擎
- **[sim2sim.stackforce.cc](https://sim2sim.stackforce.cc/)** — 策略映射设计参考
- **[unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)** — 仿真器架构参考
- **[mujoco-learning](https://github.com)** — 学习与代码参考

---

<div align="center">

**[⬆ 回到顶部](#-mujocosim)**

</div>
