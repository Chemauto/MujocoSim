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

一个开箱即用的 MuJoCo 仿真平台：多机器人（Go2 / G1 / SO-ARM101 / Booster K1）、场景随手改、
策略直接跑、状态与传感器走 ROS2 话题，全部在容器里运行。

- **配置文件即工程**：机器人、场景、策略、传感器接入都写在一份 YAML 里
- **接线表驱动策略**：策略输入输出与仿真量（关节 / IMU / 指令 / 运动参考…）的对应关系写在配置中，加载时自动校验维度与完整性

## ✨ 核心特性

- 🦿 **多机器人一键切换** —— Go2 / G1 / SO-ARM101 / Booster K1，换一行 YAML
- 🌍 **场景 XML 手改** —— 场景文件机器人无关，改完立即生效
- 🧠 **策略映射可配置** —— 观测 / 动作与策略输入输出的对应关系写在 YAML
- 🎬 **运动跟踪** —— 回放 npz 参考动作驱动策略（beyond_mimic 类）
- 📡 **ROS2 话题接口** —— 指令进（cmd_vel / joint_command / control_mode），状态与传感器出
- 📷 **传感器插拔** —— 相机 / 深度 / 2D&3D 雷达 / 高度图，一行 mounts 接入
- 🐳 **容器化运行** —— build.sh / enter.sh 双入口，VSCode Dev Containers 开箱即用

## 🚀 快速开始

```bash
# 1. 构建镜像（首次；构建完成后自动打开 VSCode Dev Container）
./build.sh

# 2. 进入容器（VSCode 容器终端，或 ./enter.sh）
./enter.sh

# 3. 运行程序
python3 sim_entry.py              # 终端1：仿真器（MuJoCo 3D 窗口）
python3 scripts/mainctl.py        # 终端2：控制器（Qt 面板：模式切换 / 遥控 / 状态）
python3 scripts/joystick_node.py  # 终端3（可选）：虚拟手柄（鼠标拖摇杆控速度）
python3 scripts/camera_viewer.py  # 终端4（可选）：相机画面监视 + YOLO 检测框（话题出现自动开窗）
```

仿真器与控制器是独立进程，通过 ROS2 话题通信，也可以直接用 `ros2 topic pub` 发指令。

仿真器参数：

```bash
python3 sim_entry.py --robot g1                          # 换机器人
python3 sim_entry.py --mode damping                      # 控制模式：damping / pd_stand / position / motion
python3 sim_entry.py --mode motion --command "0.3 0 0"   # 策略运动 + 速度指令
python3 sim_entry.py --policy dance_002 --mode motion    # 选用指定策略
python3 sim_entry.py --no-gui                            # 无界面（纯后台）
```

相机监视自带 YOLOv8s 目标检测：检测框叠加在 RGB 画面上，配合深度画面给出目标距离
（`R=x.xxm`，bbox 中央深度中位数 + 针孔反投影）。

```bash
python3 scripts/camera_viewer.py                          # 默认 yolov8s.pt，conf=0.25
python3 scripts/camera_viewer.py --yolo-conf 0.1          # 调低阈值
python3 scripts/camera_viewer.py --yolo-classes person    # 只看指定类别
python3 scripts/camera_viewer.py --no-yolo                # 纯画面
```

指定基础镜像源构建：

```bash
BASE_IMAGE=docker.m.daocloud.io/library/ros:jazzy-ros-base ./build.sh
```

## 🔧 配置

### 选机器人 / 场景 / 策略 —— `configs/config.yaml`

```yaml
robot: go2                       # go2 | g1 | so101 | booster_k1
scene: scenes/flat.xml           # scenes/ 下的场景文件
policy: velocity                 # rl_controller/robot/<robot>/policies.yaml 里的策略名

controller:
  mode: pd_stand                 # damping | pd_stand | position | motion
  kp: 25.0
  kd: 0.5
```

### 改场景 —— `scenes/*.xml`

场景文件机器人无关，加物体 / 相机抄一行：

```xml
<body name="blue ball" pos="0.8 0.5 0.3">
  <freejoint/><geom type="sphere" size="0.1" rgba="0.1 0.3 0.9 1"/>
</body>
```

### 接传感器 —— `mounts`（设备库 `assets/sensors/`）

```yaml
mounts:
  - { device: front_camera, name: head_camera, attach: base_link, pos: [0.35, 0, 0.2], rpy: [0, 0, 0] }
  - { device: lidar_3d,     attach: base_link, pos: [0, 0, 0.3] }
  - { device: heightmap,    attach: base_link }
```

- 话题自动派生：`/camera/<name>/...`、`/points`、`/heightmap`；删一行即关闭该传感器
- `rpy: [0, 0, 0]` = 朝前看；`attach` 换 body 名即换机器人（go2=base_link / g1=pelvis / booster_k1=trunk）

### 策略接线 —— `rl_controller/robot/<robot>/policies.yaml`

观测按顺序拼接成策略输入向量，动作标注控制语义：

```yaml
policies:
  velocity:
    model: policy/policy.onnx
    observations:
      - { source: gyro,               span: [0, 3] }
      - { source: projected_gravity,  span: [3, 6] }
      - { source: joint_qpos,         span: [6, 18], params: { offset: zero_pose } }
      - { source: joint_qvel,         span: [18, 30] }
      - { source: last_action,        span: [30, 42] }
    actions:
      mode: position_offset          # q* = offset + scale · action
      scale: 0.25
      control: { kp: 25.0, kd: 0.5 }
      mapping: auto                  # 或逐关节标注 sign / scale / offset / limit
```

运动跟踪策略配置 `motion: motion/xxx.npz` 并使用 `motion_command` / `motion_anchor_ori` 引脚源。
`joint_order` 可声明策略关节序与电机序的对应；`span` 为可选标注，与顺序推导不一致时报错。

配套工具：

```bash
python3 scripts/import_deploy.py <deploy.yaml> --robot go2 --name velocity   # deploy.yaml → 接线表
python3 scripts/export_rsl_checkpoint.py model_XXXX.pt --out policy.onnx     # RSL-RL checkpoint → ONNX
```

## 📡 ROS2 话题

**订阅（指令）**

| 话题 | 类型 | 说明 |
|------|------|------|
| `/control_mode` | `std_msgs/String` | damping / pd_stand / position / motion |
| `/cmd_vel` | `geometry_msgs/Twist` | 速度指令（vx, vy, yaw_rate） |
| `/joint_command` | `sensor_msgs/JointState` | 关节目标 |
| `/reset` | `std_msgs/Empty` | 重置场景 |

**发布（状态 / 传感器）**

| 话题 | 类型 | 说明 |
|------|------|------|
| `/joint_states` | `sensor_msgs/JointState` | 关节位置 / 速度 / 力矩（50 Hz） |
| `/odom` | `nav_msgs/Odometry` | 基座位姿与速度 |
| `/imu` | `sensor_msgs/Imu` | 姿态与角速度 |
| `/camera/<name>/rgb/image_raw` | `sensor_msgs/Image` | RGB 图像 |
| `/camera/<name>/rgb/camera_info` | `sensor_msgs/CameraInfo` | 相机内参 |
| `/camera/<name>/depth/image_raw` | `sensor_msgs/Image` | 深度图（32FC1，米制） |
| `/scan` | `sensor_msgs/LaserScan` | 2D 激光雷达 |
| `/points` | `sensor_msgs/PointCloud2` | 3D 激光雷达 |
| `/heightmap` | `sensor_msgs/PointCloud2` | 高度图点云 |

手柄键位（`scripts/joystick_node.py`）：左摇杆平移、右摇杆 X 转向；A=急停、B=站立、Y=策略、X=位控、start=重置。

## 📂 目录结构

```
MujocoSim/
├── scenes/                 # 🌍 场景 XML（机器人无关，手改这里）
├── assets/
│   ├── robots/             # 🦿 机器人 MJCF + 网格 + robot.yaml
│   └── sensors/            # 📷 传感器设备库（相机 / 雷达 / 高度图）
├── configs/
│   └── config.yaml         # ⚙️ 唯一入口：机器人 / 场景 / 策略 / 模式 / mounts
├── detect/
│   └── yolo/               # 🎯 YOLOv8s 检测 + 深度测距（yolov8s.pt 权重）
├── rl_controller/          # 🧠 策略推理与接线
│   └── robot/<robot>/
│       ├── policy/         #    策略模型（ONNX）
│       ├── motion/         #    运动参考（npz）
│       └── policies.yaml   #    接线表（观测输入顺序 / 动作输出标注）
├── mujocosim/              # 📦 仿真核心
│   ├── config.py           #    配置加载与校验
│   ├── world.py            #    场景加载 + 步进 + viewer
│   ├── controller.py       #    控制模式
│   ├── ros2_bridge.py      #    ROS2 话题桥
│   └── sensors/            #    传感器运行时
├── scripts/                # 🛠️ mainctl.py / joystick_node.py / 策略工具
├── simutils/               # 🕹️ 虚拟手柄（pygame）
├── docker/                 # 🐳 镜像定义（zsh + oh-my-zsh + ROS2）
├── sim_entry.py            # ▶️ 仿真器入口
├── build.sh  enter.sh      # 构建 / 进容器
└── .devcontainer/          # VSCode Dev Containers
```

## 🙏 致谢

- **[MuJoCo](https://github.com/google-deepmind/mujoco)** — 物理引擎
- **[sim2sim.stackforce.cc](https://sim2sim.stackforce.cc/)** — 策略映射设计参考
- **[unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)** — 仿真器架构参考
- **[mujoco-learning](https://github.com)** — 代码参考
- **[Loco-Mani-Plan-Base](https://github.com)** — 虚拟手柄模块
- **[Booster Robotics](https://www.boosterobotics.com/)** — K1 模型与运动数据

---

<div align="center">

**[⬆ 回到顶部](#-mujocosim)**

</div>
