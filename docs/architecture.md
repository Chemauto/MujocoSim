# 架构总览

本仓库是一个**配置驱动**的 MuJoCo 仿真平台:任意机器人 + 程序化地形 + 任意相机 +
YOLO 感知 + RL 策略控制(上位机遥控)。分层清晰,每层只依赖约定的方向。

## 分层与依赖方向

```
main.py / dashboard.py        组合根:装配一切,允许依赖所有层
        │
  ┌─────┼──────────┬───────────────┬─────────────┐
  ▼     ▼          ▼               ▼             ▼
simulation  rl_control  sensors      yolo         common
场景搭建     RL 控制层    传感器层      感知(检测)   通用层(配置/数据)
  │           │           │
  └───── common ←──────────┘  (common 不依赖任何内部包)
```

**依赖规则**(代码评审会检查):

| 层 | 允许依赖 | 禁止依赖 |
|---|---|---|
| `common` | 只允许第三方(yaml/numpy) | 所有内部包 |
| `sensors` | common | simulation / rl_control / yolo |
| `simulation` | common | rl_control / sensors / yolo |
| `rl_control` | common(经注入使用 sensors 实例) | simulation / mujoco / sensors |
| `yolo` | sensors | simulation / rl_control / common |
| `main` / `dashboard` | 全部 | — |

## 目录

| 目录/文件 | 职责 | 详细文档 |
|---|---|---|
| `common/` | YAML 配置层;数据分类模型(机器人/环境/传感器);ROS2 风格数据总线与导出 | [common.md](common.md) |
| `simulation/` | 场景搭建与运行:MJCF 运行时组装、程序化地形、SimWorld、RobotAPI | [simulation.md](simulation.md) |
| `rl_control/` | 控制器:五种模式(阻尼/力控/PD站立/位控/RL行走)、策略推理 | [rl_control.md](rl_control.md) |
| `sensors/` | 传感器:相机(RGB/深度/内参)、高度扫描 | [sensors.md](sensors.md) |
| `yolo/` | YOLO-World 开放词表检测 + 深度测距 | [yolo.md](yolo.md) |
| `configs/` | 全部 YAML 配置(场景的唯一"可变项"来源) | [configs.md](configs.md) |
| `assets/robots/` | 机器人资产(一个文件夹 = 一种机器人) | [../assets/README.md](../assets/README.md) |
| `main.py` | 无 UI 主程序:仿真 + 检测循环 / oneshot | — |
| `dashboard.py` | 上位机:模式切换 + 速度遥控 + 相机/YOLO 画面 | — |

## 数据分类(谁的数据放哪里)

| 分类 | 内容 | 话题 | 代码 |
|---|---|---|---|
| **机器人**(本体感知) | 关节角/速/力矩、base 位姿/角速度/重力投影 | `robot/state` | `common/data_model.py: RobotState` |
| **环境** | 仿真时间、场景物体位姿 | `env/state` | `common/data_model.py: EnvironmentState` |
| **传感器**(外感知) | 高度扫描 187 点;相机帧(体积大,不进 JSONL) | `sensor/height_scan` | `common/data_model.py: SensorData` |

## 数据流

```
                    ┌──────── 组合根每物理步调用 world.step() ───────┐
                    │                                              │
                    │  1. pre_step()  —— 控制器算力矩写入 ctrl       │
                    │  2. exporter.update() —— 按 rate 节流采集发布  │
                    │  3. mj_step()   —— 物理步进                   │
                    ▼                                              │
 mujoco physics ◀──────────────────────────────────────────────────┘
      │ 仿真状态更新
      ▼
 DataBus.publish ──▶ JsonlRecorder ──▶ outputs/data/*.jsonl
```

## 组合根的装配顺序(伪代码)

```python
cfg      = load_config("configs/go2.yaml")       # common
world    = SimWorld(cfg, camera_factory=...)     # simulation:场景 + RobotAPI(camera 注入)
scanner  = HeightScanner(world.model, world.data, ...)   # sensors
controller = RobotController(world.robot, cfg.controller, scanner)  # rl_control
world.pre_step = controller.compute              # 控制钩子
world.exporter = DataExporter(cfg.data, world, controller, scanner)  # 数据导出
controller.reset_robot(world)                    # 站姿出生
```

## 运行

```bash
python dashboard.py                # 上位机(推荐)
python main.py                     # 纯仿真 + 检测循环
python main.py --oneshot           # 无窗口单次检测
```
