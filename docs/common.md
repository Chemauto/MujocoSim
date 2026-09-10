# common/ —— 通用层

全平台唯一允许被所有层依赖的包(它自己只依赖第三方库)。

## config.py —— 配置层

`configs/*.yaml` → dataclass,是所有可变项的唯一来源(代码零硬编码):

| dataclass | 对应 YAML 段 | 消费方 |
|---|---|---|
| `SimConfig` | 顶层(robot/spawn/timestep/realtime) | 组合根 |
| `ObjectCfg` | `objects:` | simulation/builder |
| `CameraCfg` | `cameras:` | simulation/builder |
| `ControllerCfg` | `controller:` | rl_control |
| `YoloCfg` | `yolo:` | yolo / dashboard / main |
| `DataCfg` | `data:` | common/data_bus |

`load_config(path, robot=...)`:路径解析、默认值补全、命令行覆盖机器人。

## data_model.py —— 数据分类模型

三类标准快照,**新增数据先归类到这里**:

| 类 | 分类 | 字段 | 话题 |
|---|---|---|---|
| `RobotState` | 机器人本体 | joint_names/q/dq/tau、base_pos/quat、体系角速度、重力投影 | `robot/state` |
| `EnvironmentState` | 环境 | sim_time、物体位姿 | `env/state` |
| `SensorData` | 传感器(外感知) | height_scan(相机帧体积大,走 sensors 层单独通道) | `sensor/height_scan` |

## data_bus.py —— 数据总线与导出(ROS2 风格)

```
DataBus.publish(topic, stamp, data)      # 发布
DataBus.subscribe(topic, callback)       # 订阅
JsonlRecorder(path)                      # 订阅者:按行写 JSONL
DataExporter(cfg.data, world, controller, height_scanner)   # 采集器
```

- `DataExporter.update()` 由组合根每物理步调用(或经 `world.exporter` 挂载自动调用),
  内部按 `data.rate`(Hz,仿真时间)节流
- 导出格式:JSONL,每行 `{"t": 秒, "topic": 话题, "data": {...}}`,
  pandas 直接 `pd.read_json(path, lines=True)`
- 文件落在 `data.dir`(默认 `outputs/data/run_<时间戳>.jsonl`),已 git-忽略
- 扩展新话题:在 data_model.py 加快照类 → DataExporter.update() 里发布 →
  YAML `data.topics` 加话题名;接真 ROS2 时把 DataBus 换成 rclpy publisher 即可

## 读取导出数据

```python
import pandas as pd
df = pd.read_json("outputs/data/run_xxx.jsonl", lines=True)
robot = df[df.topic == "robot/state"]
q = robot.data.apply(lambda d: d["q"])     # 每行 12 关节角
```
