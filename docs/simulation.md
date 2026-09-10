# simulation/ —— 仿真场景层

只负责**场景的搭建与运行**,不含控制器/感知逻辑。

## 文件

| 文件 | 职责 |
|---|---|
| `builder.py` | 运行时组装场景:`机器人 MJCF + 场景模板 + objects + terrain + cameras` → `MjModel` |
| `terrain.py` | 程序化地形(移植自 llm-legged-lab terrain_tool):stairs / suspend_stairs / rough_ground / perlin_hfield / image_hfield / box |
| `world.py` | `SimWorld`:物理步进、viewer、相机渲染接口、物体位姿查询、pre_step/exporter 钩子 |
| `robot_api.py` | `RobotAPI`:机器人访问接口(rl_control 与仿真之间的唯一通道) |

## SimWorld 对外接口

```python
world.step(n=1)            # 物理步进:pre_step() -> exporter.update() -> mj_step
world.sim_time             # 仿真时间(秒)
world.robot                # RobotAPI:机器人访问接口(见下)
world.cameras[name]        # 相机字典(组合根经 camera_factory 注入的 SimCamera)
world.object_poses()       # 环境数据:物体位姿 {name: {pos, quat}}
world.spawn_pos            # 出生位置(cfg.spawn 或 robot.yaml 默认值)
world.open_viewer()        # 交互 viewer;world.close() 关闭一切
world.pre_step             # 钩子:组合根挂控制器(compute)
world.exporter             # 钩子:组合根挂 DataExporter
```

## RobotAPI(rl_control 的唯一机器人通道)

```python
robot.joint_q(names) / joint_dq(names)     # 按名字取关节角/速(rad, rad/s)
robot.actuator_ids(names)                  # 关节名 -> 执行器下标
robot.set_torques(act_ids, torques)        # 写力矩(N·m)
robot.joint_range_rad(name)                # 关节限位(弧度,MuJoCo 原值)
robot.base_pose()                          # base 位姿(世界系)
robot.base_attitude()                      # (体系角速度, 重力投影, R)
robot.robot_state(joint_names, tau)        # -> RobotState(数据总线用)
robot.reset(base_pos, base_quat, joint_names, joint_pos)   # 整体复位
```

关键细节:**MuJoCo freejoint 的 qvel 角速度在"父子正中帧"而非体系**,
`base_attitude()` 内部用 `mj_objectVelocity` 转成真正的体系角速度——
静止时两者相等,运动中差异会导致 RL 策略行为异常(踩过的坑)。

## 场景搭建(builder)要点

- 机器人 XML 用 Python 读文本 + `MjSpec.from_string`,mesh 经 `spec.assets`
  字节流喂入——**绕开 MuJoCo C 层 fopen 打不开中文路径的问题**
- 机器人 worldbody 层的 floor/light 自动剥离(场景统一提供),场景资产
  统一 `scene_` 前缀命名避免重名
- 根 body 无 freejoint 自动补;`spawn`/`view_extent` 来自 robot.yaml 与场景配置
- 离屏渲染缓冲按配置相机的最大分辨率自动设置
- 相机挂载:`attach`(跟随某 body)、`track`(原生跟踪看向某 body)、
  `lookat`/`euler`/`quat` 姿态,全在 YAML 配置
- 场景模板(地面尺寸/灯光/天空盒)是平台固定底座,不随 YAML 变化;
  地形与物体才是场景内容

## 相机注入(分层)

`SimWorld(cfg, camera_factory=...)`:组合根传入
`lambda model, data, name, w, h: SimCamera(...)` 创建相机——
这样 simulation 层无需 import sensors(依赖规则见 architecture.md)。

## 添加新机器人

见 [../assets/README.md](../assets/README.md):一个文件夹 + MJCF + 5 行 robot.yaml,
`configs/` 里把 `robot:` 改成文件夹名即可,零代码。
