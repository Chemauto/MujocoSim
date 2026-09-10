# configs/ —— 场景配置

一个 YAML = 一个场景。**换机器人/相机/物体/地形/策略/数据导出,只改这里。**

| 文件 | 场景 |
|---|---|
| `go2.yaml` | go2 四足 + 车载/俯视相机 + 3 个检测物体 + walk 策略 |
| `go2_terrain.yaml` | go2 + 程序化地形(斜坡/楼梯/悬浮楼梯/乱石地/Perlin 高程图)+ 可推箱 |
| `g1.yaml` | g1 人形(控制器 12 腿关节,未填策略) |

## 配置段速查

| 段 | 消费方 | 说明 |
|---|---|---|
| `robot` / `spawn` / `timestep` / `realtime` | simulation | 机器人名 = `assets/robots/` 文件夹名 |
| `objects` | simulation/builder + YOLO 类别 | name/type(box/cylinder/sphere/capsule)/size(完整尺寸)/pos/rgba/euler(度)/mass/friction/static |
| `terrain` | simulation/terrain.py | 列表,`type`: box/stairs/suspend_stairs/rough_ground/perlin_hfield/image_hfield |
| `cameras` | simulation/builder | name/attach(BASE=机器人根)/pos/lookat/euler/quat/track/fovy/width/height |
| `controller` | rl_control | joint_names(策略关节序)/step_dt/stiffness/damping/default_joint_pos/action_scale/action_clip/obs_clip/policy/obs_order/command_smoothing/force_limit/height_scan(size/resolution/ray_offset_z/height_offset/geomgroup)/commands/initial_mode |
| `yolo` | yolo | enabled/camera/weights/conf/classes/interval/save_dir |
| `data` | common/data_bus | enabled/format/dir/rate/topics |

各段字段含义与取值见对应模块文档:
[simulation.md](simulation.md) / [rl_control.md](rl_control.md) / [common.md](common.md)。

## 最小可用配置

```yaml
robot: go2        # 只要有这一行,其余全部走默认值
```

## 添加新机器人

见 [../assets/README.md](../assets/README.md):MJCF + 5 行 robot.yaml。
