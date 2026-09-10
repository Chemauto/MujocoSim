# sensors/ —— 传感器层

外感知传感器(相机、高度扫描)统一放这里;数据归类为 `SensorData`
(见 [common.md](common.md))。允许依赖 common,禁止依赖 simulation/rl_control/yolo。

## camera.py —— SimCamera

| 接口 | 说明 |
|---|---|
| `render_rgb()` | RGB uint8 [H,W,3] |
| `render_depth()` | 深度 float [H,W](米) |
| `render_rgbd()` | 一次渲染返回 (rgb, depth) |
| `intrinsics()` | 针孔内参 {fx, fy, cx, cy}(由 fovy 推出) |
| `pose()` | 相机当前世界位姿 (pos, R) |
| `cam_to_world(p)` / `cv_to_world(p)` | 相机系 → 世界系(MuJoCo 原生 / OpenCV 惯例) |
| `deproject_pixel(depth, u, v, K)` | 像素 + 深度 -> 相机系 3D 点 |
| `depth_colormap(depth)` | 深度伪彩图 |
| `imread_u / imwrite_u` | 中文路径安全的图片读写 |

- 相机本体(挂载位置/分辨率/FOV)在 `simulation/builder.py` 按 YAML 配置挂进模型;
  本层只负责渲染与几何
- 分辨率各相机独立(一台相机一个 Renderer)

坐标约定:`xyz_cam` 用 OpenCV 惯例(x 右、y 下、z 向前为正);
深度测的是**可见表面**,比物体质心近半个物体尺寸。

## height_scan.py —— HeightScanner

RL 策略的外感知:187 点(17×11,1.6×1.0m @ 0.1m)相对高度扫描。

```
网格随 base 偏航旋转 → 平移到 base → 射线起点 +20m → 竖直向下 mj_ray
heights = base_z - hit_z - height_offset(0.5)
只打 geomgroup(默认 (1,1,0,0,0,0),即地面/地形/物体);命中机器人
body 时自动越过续射,保证永不打中自身;未命中记 0
```

全部参数(size/resolution/ray_offset_z/height_offset/geomgroup/max_bounces 续射预算)在配置
`controller.height_scan` 段可调;机器人自遮挡由 `robot_body_ids`
(组合根自动传入)处理,任何机器人无需额外配置。

逐行对照 llm-legged-lab `unitree_sdk2py_bridge._compute_height_map` 移植,
参数在配置 `controller.height_scan` 段。

数据分类:高度扫描属于**传感器**(`sensor/height_scan` 话题);
关节/电机属于**机器人本体**(`robot/state`,见 [common.md](common.md))。
