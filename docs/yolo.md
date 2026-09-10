# yolo/ —— 感知层(YOLO-World 检测 + 深度测距)

依赖 sensors(相机渲染),被 main/dashboard 消费。开放词表:类别就是任意文本。

## detector.py —— YoloWorldDetector

| 接口 | 说明 |
|---|---|
| `YoloWorldDetector(weights, classes, conf)` | 权重路径/类别/阈值全来自配置;**构造时立即加载权重**(缺失/下载失败在构造处即可感知并降级) |
| `detect(rgb)` | RGB 图 -> `[Detection(name, conf, xyxy, distance?, xyz_cam?, xyz_world?)]` |
| `annotate(rgb, detections)` | 画框 + 标签 + 距离,返回 RGB |

- `set_classes` 支持任意文本类别;配置里 `classes` 省略时自动用场景物体名
- 小模型对纯色几何体召回一般:换 `yolov8m-worldv2.pt`、类别写同义词
  (`box, red box`)可明显改善

## measure.py —— 深度测距

`measure_detections(detections, camera, depth=None)` 就地给每个 Detection 填:

| 字段 | 含义 |
|---|---|
| `distance` | bbox 深度中位数反投影的直线距离(米) |
| `xyz_cam` | 相机系坐标(OpenCV 惯例,x 右 y 下 z 前) |
| `xyz_world` | 世界系坐标(经相机外参变换) |

深度测的是**可见表面**,比质心近半个物体尺寸——属正常现象(与真实深度相机一致)。
