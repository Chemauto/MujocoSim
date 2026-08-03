# mujoco-yolo-camera

一个最小化的 MuJoCo 仿真——一个方块 + 一个相机——支持 RGB 与深度渲染、对渲染出的
RGB 做开放词表检测(YOLO-World)与 SAM3 概念分割,以及点击像素或传入物体名称由深度
反算距离。结构参考 [`Carema`](../Carema)(Orbbec Astra)项目:深度算法放 `utils/`,
可运行的示例作为脚本。

## 目录结构

```
.
├── scene.xml            # 地板 + 方块(freejoint)+ 一个对准方块的相机
├── main.py              # 打开 MuJoCo 仿真器窗口,运行物理
├── camera_viewer.py     # 渲染并查看相机的 RGB + 深度
├── scripts/
│   ├── get_distance.py          # 打开 RGB,点击像素 -> 由深度得到距离
│   ├── detect_and_distance.py   # 传入名称 -> YOLO bbox + 该 bbox 的距离
│   └── sam_segment.py           # 传入名称 -> SAM3 mask + 距离 + 物体可见尺寸
├── utils/
│   └── depth.py         # 深度工具:内参、反投影、伪彩图
└── yolo/
    └── detect.py        # 渲染 RGB -> YOLO-World 检测(+ 相机内参)
```

## 依赖

Python 3.10+:

```bash
pip install mujoco opencv-python matplotlib numpy ultralytics
```

首次运行 YOLO-World 时会自动下载其权重(`*.pt`)和 CLIP 文本编码器(约 338 MB,
下载到 `yolo/weights/`),都已 git-忽略,见 `.gitignore`。
SAM3 权重 `sam3.pt` 是 gated 模型,不会自动下载,需自行申请(见下文「说明」)。

## 运行

```bash
python main.py                 # 1) 仿真器窗口(需要图形界面)
python camera_viewer.py        # 2) 相机 RGB + 深度(保存 shot.png)
python scripts/get_distance.py # 3) 点击测距(需要图形界面)
cd yolo && python detect.py    # 4) 对渲染的 RGB 做 YOLO-World 检测(保存 detect.png)
python scripts/detect_and_distance.py "red cube"  # 5) 传入名称 -> bbox + 距离(保存 detect_and_distance.png)
python scripts/sam_segment.py "cube"              # 6) SAM3 概念分割 -> mask + 距离 + 尺寸(保存 sam_segment.png)
```

## 距离的计算方式

对像素 `(u, v)` 处的深度值 `z` 做针孔反投影:

```
x = (u - cx) * z / fx
y = (v - cy) * z / fy
R = sqrt(x^2 + y^2 + z^2)     # 到相机的欧氏距离
```

内参由相机的垂直视场角得到:`fx = fy = H / (2 * tan(fovy / 2))`,`cx = W/2`,
`cy = H/2`。详见 `utils/depth.py`。

- `detect_and_distance.py`:取 **bbox** 区域深度的中位数作为 `z`,在 bbox 中心反投影,
  得到该物体的 `X/Y/Z/R`。
- `sam_segment.py`:用 **SAM3 mask** 紧贴物体,深度中位数更纯净,`X/Y/Z/R` 更准;
  另把 mask 全像素反投影成 3D 点云取包围盒,得到物体的**可见尺寸**(单视图无法恢复背面)。

## 说明

- GUI 脚本(`main.py`、`scripts/get_distance.py`)需要图形界面;其余脚本离屏渲染,
  无头环境也能跑。
- `scene.xml` 用 `mode="targetbody"`,相机会始终对准方块。
- 全程单位为米(MuJoCo 默认米制)。
- `sam_segment.py` 需手动获取 `sam3.pt`:到 [facebook/sam3](https://huggingface.co/facebook/sam3)
  申请访问并下载,再用 `SAM3_WEIGHTS=/path/to/sam3.pt python scripts/sam_segment.py "cube"` 运行。
