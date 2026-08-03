"""detect_and_distance.py — 传入物体名称,返回它的 bbox 和距离。

流程:
  1. 从 MuJoCo 相机渲染一帧 RGB(+ 深度)
  2. YOLO-World(开放词表)识别指定名称的物体,得到 bbox
  3. 用深度 + 相机内参算出该 bbox 的距离

用法:
  python detect_and_distance.py "red cube"   # 名称作为命令行参数
  python detect_and_distance.py              # 运行后交互输入名称
"""
from __future__ import annotations

import math
import os
import sys

import cv2
import numpy as np
import mujoco
from ultralytics import YOLOWorld

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.depth import get_intrinsics

XML = "/data/camera/Mujoco/scene.xml"
WEIGHTS = "/home/xcj/work/YoloProject/yolov8m-world.pt"
CAMERA = "cam"
WIDTH, HEIGHT = 640, 480
STEPS = 50
CONF = 0.15
OUT = "/data/camera/Mujoco/scripts/detect_and_distance.png"


def render():
    """渲染一帧 RGB + 深度,返回 (rgb, depth, K)。"""
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA)
    for _ in range(STEPS):
        mujoco.mj_step(model, data)
    r = mujoco.Renderer(model, HEIGHT, WIDTH)
    r.update_scene(data, camera=CAMERA)
    rgb = r.render()
    r.enable_depth_rendering()
    depth = r.render()
    K = get_intrinsics(model, cam_id, WIDTH, HEIGHT)
    return rgb, depth, K


def bbox_distance(depth, xyxy, K):
    """bbox 的距离:取 bbox 区域深度的中位数,在中心像素反投影。

    返回 {xyxy, xyz_m=(x,y,z), range_m}(相机坐标系,米),无有效深度则返回 None。
    """
    x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(depth.shape[1], x2), min(depth.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    roi = depth[y1:y2, x1:x2]
    roi = roi[np.isfinite(roi) & (roi > 0)]
    if roi.size == 0:
        return None
    z = float(np.median(roi))
    cu, cv = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    x = (cu - K["cx"]) * z / K["fx"]
    y = (cv - K["cy"]) * z / K["fy"]
    return {"xyxy": (x1, y1, x2, y2), "xyz_m": (x, y, z),
            "range_m": math.sqrt(x * x + y * y + z * z)}


def detect_and_measure(name):
    """在 MuJoCo RGB 中检测 `name`,返回 [{class, conf, bbox, distance}, ...]。"""
    rgb, depth, K = render()
    yolo = YOLOWorld(WEIGHTS)
    yolo.set_classes([name])
    bgr = np.ascontiguousarray(rgb[..., ::-1])          # RGB -> BGR(ultralytics 约定)
    res = yolo.predict(bgr, conf=CONF, verbose=False)[0]

    out = []
    for b in res.boxes:
        xyxy = [float(v) for v in b.xyxy[0].tolist()]
        out.append({
            "class": name,
            "conf": float(b.conf[0]),
            "bbox": xyxy,
            "distance": bbox_distance(depth, xyxy, K),
        })

    _save_viz(res, out)
    return out


def _save_viz(res, results):
    vis = res.plot()                                     # BGR,已画好 bbox + 类别标签
    for r in results:
        d = r["distance"]
        if not d:
            continue
        x1, y1, _, _ = d["xyxy"]
        cv2.putText(vis, f"R={d['range_m']:.2f} m", (int(x1), max(15, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(OUT, vis)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else input("输入要识别的物体名称: ").strip()
    if not name:
        print("名称为空,退出。")
        return []
    results = detect_and_measure(name)
    print(f"\n检测 '{name}':找到 {len(results)} 个")
    for i, r in enumerate(results):
        d = r["distance"]
        if d:
            x, y, z = d["xyz_m"]
            print(f"  [{i}] conf={r['conf']:.2f}  bbox={[round(v, 1) for v in r['bbox']]}  "
                  f"-> X={x:.2f} Y={y:.2f} Z={z:.2f}  R={d['range_m']:.2f} m")
        else:
            print(f"  [{i}] conf={r['conf']:.2f}  bbox={r['bbox']}  -> 无有效深度")
    print(f"标注图已保存:{OUT}")
    return results


if __name__ == "__main__":
    main()
