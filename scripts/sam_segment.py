"""sam_segment.py — 传入名称 -> SAM3 概念分割 mask -> 距离 + 物体质心/可见尺寸。

流程:
  1. 从 MuJoCo 相机渲染一帧 RGB(+ 深度)
  2. SAM3SemanticPredictor 用文本提示做概念分割,得到 mask
  3. 对每个 mask:
       - depth[mask] 取中位数 -> 在 mask 质心像素反投影 -> X/Y/Z/R(距离)
       - mask 全部像素各自反投影成 3D 点云 -> 空间包围盒 -> 物体可见尺寸
  4. 保存 mask 叠加 + 距离/尺寸的可视化,返回结果

相当于 detect_and_distance.py 的「mask 精修版」:mask 紧贴物体,测距比 bbox 更准。

用法:
  python sam_segment.py "cube"   # 名称作为命令行参数
  python sam_segment.py          # 运行后交互输入名称
"""
from __future__ import annotations

import os
import sys
import tempfile

import cv2
import numpy as np
import mujoco
from ultralytics.models.sam import SAM3SemanticPredictor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.depth import get_intrinsics

XML = "/data/camera/Mujoco/scene.xml"
CAMERA = "cam"
# sam3.pt 是 gated 权重(https://huggingface.co/facebook/sam3,需申请),不会自动下载。
# 拿到后:export SAM3_WEIGHTS=/path/to/sam3.pt  或直接放在当前目录。
MODEL = os.environ.get("SAM3_WEIGHTS", "sam3.pt")
WIDTH, HEIGHT = 640, 480
STEPS = 50
CONF = 0.25
OUT = "/data/camera/Mujoco/scripts/sam_segment.png"


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


def mask_metrics(depth, mask, K):
    """由单个 mask 算距离与可见尺寸。无有效深度返回 None。

    返回 {centroid_pix, xyz_m, range_m, z_med_m, size_m=(sx,sy,sz)}(米)。
    size 是该 mask 可见表面的 3D 包围盒范围(单视图看不到的背面无法恢复)。
    """
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    z_all = depth[ys, xs]
    valid = np.isfinite(z_all) & (z_all > 0)
    if not valid.any():
        return None
    xs, ys, z_all = xs[valid], ys[valid], z_all[valid]

    z_med = float(np.median(z_all))
    cu, cv = float(xs.mean()), float(ys.mean())
    Xc = (cu - K["cx"]) * z_med / K["fx"]
    Yc = (cv - K["cy"]) * z_med / K["fy"]
    Zc = z_med

    # mask 点云的 3D 包围盒 -> 可见尺寸
    xx = (xs - K["cx"]) * z_all / K["fx"]
    yy = (ys - K["cy"]) * z_all / K["fy"]
    sx = float(xx.max() - xx.min())
    sy = float(yy.max() - yy.min())
    sz = float(z_all.max() - z_all.min())

    return {
        "centroid_pix": (cu, cv),
        "xyz_m": (Xc, Yc, Zc),
        "range_m": float(np.sqrt(Xc * Xc + Yc * Yc + Zc * Zc)),
        "z_med_m": z_med,
        "size_m": (sx, sy, sz),
    }


def segment_and_measure(name):
    """用 SAM3 概念分割 `name`,返回 [{mask, metrics}, ...]。"""
    rgb, depth, K = render()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    overrides = {"conf": CONF, "task": "segment", "mode": "predict",
                 "model": MODEL, "quantize": 16, "save": False, "verbose": False}
    predictor = SAM3SemanticPredictor(overrides=overrides)

    # set_image 用临时文件传路径(和文档用法一致,最稳)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = f.name
    try:
        cv2.imwrite(tmp, bgr)
        predictor.set_image(tmp)
        res = predictor(text=[name])
        res = res[0] if isinstance(res, (list, tuple)) else res
    finally:
        os.unlink(tmp)

    raw_masks = res.masks.data.cpu().numpy() if res.masks is not None else np.empty((0, 0, 0))
    H, W = depth.shape
    results = []
    for m in raw_masks:
        m2 = m.astype(bool)
        if m2.shape != (H, W):                       # 对齐到深度图分辨率
            m2 = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        results.append({"mask": m2, "metrics": mask_metrics(depth, m2, K)})

    _save_viz(rgb, results, name)
    return results


def _save_viz(rgb, results, name):
    vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    overlay = vis.copy()
    for r in results:
        overlay[r["mask"]] = (0, 255, 0)             # 绿色填充 mask
    vis = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)

    for i, r in enumerate(results):
        mt = r["metrics"]
        if not mt:
            continue
        cu, cv = mt["centroid_pix"]
        X, Y, Z = mt["xyz_m"]
        sx, sy, sz = mt["size_m"]
        cv2.drawMarker(vis, (int(cu), int(cv)), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(vis, f"R={mt['range_m']:.2f}m  size={sx*100:.0f}x{sy*100:.0f}x{sz*100:.0f}cm",
                    (10, 30 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(OUT, vis)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else input("输入要分割的概念名称: ").strip()
    if not name:
        print("名称为空,退出。")
        return []
    results = segment_and_measure(name)
    print(f"\nSAM3 概念分割 '{name}':{len(results)} 个 mask")
    for i, r in enumerate(results):
        mt = r["metrics"]
        if not mt:
            print(f"  [{i}] 无有效深度")
            continue
        x, y, z = mt["xyz_m"]
        sx, sy, sz = mt["size_m"]
        print(f"  [{i}] X={x:.2f} Y={y:.2f} Z={z:.2f}  R={mt['range_m']:.2f} m   "
              f"可见尺寸={sx*100:.0f}x{sy*100:.0f}x{sz*100:.0f} cm")
    print(f"可视化已保存:{OUT}")
    return results


if __name__ == "__main__":
    main()
