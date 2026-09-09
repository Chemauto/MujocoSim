"""main.py —— 主程序:配置驱动的 MuJoCo 仿真 + YOLO 检测 + 深度测距。

用法:
  python main.py                              # 默认 configs/go2.yaml,交互 viewer
  python main.py --config configs/g1.yaml     # 换机器人/场景,只改配置
  python main.py --robot g1                   # 命令行覆盖机器人
  python main.py --oneshot                    # 无窗口跑一次检测就退出(调试/测试用)
  python main.py --no-yolo                    # 只看仿真

改机器人、相机位置/数量、物体、YOLO 类别:全部在 configs/*.yaml,不动代码。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # 支持从任意目录启动

from simulation import SimWorld, load_config
from sensors.camera import depth_colormap, imwrite_u
from yolo import YoloWorldDetector, measure_detections

PROJECT_ROOT = Path(__file__).resolve().parent


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="MuJoCo + YOLO 通用仿真平台")
    ap.add_argument("-c", "--config", default=None, help="配置文件(默认 configs/go2.yaml)")
    ap.add_argument("--robot", default=None, help="覆盖配置里的机器人名(assets/robots 下的文件夹名)")
    ap.add_argument("--oneshot", action="store_true", help="无窗口:步进一段时间后检测一次并退出")
    ap.add_argument("--warmup", type=float, default=1.0, help="oneshot 模式步进秒数(默认 1)")
    ap.add_argument("--no-yolo", action="store_true", help="不运行 YOLO 检测")
    ap.add_argument("--no-viewer", action="store_true", help="不开交互窗口(持续运行模式)")
    return ap


def save_outputs(world: SimWorld, detector, save_dir: Path) -> None:
    """把检测相机这一帧的 RGB/深度/标注图存盘。"""
    cam = world.cameras[world.cfg.yolo.camera]
    save_dir.mkdir(parents=True, exist_ok=True)
    rgb, depth = cam.render_rgbd()
    dets = []
    if detector is not None:
        dets = measure_detections(detector.detect(rgb), cam, depth)
    imwrite_u(str(save_dir / "rgb.png"), rgb[..., ::-1])
    imwrite_u(str(save_dir / "depth.png"), depth_colormap(depth))
    imwrite_u(str(save_dir / "annotated.png"), detector.annotate(rgb, dets)[..., ::-1])
    print(f"[save] 本帧图像已存到 {save_dir}")


def report(world: SimWorld, detections) -> None:
    pos, _ = world.robot_base_pose()
    print(f"t={world.sim_time:6.2f}s  {world.cfg.robot} base=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})", end="")
    if not detections:
        print("  |  未检测到目标")
        return
    print(f"  |  检测到 {len(detections)} 个目标:")
    for d in detections:
        line = f"    - {d.name} conf={d.conf:.2f}"
        if d.distance is not None:
            line += f"  距离={d.distance:.2f}m"
        if d.xyz_world is not None:
            line += f"  世界坐标=({d.xyz_world[0]:.2f},{d.xyz_world[1]:.2f},{d.xyz_world[2]:.2f})"
        print(line)


def make_detector(cfg):
    """按配置建 YOLO 检测器;权重不可用时降级为 None 并提示。"""
    if not cfg.yolo.enabled:
        return None
    weights = Path(cfg.yolo.weights)
    if not weights.is_absolute():
        weights = PROJECT_ROOT / weights
    try:
        det = YoloWorldDetector(str(weights), cfg.yolo_classes(), conf=cfg.yolo.conf)
        print(f"[yolo] 权重={weights}  类别={cfg.yolo_classes()}  conf={cfg.yolo.conf}")
        return det
    except Exception as e:                     # 权重缺失/下载失败不影响仿真本体
        print(f"[yolo] 初始化失败({e}),本次运行不启用检测")
        return None


def main() -> int:
    args = build_arg_parser().parse_args()
    cfg = load_config(args.config, robot=args.robot)
    if args.no_yolo:
        cfg.yolo.enabled = False

    print(f"[sim] 机器人={cfg.robot}  出生点={cfg.spawn or '默认'}  "
          f"相机={[c.name for c in cfg.cameras]}  物体={[o.name for o in cfg.objects]}")
    world = SimWorld(cfg)

    detector = make_detector(cfg)
    last_detect = -1e9
    save_dir = PROJECT_ROOT / cfg.yolo.save_dir

    if args.oneshot:                           # 无窗口:跑一段、检测一次、退出
        for _ in range(int(args.warmup / cfg.timestep)):
            world.step()
        if detector is not None and cfg.yolo.camera in world.cameras:
            cam = world.cameras[cfg.yolo.camera]
            rgb, depth = cam.render_rgbd()
            dets = measure_detections(detector.detect(rgb), cam, depth)
            report(world, dets)
            save_outputs(world, detector, save_dir)
        world.close()
        return 0

    if not args.no_viewer:
        world.open_viewer()
    print("[sim] 运行中(Ctrl+C 或关窗口退出)…")

    def on_update(w: SimWorld) -> None:
        nonlocal last_detect
        if detector is None or not w.cameras:
            return
        if w.sim_time - last_detect < w.cfg.yolo.interval:
            return
        last_detect = w.sim_time
        cam = w.cameras[w.cfg.yolo.camera]
        rgb, depth = cam.render_rgbd()
        dets = measure_detections(detector.detect(rgb), cam, depth)
        report(w, dets)
        save_dir.mkdir(parents=True, exist_ok=True)
        imwrite_u(str(save_dir / "latest_annotated.png"),
                  detector.annotate(rgb, dets)[..., ::-1])

    try:
        world.run(on_update=on_update)
    except KeyboardInterrupt:
        print("\n[sim] 已停止")
    finally:
        world.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
