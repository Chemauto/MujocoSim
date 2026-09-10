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

from common import load_config
from rl_control import RobotController
from sensors import HeightScanner, depth_colormap, imwrite_u
from simulation import SimWorld
from common.data_bus import DataExporter
from yolo import YoloWorldDetector, measure_detections, resolve_weights

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


def build_controller(world, cfg):
    """组合根:按配置装配 控制器 + 高度扫描仪,并把控制钩子挂到 world。"""
    if cfg.controller is None:
        return None
    hs = cfg.controller.height_scan
    scanner = HeightScanner(world.model, world.data,
                            base_body=world.robot_meta["base_body"],
                            size=tuple(hs.get("size", [1.6, 1.0])),
                            resolution=hs.get("resolution", 0.1),
                            ray_offset_z=hs.get("ray_offset_z", 20.0),
                            height_offset=hs.get("height_offset", 0.5),
                            geomgroup=hs.get("geomgroup", (1, 1, 0, 0, 0, 0)),
                            max_bounces=hs.get("max_bounces", 32),
                            robot_body_ids=world.robot.body_ids())
    controller = RobotController(world.robot, cfg.controller, height_scanner=scanner)
    world.pre_step = controller.compute           # 每步 mj_step 前算力矩
    controller.reset_robot(world)                 # 站姿出生
    return controller


def save_outputs(rgb, depth, dets, detector, save_dir: Path) -> None:
    """把已经检测好的一帧(rgb/深度/检测列表)存盘。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    imwrite_u(str(save_dir / "rgb.png"), rgb[..., ::-1])
    imwrite_u(str(save_dir / "depth.png"), depth_colormap(depth))
    if detector is not None:
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
    try:
        weights = resolve_weights(cfg.yolo.weights)
        det = YoloWorldDetector(weights, cfg.yolo_classes(), conf=cfg.yolo.conf)
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
    from sensors import SimCamera
    world = SimWorld(cfg, camera_factory=lambda m, d, name, w, h: SimCamera(
        m, d, name, width=w, height=h))
    controller = build_controller(world, cfg)     # 控制器(配置缺 controller 段则为 None)
    world.exporter = DataExporter(cfg.data, world, controller,
                                  controller.scanner if controller else None)
    print(f"[sim] 控制模式: {controller.mode if controller else '无(自由运动)'}")

    detector = make_detector(cfg)
    last_detect = -1e9
    save_dir = PROJECT_ROOT / cfg.yolo.save_dir

    if args.oneshot:                           # 无窗口:跑一段、检测一次、退出
        try:
            for _ in range(int(args.warmup / cfg.timestep)):
                world.step()
            if detector is not None and cfg.yolo.camera in world.cameras:
                cam = world.cameras[cfg.yolo.camera]
                rgb, depth = cam.render_rgbd()
                dets = measure_detections(detector.detect(rgb), cam, depth)
                report(world, dets)
                save_outputs(rgb, depth, dets, detector, save_dir)
        finally:
            world.close()
        return 0

    if not args.no_viewer:
        world.open_viewer()
    print("[sim] 运行中(Ctrl+C 或关窗口退出)…")

    def on_update(w: SimWorld) -> None:
        nonlocal last_detect
        if detector is None or w.cfg.yolo.camera not in w.cameras:
            return
        if w.sim_time - last_detect < w.cfg.yolo.interval:
            return
        last_detect = w.sim_time
        cam = w.cameras[w.cfg.yolo.camera]
        try:
            rgb, depth = cam.render_rgbd()
            dets = measure_detections(detector.detect(rgb), cam, depth)
            report(w, dets)
            save_dir.mkdir(parents=True, exist_ok=True)
            imwrite_u(str(save_dir / "latest_annotated.png"),
                      detector.annotate(rgb, dets)[..., ::-1])
        except Exception as e:                     # 单帧检测/存图异常不终止仿真
            print(f"[yolo] 本帧检测失败(跳过): {e!r}")

    try:
        world.run(on_update=on_update)
    except KeyboardInterrupt:
        print("\n[sim] 已停止")
    finally:
        world.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
