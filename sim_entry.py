from __future__ import annotations

import argparse
import threading

import rclpy

from mujocosim.config import load_config
from mujocosim.controller import RobotController
from mujocosim.ros2_bridge import SimRos2Bridge
from mujocosim.sensors import SensorSuite
from mujocosim.world import SimWorld
from rl_controller import RlController, load_policies, policies_path, print_wiring_table


def main() -> None:
    parser = argparse.ArgumentParser(description="MujocoSim 仿真器")
    parser.add_argument("--config", default="configs/config.yaml", help="配置文件路径")
    parser.add_argument(
        "--robot", default=None, help="覆盖配置里的机器人（go2 / g1 / so101）"
    )
    parser.add_argument(
        "--policy", default=None, help="覆盖配置里的策略名（空串 = 不跑策略）"
    )
    parser.add_argument(
        "--mode",
        default=None,
        help="覆盖控制模式（damping / pd_stand / position / motion）",
    )
    parser.add_argument(
        "--command", default=None, help="速度指令，如 '0.3 0 0'（vx vy yaw_rate）"
    )
    parser.add_argument("--no-gui", action="store_true", help="不开窗口，纯后台步进")
    parser.add_argument(
        "--max-time", type=float, default=None, help="仿真秒数，到点退出"
    )
    args = parser.parse_args()

    cfg = load_config(args.config, robot=args.robot)
    if args.policy is not None:
        cfg.policy_name = args.policy or None
    elif args.robot is not None and cfg.policy_name:
        print(
            f"[sim] --robot {args.robot}: 策略按机器人走，忽略配置里的 '{cfg.policy_name}'"
        )
        cfg.policy_name = None
    world = SimWorld(cfg)
    suite = SensorSuite(cfg, world)

    rl = None
    if cfg.policy_name:
        profiles = load_policies(
            policies_path(cfg.robot), cfg.robot_meta, len(cfg.command.components)
        )
        if cfg.policy_name not in profiles:
            raise SystemExit(
                f"[sim] policies.yaml 里没有策略 '{cfg.policy_name}'，可选: {sorted(profiles)}"
            )
        profile = profiles[cfg.policy_name]
        print_wiring_table(profile, cfg.robot_meta)
        rl = RlController(
            profile,
            cfg.robot_meta,
            world,
            policies_path(cfg.robot).parent,
            len(cfg.command.components),
            suite=suite,
        )
        if args.command:
            rl.set_command([float(v) for v in args.command.split()])

    controller = RobotController(cfg, world)
    if rl is not None:
        controller.action_fn = rl
    if args.mode:
        controller.set_mode(args.mode)
    if controller.mode == "motion" and rl is not None:
        controller.set_gains(rl.kp, rl.kd)

    rclpy.init()
    bridge = SimRos2Bridge(
        cfg,
        world,
        controller,
        on_cmd_vel=rl.set_command if rl is not None else None,
        suite=suite,
    )
    spin_thread = threading.Thread(target=rclpy.spin, args=(bridge,), daemon=True)
    spin_thread.start()

    print(
        f"[sim] robot={cfg.robot} scene={cfg.scene} mode={controller.mode} "
        f"joints={len(cfg.robot_meta.joint_names)} sensors={len(cfg.mounts)}"
    )

    def control_step(w) -> None:
        controller.step()
        suite.step(w.sim_time())

    try:
        world.run(
            gui=not args.no_gui,
            max_sim_time=args.max_time,
            on_control_step=control_step,
        )
    finally:
        bridge.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    print(f"[sim] done, sim_time={world.sim_time():.2f}s")


if __name__ == "__main__":
    main()
