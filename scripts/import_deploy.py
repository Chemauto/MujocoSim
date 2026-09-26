from __future__ import annotations

import argparse
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

OBS_SOURCE_MAP = {
    "base_ang_vel": "gyro",
    "projected_gravity": "projected_gravity",
    "velocity_commands": "command",
    "joint_pos_rel": "joint_qpos",
    "joint_vel_rel": "joint_qvel",
    "last_action": "last_action",
}


def _fmt(value) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(str(float(v)) for v in value) + "]"
    return str(float(value))


def _clip_pair(raw) -> list[float] | None:
    if raw is None:
        return None
    if isinstance(raw[0], (list, tuple)):
        los = [float(p[0]) for p in raw]
        his = [float(p[1]) for p in raw]
        return [min(los), max(his)]
    return [float(raw[0]), float(raw[1])]


def convert(deploy: dict, meta: dict, name: str, model: str) -> dict:
    joint_names = meta["joint_names"]
    joint_ids_map = deploy.get("joint_ids_map") or list(range(len(joint_names)))
    joint_order = [joint_names[i] for i in joint_ids_map]

    default_pose = [float(v) for v in deploy.get("default_joint_pos", [])]
    if len(default_pose) != len(joint_names):
        raise SystemExit(
            f"default_joint_pos 长度 {len(default_pose)} != {len(joint_names)}"
        )

    obs_lines = []
    cursor = 0
    for obs_name, term in deploy.get("observations", {}).items():
        source = OBS_SOURCE_MAP.get(obs_name)
        if source is None:
            raise SystemExit(f"观测项 '{obs_name}' 暂无对应引脚源（需要手工接线）")
        params = {}
        if source == "command":
            params["components"] = ["vx", "vy", "yaw_rate"]
        if source == "joint_qpos":
            params["offset"] = default_pose
        scale = term.get("scale", 1.0)
        clip = _clip_pair(term.get("clip"))
        history = int(term.get("history_length", 1))
        dim = len(scale) if isinstance(scale, list) else None
        if source in ("gyro", "projected_gravity"):
            dim = 3
        elif source == "command":
            dim = 3
        elif source in ("joint_qpos", "joint_qvel", "last_action"):
            dim = len(joint_names)
        if dim is None:
            raise SystemExit(f"观测项 '{obs_name}' 维度无法推导")
        span = [cursor, cursor + dim * history]
        cursor = span[1]
        pin = {"source": source, "span": span}
        if params:
            pin["params"] = params
        if not (isinstance(scale, list) and all(float(v) == 1.0 for v in scale)):
            pin["scale"] = (
                [float(v) for v in scale] if isinstance(scale, list) else float(scale)
            )
        if clip is not None:
            pin["clip"] = clip
        if history > 1:
            pin["history"] = history
        obs_lines.append(pin)

    action_term = deploy.get("actions", {}).get("JointPositionAction", {})
    action_scale = action_term.get("scale", 1.0)
    action_offset = action_term.get("offset", default_pose)
    control = {
        "kp": [float(v) for v in deploy.get("stiffness", [25.0] * len(joint_names))],
        "kd": [float(v) for v in deploy.get("damping", [0.5] * len(joint_names))],
    }

    return {
        "model": model,
        "input_name": "obs",
        "output_name": "actions",
        "joint_order": joint_order,
        "observations": obs_lines,
        "actions": {
            "mode": "position_offset",
            "scale": [float(v) for v in action_scale]
            if isinstance(action_scale, list)
            else float(action_scale),
            "offset": [float(v) for v in action_offset]
            if isinstance(action_offset, list)
            else float(action_offset),
            "control": control,
            "mapping": "auto",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="unitree_rl_lab deploy.yaml → policies.yaml"
    )
    parser.add_argument("deploy", help="deploy.yaml 路径")
    parser.add_argument("--robot", required=True, help="go2 / g1 / so101")
    parser.add_argument("--name", required=True, help="策略名（如 velocity）")
    parser.add_argument("--model", default="policy/policy.onnx", help="模型相对路径")
    parser.add_argument(
        "--out",
        default=None,
        help="输出 policies.yaml（默认 rl_controller/robot/<robot>/policies.yaml）",
    )
    args = parser.parse_args()

    deploy = yaml.safe_load(Path(args.deploy).read_text(encoding="utf-8"))
    meta_path = REPO_ROOT / "assets" / "robots" / args.robot / "robot.yaml"
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    profile = convert(deploy, meta, args.name, args.model)

    out = (
        Path(args.out)
        if args.out
        else REPO_ROOT / "rl_controller" / "robot" / args.robot / "policies.yaml"
    )
    data = {"active": args.name, "policies": {}}
    if out.is_file():
        data = yaml.safe_load(out.read_text(encoding="utf-8")) or data
        data.setdefault("policies", {})
    data["policies"][args.name] = profile
    data["active"] = args.name
    out.write_text(
        yaml.safe_dump(
            data, sort_keys=False, allow_unicode=True, default_flow_style=None
        ),
        encoding="utf-8",
    )
    n_obs = profile["observations"][-1]["span"][1]
    print(
        f"[import] {args.name}: obs_dim={n_obs}  joints={len(profile['joint_order'])}"
    )
    print(f"[import] 写入 {out}")


if __name__ == "__main__":
    main()
