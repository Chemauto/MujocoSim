from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml

ACTIVATIONS = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "selu": nn.SELU}


def build_actor(state_dict: dict, activation: str) -> nn.Sequential:
    weights = sorted(
        (k for k in state_dict if k.startswith("actor.") and k.endswith(".weight")),
        key=lambda k: int(k.split(".")[1]),
    )
    layers: list[nn.Module] = []
    act_cls = ACTIVATIONS[activation]
    for i, key in enumerate(weights):
        w = state_dict[key]
        layers.append(nn.Linear(w.shape[1], w.shape[0]))
        if i < len(weights) - 1:
            layers.append(act_cls())
    actor = nn.Sequential(*layers)
    actor_state = {
        k.removeprefix("actor."): v
        for k, v in state_dict.items()
        if k.startswith("actor.")
    }
    actor.load_state_dict(actor_state)
    actor.eval()
    return actor


def main() -> None:
    parser = argparse.ArgumentParser(description="RSL-RL checkpoint → policy.onnx")
    parser.add_argument("checkpoint", help="model_XXXX.pt 路径")
    parser.add_argument(
        "--agent", default=None, help="agent.yaml（默认取 checkpoint 同级 params/）"
    )
    parser.add_argument("--out", default="policy.onnx")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    agent_path = (
        Path(args.agent) if args.agent else ckpt_path.parent / "params" / "agent.yaml"
    )
    agent = yaml.safe_load(agent_path.read_text(encoding="utf-8"))
    activation = str(agent.get("policy", {}).get("activation", "elu")).lower()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    actor = build_actor(state_dict, activation)

    in_layer = actor[0]
    out_layer = actor[-1]
    assert isinstance(in_layer, nn.Linear) and isinstance(out_layer, nn.Linear)
    in_dim = int(in_layer.in_features)
    dummy = torch.zeros(1, in_dim)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        actor,
        (dummy,),
        str(out_path),
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
        opset_version=17,
    )
    print(
        f"[export] {ckpt_path.name} -> {out_path}  "
        f"obs[{in_dim}] -> actions[{int(out_layer.out_features)}]"
    )


if __name__ == "__main__":
    main()
