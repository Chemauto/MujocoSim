"""policy.py —— TorchScript 策略加载与推理(与具体机器人解耦)。

加载 llm-legged-lab 导出的 actor(TorchScript):
  输入 = 观测向量(拼接顺序由 controller 配置的 obs_order 决定)
  输出 = 原始动作向量,controller 侧再做 offset/scale/clip

注意:
  - 相对路径按项目根解析;torch 的 C 层 fopen 打不开含中文的路径,
    因此用字节流加载(BytesIO)。
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class PolicyRunner:
    """一个导出策略 = 一个 PolicyRunner。"""

    def __init__(self, path: str):
        p = Path(path)
        if not p.is_absolute():
            p = _PROJECT_ROOT / p                    # 相对路径按项目根解析
        if not p.is_file():
            raise FileNotFoundError(f"策略文件不存在: {p}")
        import torch                                 # 延迟导入,不用策略零开销
        self.path = p
        self._torch = torch
        buf = io.BytesIO(p.read_bytes())             # 绕开中文路径 fopen 限制
        self.model = torch.jit.load(buf, map_location="cpu")
        self.model.eval()

    def infer(self, obs: np.ndarray) -> np.ndarray:
        """obs float32[232] -> 原始动作 float64[12]。"""
        torch = self._torch
        with torch.no_grad():
            out = self.model(torch.from_numpy(np.ascontiguousarray(obs, dtype=np.float32))
                             .unsqueeze(0))
        return out.squeeze(0).cpu().numpy().astype(np.float64)
