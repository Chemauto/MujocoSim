#!/usr/bin/env python3
"""虚拟 Xbox 手柄：运行即显示手柄窗口，鼠标操作，实时打印状态。

用法:
  python simutils/Joystick/virtual_gamepad.py          # 虚拟手柄（默认）
  python simutils/Joystick/virtual_gamepad.py --xbox    # 真实 Xbox 手柄

鼠标操作说明:
  左/右摇杆 → 拖拽圆盘中的小圆
  A/B/X/Y/LB/RB/Start/Back → 点击对应按键
  Dpad → 点击上下左右箭头
  L2/R2 → 在扳机区域内按住拖拽
"""
import argparse
import sys
from pathlib import Path

# 确保 simutils 可 import
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simutils.Joystick import make_joystick, state_to_joint_vel, JointMapping

import pygame


def main():
    parser = argparse.ArgumentParser(description="虚拟/真实 Xbox 手柄演示")
    parser.add_argument("--xbox", action="store_true", help="使用真实 Xbox 手柄（默认虚拟）")
    parser.add_argument("--max-vel", type=float, default=30.0, help="摇杆满杆关节速度（度/秒）")
    args = parser.parse_args()

    kind = "xbox" if args.xbox else "sim"
    js = make_joystick(kind)
    mapping = JointMapping(max_vel=args.max_vel)

    clock = pygame.time.Clock()
    last_line = None
    print(f"手柄已启动（{kind}），鼠标操作，Ctrl+C 退出\n")
    try:
        while True:
            js.update()
            st = js.get_state()
            vel, actions = state_to_joint_vel(st, mapping)

            # 只在状态变化时打印
            pressed_btns = [k for k, v in st.buttons.items() if v]
            active_acts = [k for k, v in actions.items() if v]
            line = (
                f"L({st.axes['left_x']:+.2f},{st.axes['left_y']:+.2f}) "
                f"R({st.axes['right_x']:+.2f},{st.axes['right_y']:+.2f}) "
                f"L2={st.axes['l2']:.2f} R2={st.axes['r2']:.2f} "
                f"vel={np.round(vel,1)} "
                f"btn={pressed_btns or '-'} act={active_acts or '-'}"
            )
            if line != last_line:
                print(f"\r{line:<100}", end="", flush=True)
                last_line = line

            clock.tick(30)
    except KeyboardInterrupt:
        pass
    finally:
        js.close()
        print("\n已退出")


if __name__ == "__main__":
    import numpy as np
    main()
