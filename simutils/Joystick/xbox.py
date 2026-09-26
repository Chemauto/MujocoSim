"""XboxJoystick：通过 pygame.joystick 读取真实手柄（自动识别 PS4 / Xbox / 其他）。

Linux 下 PS4(DualShock/DualSense) 和 Xbox 手柄的轴/按键布局相同，
自动检测后打印手柄名称和类型，支持 config 覆盖轴/按键映射。
启动后短暂忽略按键（防抖），避免手柄连接瞬间的虚假信号。
"""
import time

import pygame

from .base import Joystick, JoystickState

# ── 轴映射（PS4 / Xbox 通用：axis 0/1=左摇杆，2=L2，3/4=右摇杆，5=R2）──
DEFAULT_AXES = {
    0: "left_x",
    1: "left_y",
    2: "l2",
    3: "right_x",
    4: "right_y",
    5: "r2",
}

# ── 按键映射 ──
# Xbox:  0=A  1=B  2=X  3=Y  4=LB 5=RB 6=Back 7=Start 8=Guide 9=L3 10=R3
# PS4:   0=×  1=○  2=□  3=△  4=L1 5=R1 6=Share 7=Options 8=PS 9=L3 10=R3
DEFAULT_BUTTONS = {
    0: "a",          # Xbox A / PS4 Cross(×)
    1: "b",          # Xbox B / PS4 Circle(○)
    2: "x",          # Xbox X / PS4 Square(□)
    3: "y",          # Xbox Y / PS4 Triangle(△)
    4: "lb",         # LB / L1
    5: "rb",         # RB / R1
    6: "back",       # Back / Share
    7: "start",      # Start / Options
    8: "guide",      # Guide / PS
    9: "lstick",     # L3 (摇杆按下)
    10: "rstick",    # R3 (摇杆按下)
}

# Dpad: hat 0
HAT_TO_DPAD = {
    (0, 1): "dpad_up",
    (0, -1): "dpad_down",
    (-1, 0): "dpad_left",
    (1, 0): "dpad_right",
}

# 手柄名称关键字 → 类型
_CONTROLLER_TYPES = {
    "sony": "PS4/PS5",
    "dualshock": "PS4",
    "dualsense": "PS5",
    "wireless controller": "PS4/PS5",
    "xbox": "Xbox",
    "x-box": "Xbox",
    "microsoft": "Xbox",
}


def _detect_type(name: str) -> str:
    """根据手柄名称推断类型。"""
    lower = name.lower()
    for keyword, ctype in _CONTROLLER_TYPES.items():
        if keyword in lower:
            return ctype
    return "通用"


class XboxJoystick(Joystick):
    """读取真实手柄（通过 pygame.joystick，自动识别类型）。"""

    def __init__(self, device_index=0, axis_map=None, button_map=None, warmup_sec=0.5):
        self._device_index = device_index
        self._axis_map = axis_map or dict(DEFAULT_AXES)
        self._button_map = button_map or dict(DEFAULT_BUTTONS)
        self._state = JoystickState()
        self._joy = None
        self._name = ""
        self._type = ""
        self._warmup_sec = warmup_sec
        self._warmup_until = 0.0  # 连接后忽略按键的时间戳
        self._connect_time = 0.0
        self._frame = 0

    def _ensure_init(self):
        if self._joy is not None:
            return
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("未检测到手柄，请检查连接（USB/蓝牙）")
        self._joy = pygame.joystick.Joystick(self._device_index)
        self._joy.init()
        self._name = self._joy.get_name()
        self._type = _detect_type(self._name)
        self._connect_time = time.time()
        self._warmup_until = self._connect_time + self._warmup_sec
        self._print_info()
        if self._warmup_sec > 0:
            print(f"  ⏳ 启动防抖：前 {self._warmup_sec} 秒忽略按键（避免虚假信号）")

    def connect(self) -> None:
        """连接手柄并打印信息。"""
        self._ensure_init()
        self._print_info()

    def _print_info(self):
        j = self._joy
        print(f"🎮 手柄已连接: {self._name}")
        print(f"   类型: {self._type}  |  轴: {j.get_numaxes()}  按键: {j.get_numbuttons()}  Hat: {j.get_numhats()}")
        print()
        print("   输入 → 状态轴/按键:")
        labels = {
            "left_x": "左摇杆 左/右", "left_y": "左摇杆 前/后",
            "right_x": "右摇杆 左/右", "right_y": "右摇杆 前/后",
            "l2": "L2 扳机", "r2": "R2 扳机",
        }
        for axis_idx, name in sorted(self._axis_map.items()):
            print(f"     axis {axis_idx} → {labels.get(name, name)}")
        print(f"     Hat 0 → Dpad (上/下/左/右)")
        btn_labels = {
            "a": "A/×(Cross)", "b": "B/○(Circle)", "x": "X/□(Square)", "y": "Y/△(Triangle)",
            "lb": "LB/L1", "rb": "RB/R1",
            "back": "Back/Share", "start": "Start/Options", "guide": "Guide/PS",
            "lstick": "L3(摇杆按下)", "rstick": "R3(摇杆按下)",
        }
        for btn_idx, name in sorted(self._button_map.items()):
            print(f"     btn {btn_idx} → {btn_labels.get(name, name)}")
        print()

    def update(self) -> None:
        self._ensure_init()
        for event in pygame.event.get():
            pass
        self._frame += 1

        # 启动防抖：连接后 warmup_sec 内忽略所有按键（避免虚假信号）
        in_warmup = time.time() < self._warmup_until

        # 读轴
        for axis_idx, name in self._axis_map.items():
            if axis_idx < self._joy.get_numaxes():
                raw = self._joy.get_axis(axis_idx)
                if name in ("l2", "r2"):
                    # 扳机：-1(松)~1(按) → 0~1
                    self._state.axes[name] = max(0.0, (raw + 1.0) / 2.0)
                else:
                    # 摇杆：Y 轴取反（驱动上推=负，约定上推=正），X 轴原样
                    val = -raw if name.endswith("_y") else raw
                    self._state.axes[name] = val if abs(val) > 0.08 else 0.0

        # 读按键（warmup 期间忽略，避免连接瞬间虚假信号）
        if not in_warmup:
            for btn_idx, name in self._button_map.items():
                if btn_idx < self._joy.get_numbuttons():
                    self._state.buttons[name] = bool(self._joy.get_button(btn_idx))

        # 读 Dpad (hat)
        for name in ("dpad_up", "dpad_down", "dpad_left", "dpad_right"):
            self._state.buttons[name] = False
        if self._joy.get_numhats() > 0:
            hat = self._joy.get_hat(0)
            for offset, name in HAT_TO_DPAD.items():
                if hat == offset:
                    self._state.buttons[name] = True

    def get_state(self) -> JoystickState:
        return self._state.copy()

    def close(self) -> None:
        if self._joy is not None:
            self._joy.quit()
            self._joy = None


if __name__ == "__main__":
    """独立运行：连接手柄，打印状态。"""
    js = XboxJoystick()
    js.connect()
    last = None
    clock = pygame.time.Clock()
    print("操作手柄查看状态，Ctrl+C 退出")
    try:
        while True:
            js.update()
            st = js.get_state()
            pressed = [k for k, v in st.buttons.items() if v]
            line = (
                f"L({st.axes['left_x']:+.2f},{st.axes['left_y']:+.2f}) "
                f"R({st.axes['right_x']:+.2f},{st.axes['right_y']:+.2f}) "
                f"L2={st.axes['l2']:.2f} R2={st.axes['r2']:.2f} "
                f"btn={pressed or '-'}"
            )
            if line != last:
                print(f"\r{line:<90}", end="", flush=True)
                last = line
            clock.tick(30)
    except KeyboardInterrupt:
        pass
    finally:
        js.close()
        print("\n已退出")
