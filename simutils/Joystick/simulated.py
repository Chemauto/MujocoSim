"""SimulatedJoystick：pygame 窗口模拟 Xbox 手柄，鼠标操作。

窗口 950×650，画出手柄主体轮廓+握把，更像真实 Xbox 手柄。
独立运行 `python simutils/Joystick/simulated.py` 可看状态。
"""
import pygame

from .base import Joystick, JoystickState

# ── 配色（Xbox 风格）──
BG            = (0, 0, 0)
BODY          = (38, 38, 52)
BODY_BORDER   = (58, 58, 78)
GRIP          = (32, 32, 46)
STICK_BG      = (28, 28, 40)
STICK_KNOB    = (85, 85, 110)
STICK_ACTIVE  = (40, 220, 120)
BTN_NORMAL    = (55, 55, 75)
BTN_PRESSED   = (90, 90, 120)
DPAD_COLOR    = (50, 50, 68)
WHITE         = (230, 230, 240)
GRAY          = (120, 120, 145)
DIM           = (75, 75, 95)
ABXY_COLORS   = {"a": (30, 200, 80), "b": (210, 55, 55), "x": (55, 110, 220), "y": (220, 190, 30)}
TRIGGER_ON    = {"l2": (220, 130, 40), "r2": (40, 200, 110)}

# ── 控件列表 ──
# 每项：(name, kind, ...)
CONTROLS = [
    # 摇杆：(name, "stick", cx, cy, outer_r, knob_r)
    ("left_stick",  "stick", 240, 280, 72, 25),
    ("right_stick", "stick", 660, 380, 72, 25),
    # ABXY：(name, "btn", cx, cy, r, color_key)
    ("a", "btn", 755, 275, 24, "a"),
    ("b", "btn", 805, 225, 24, "b"),
    ("x", "btn", 705, 225, 24, "x"),
    ("y", "btn", 755, 175, 24, "y"),
    # Dpad：(name, "btn", cx, cy, r)
    ("dpad_up",    "dpad", 180, 365, 20),
    ("dpad_down",  "dpad", 180, 435, 20),
    ("dpad_left",  "dpad", 145, 400, 20),
    ("dpad_right", "dpad", 215, 400, 20),
    # 肩键：(name, "bumper", x, y, w, h)
    ("lb", "bumper", 240, 108, 95, 36),
    ("rb", "bumper", 615, 108, 95, 36),
    # 功能键：(name, "btn", cx, cy, r)
    ("back",  "btn", 420, 215, 16, None),
    ("start", "btn", 530, 215, 16, None),
    ("guide", "btn", 475, 150, 24, None),
    # 扳机：(name, "trigger", cx, cy, w, h)
    ("l2", "trigger", 195, 52, 48, 95),
    ("r2", "trigger", 757, 52, 48, 95),
]

DEADZONE = 0.06


class SimulatedJoystick(Joystick):
    """pygame 窗口绘制 Xbox 手柄，鼠标拖拽/点击操作。"""

    def __init__(self, size=(950, 650)):
        self._size = size
        self._screen = None
        self._state = JoystickState()
        self._dragging = None
        self._font = None
        self._font_sm = None

    def _ensure_init(self):
        if self._screen is not None:
            return
        pygame.init()
        self._screen = pygame.display.set_mode(self._size)
        pygame.display.set_caption("Xbox Controller (Simulated)")
        self._font = pygame.font.SysFont(None, 22)
        self._font_sm = pygame.font.SysFont(None, 17)

    def update(self) -> None:
        self._ensure_init()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._on_mouse_down(event.pos)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self._on_mouse_up()
            elif event.type == pygame.MOUSEMOTION and self._dragging:
                self._on_mouse_drag(event.pos)
        self._draw()

    def get_state(self) -> JoystickState:
        return self._state.copy()

    def close(self) -> None:
        if self._screen is not None:
            pygame.quit()
            self._screen = None

    # ── 鼠标交互 ──

    def _on_mouse_down(self, pos):
        mx, my = pos
        for c in CONTROLS:
            name, kind = c[0], c[1]
            if kind == "stick":
                _, _, cx, cy, r, _ = c
                if _in_circle(mx, my, cx, cy, r):
                    self._dragging = (name, (mx, my))
                    self._update_stick(name, c, mx, my)
                    return
            elif kind in ("btn", "dpad"):
                _, _, cx, cy, r = c[:5]
                if _in_circle(mx, my, cx, cy, r + 5):
                    self._state.buttons[name] = True
                    self._dragging = (name, None)
                    return
            elif kind == "bumper":
                _, _, bx, by, bw, bh = c
                if bx <= mx <= bx + bw and by <= my <= by + bh:
                    self._state.buttons[name] = True
                    self._dragging = (name, None)
                    return
            elif kind == "trigger":
                _, _, cx, cy, tw, th = c
                if abs(mx - cx) <= tw and abs(my - cy) <= th:
                    val = 1.0 - max(0.0, min(1.0, (my - cy) / th))
                    self._state.axes[name] = val
                    self._dragging = (name, None)
                    return

    def _on_mouse_up(self):
        if not self._dragging:
            return
        name = self._dragging[0]
        for c in CONTROLS:
            if c[0] != name:
                continue
            kind = c[1]
            if kind == "stick":
                prefix = name.replace("_stick", "")
                self._state.axes[f"{prefix}_x"] = 0.0
                self._state.axes[f"{prefix}_y"] = 0.0
            elif kind in ("btn", "dpad", "bumper"):
                self._state.buttons[name] = False
            elif kind == "trigger":
                self._state.axes[name] = 0.0
            break
        self._dragging = None

    def _on_mouse_drag(self, pos):
        if not self._dragging:
            return
        name = self._dragging[0]
        for c in CONTROLS:
            if c[0] != name:
                continue
            if c[1] == "stick":
                self._update_stick(name, c, pos[0], pos[1])
            elif c[1] == "trigger":
                _, _, cx, cy, tw, th = c
                self._state.axes[name] = 1.0 - max(0.0, min(1.0, (pos[1] - cy) / th))
            break

    def _update_stick(self, name, ctrl, mx, my):
        _, _, cx, cy, r, _ = ctrl
        dx = (mx - cx) / r         # 右推 = X 正
        dy = (cy - my) / r         # 前推(上) = Y 正
        length = (dx ** 2 + dy ** 2) ** 0.5
        if length > 1.0:
            dx /= length
            dy /= length
        prefix = name.replace("_stick", "")
        self._state.axes[f"{prefix}_x"] = dx if abs(dx) > DEADZONE else 0.0
        self._state.axes[f"{prefix}_y"] = dy if abs(dy) > DEADZONE else 0.0

    # ── 绘制 ──

    def _draw(self):
        sc = self._screen
        sc.fill(BG)
        self._draw_body(sc)
        for c in CONTROLS:
            kind = c[1]
            if kind == "stick":
                self._draw_stick(sc, c)
            elif kind in ("btn", "dpad"):
                self._draw_button(sc, c)
            elif kind == "bumper":
                self._draw_bumper(sc, c)
            elif kind == "trigger":
                self._draw_trigger(sc, c)
        pygame.display.flip()

    def _draw_body(self, sc):
        """画手柄主体轮廓 + 握把，模拟 Xbox 手柄外形。"""
        # 主体
        pygame.draw.rect(sc, BODY, (110, 50, 730, 540), border_radius=65)
        pygame.draw.rect(sc, BODY_BORDER, (110, 50, 730, 540), 3, border_radius=65)
        # 左握把
        pygame.draw.rect(sc, GRIP, (85, 330, 185, 300), border_radius=50)
        pygame.draw.rect(sc, BODY_BORDER, (85, 330, 185, 300), 2, border_radius=50)
        # 右握把
        pygame.draw.rect(sc, GRIP, (680, 330, 185, 300), border_radius=50)
        pygame.draw.rect(sc, BODY_BORDER, (680, 330, 185, 300), 2, border_radius=50)
        # 中央装饰条
        pygame.draw.line(sc, DIM, (350, 120), (600, 120), 2)

    def _draw_stick(self, sc, c):
        _, _, cx, cy, r, kr = c
        prefix = c[0].replace("_stick", "")
        vx = self._state.axes.get(f"{prefix}_x", 0.0)
        vy = self._state.axes.get(f"{prefix}_y", 0.0)

        # 外圈凹槽
        pygame.draw.circle(sc, STICK_BG, (cx, cy), r)
        pygame.draw.circle(sc, BODY_BORDER, (cx, cy), r, 2)
        # 凹槽内环
        pygame.draw.circle(sc, (25, 25, 38), (cx, cy), r - 8, 2)

        # 旋钮（Y 轴：屏幕 Y 向下，但前推=Y正，所以取反）
        kx = int(cx + vx * (r - kr - 5))
        ky = int(cy - vy * (r - kr - 5))
        active = abs(vx) > 0.02 or abs(vy) > 0.02
        # 旋钮阴影
        pygame.draw.circle(sc, (20, 20, 30), (kx + 2, ky + 2), kr)
        # 旋钮本体
        color = STICK_ACTIVE if active else STICK_KNOB
        pygame.draw.circle(sc, color, (kx, ky), kr)
        pygame.draw.circle(sc, tuple(min(c + 35, 255) for c in color), (kx, ky), kr, 2)
        # 旋钮中心点
        pygame.draw.circle(sc, tuple(min(c + 60, 255) for c in color), (kx, ky), 4)

        # 轴数值标签
        label = self._font.render(f"X:{vx:+.2f} Y:{vy:+.2f}", True, GRAY)
        sc.blit(label, (cx - label.get_width() // 2, cy + r + 8))

    def _draw_button(self, sc, c):
        if len(c) >= 5 and c[4] is not None:
            name, _, cx, cy, r, color_key = c[0], c[1], c[2], c[3], c[4], c[5] if len(c) > 5 else None
        else:
            name, _, cx, cy, r = c[0], c[1], c[2], c[3], c[4]
            color_key = None
        pressed = self._state.buttons.get(name, False)

        if color_key and color_key in ABXY_COLORS:
            base = ABXY_COLORS[color_key] if not pressed else tuple(min(v + 40, 255) for v in ABXY_COLORS[color_key])
        else:
            base = BTN_PRESSED if pressed else BTN_NORMAL

        # 阴影
        pygame.draw.circle(sc, (15, 15, 22), (cx + 2, ky := cy + 2), r)
        # 按键主体
        pygame.draw.circle(sc, base, (cx, cy), r)
        pygame.draw.circle(sc, tuple(min(v + 30, 255) for v in base), (cx, cy), r, 2)

        # 标签
        display_name = name.upper().replace("DPAD_", "")
        lbl = self._font_sm.render(display_name, True, WHITE if pressed else (180, 180, 200))
        sc.blit(lbl, (cx - lbl.get_width() // 2, cy - lbl.get_height() // 2))

    def _draw_bumper(self, sc, c):
        _, _, bx, by, bw, bh = c
        pressed = self._state.buttons.get(c[0], False)
        color = BTN_PRESSED if pressed else BTN_NORMAL
        pygame.draw.rect(sc, (15, 15, 22), (bx + 2, by + 2, bw, bh), border_radius=12)
        pygame.draw.rect(sc, color, (bx, by, bw, bh), border_radius=12)
        pygame.draw.rect(sc, tuple(min(v + 25, 255) for v in color), (bx, by, bw, bh), 2, border_radius=12)
        lbl = self._font.render(c[0].upper(), True, WHITE if pressed else GRAY)
        sc.blit(lbl, (bx + bw // 2 - lbl.get_width() // 2, by + bh // 2 - lbl.get_height() // 2))

    def _draw_trigger(self, sc, c):
        _, _, cx, cy, tw, th = c
        name = c[0]
        val = self._state.axes.get(name, 0.0)
        x, y = cx - tw // 2, cy

        # 背景槽
        pygame.draw.rect(sc, (20, 20, 32), (x, y, tw, th), border_radius=8)
        # 填充（从下往上，val=1 时满）
        fill_h = int(val * (th - 4))
        if fill_h > 0:
            tc = TRIGGER_ON.get(name, (100, 180, 140))
            pygame.draw.rect(sc, tc, (x + 2, y + th - 2 - fill_h, tw - 4, fill_h), border_radius=6)
        # 边框
        pygame.draw.rect(sc, BODY_BORDER, (x, y, tw, th), 2, border_radius=8)
        # 标签
        lbl = self._font_sm.render(f"{name.upper()} {val:.0%}", True, WHITE)
        sc.blit(lbl, (cx - lbl.get_width() // 2, y - 18))


def _in_circle(mx, my, cx, cy, r):
    return (mx - cx) ** 2 + (my - cy) ** 2 <= r * r


if __name__ == "__main__":
    """独立运行：显示虚拟手柄，实时打印 axes/buttons。"""
    js = SimulatedJoystick()
    last = None
    clock = pygame.time.Clock()
    print("虚拟手柄已启动，鼠标操作，Ctrl+C 退出")
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
