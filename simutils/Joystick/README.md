# 🎮 Joystick — 统一手柄接口（虚拟 Xbox / 真实 Xbox）

一个接口，两种手柄——虚拟 Xbox 窗口鼠标操控、真实 Xbox 手柄即插即用，`collect_data.py` sim 模式直接调用，关节连续转动替代逐关节键盘。

Jump to: [Install](#-install) · [Quick start](#-quick-start) · [Interface](#-interface) · [Config](#%EF%B8%8F-config) · [Key mapping](#%EF%B8%8F-key--axis-mapping)

---

## ✨ What you get

- 🕹️ **SimulatedJoystick** — pygame 窗口画出 Xbox 手柄，鼠标拖拽摇杆/点击按键，L2/R2 拖拽 0→1 连续。
- 🎮 **XboxJoystick** — 真实 Xbox 手柄 `pygame.joystick` 即插即用。
- 🔌 **统一接口** — 两种手柄 `get_state()` 返回相同 `JoystickState`，调用方无感切换。
- ⚡ **collect 集成** — `config.yaml` 一行切换 `input.device: joystick`，摇杆速度积分到关节，按键映射录制/保存/重置。

---

## 📦 Install

```bash
pip install pygame        # 仅此一个依赖（lerobot conda 环境）
```

> ⚠️ `pygame` 仅在 `input.device: joystick` 时 lazy import，键盘模式零开销。

---

## 🚀 Quick start

```bash
# 虚拟手柄（独立运行，鼠标操作，Ctrl+C 退出）
python simutils/Joystick/virtual_gamepad.py

# 真实 Xbox（插上手柄后）
python simutils/Joystick/virtual_gamepad.py --xbox

# collect_data.py 用手柄控制 sim 机械臂
# config.yaml → input.device: joystick
python scripts/collect_data.py
```

---

## 🧩 Interface

```python
from simutils.Joystick import make_joystick, state_to_joint_vel, JointMapping

js = make_joystick("sim")       # "sim" | "xbox"
js.update()                     # 每帧调用，读取事件
state = js.get_state()          # → JoystickState

vel, actions = state_to_joint_vel(state, JointMapping(max_vel=30.0))
# vel: np.ndarray(6) — 关节速度（度/秒）
# actions: {"toggle_recording": bool, "save": bool, "discard": bool, "reset": bool}

js.close()
```

### `JoystickState`

| 字段 | 类型 | 说明 |
|:---|:---|:---|
| `axes` | `dict[str, float]` | `left_x/y`, `right_x/y`, `l2`, `r2` ∈ [-1,1]（摇杆前推=Y正，右推=X正） |
| `buttons` | `dict[str, bool]` | `a/b/x/y/lb/rb/back/start/guide/dpad_*/lstick/rstick` |

### `JointMapping`

| 字段 | 默认值 | 说明 |
|:---|:---|:---|
| `axis_map` | `{left_y: (1,+1), left_x: (0,+1), right_y: (2,+1), right_x: (4,+1)}` | 摇杆轴 → (关节索引, 方向) |
| `dpad_joint` | `3` | 十字键上下控制的关节 |
| `max_vel` | `30.0` | 摇杆满杆对应关节速度（度/秒） |

---

## ⚙️ Config

```yaml
# config.yaml
input:
  device: keyboard          # keyboard | joystick
  joystick:
    kind: sim               # sim（虚拟手柄窗口）| xbox（真实 Xbox）
    max_vel: 30.0           # 摇杆满杆 → 关节速度（度/秒）
```

---

## 🕹️ Key & Axis Mapping

### 摇杆 → SO101 关节

| 手柄输入 | 关节 | 默认方向 |
|:---:|:---:|:---:|
| 左摇杆 Y | shoulder_lift (1) | 前推=正 |
| 左摇杆 X | shoulder_pan (0) | 右推=正 |
| 右摇杆 Y | elbow_flex (2) | 前推=正 |
| 右摇杆 X | wrist_roll (4) | 右推=正 |
| Dpad ↑/↓ | wrist_flex (3) | ↑=正 |
| R2 / L2 | gripper (5) | 开 / 合 |

### 按键 → 动作

| 按键 | 动作 |
|:---:|:---:|
| Start | 开始/暂停录制 |
| A | 保存 episode |
| B | 丢弃 episode |
| Back | 重置（回 home） |

> cv2 键盘 hotkeys（Space/Enter/ESC 等）无论 `keyboard`/`joystick` 模式都保留。

---

## 📚 Dependencies

| 包 | 用途 | 必须? |
|:---|:---|:---:|
| `pygame` | 虚拟手柄渲染 + 真实手柄读取 | ✅（`pip install pygame`） |

---

## 📖 See also

- [../../manipulation/VLA/README.md](../../manipulation/VLA/README.md) — 完整 VLA 管线（collect/train/eval）
- [../../manipulation/simulators/mujoco/README.md](../../manipulation/simulators/mujoco/README.md) — 仿真环境 + 标定
