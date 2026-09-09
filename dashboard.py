"""dashboard.py —— 上位机:模式切换 + 速度遥控 + 相机/YOLO 识别 + 状态监视。

模式:阻尼模式 / 力控模式 / PD站立 / 位控模式 / RL行走
  - 阻尼模式   关节只有阻尼,机器人瘫软(转运/急停姿态)
  - 力控模式   12 个关节力矩滑条,直接输出力矩
  - PD站立     锁默认站姿(kp/kd 来自配置)
  - 位控模式   12 个关节角度滑条,PD 跟踪摆姿势
  - RL行走     训练好的策略(TorchScript)50Hz 出目标角,vx/vy/wz 遥控

相机面板显示配置里 YOLO 相机的实时画面;yolo.enabled 时叠加检测框和距离。

用法:
  python dashboard.py                       # 默认 configs/go2.yaml
  python dashboard.py --config configs/go2_terrain.yaml
  python dashboard.py --robot g1            # 命令行覆盖机器人
  python dashboard.py --no-viewer           # 不开 3D 窗口
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tkinter as tk
from tkinter import ttk

from simulation import SimWorld, load_config
from simulation.controller import MODES, MODE_LABELS

RAD2DEG, DEG2RAD = 180.0 / np.pi, np.pi / 180.0

# ---------------- 主题 ----------------
BG = "#14151a"        # 窗口底
PANEL = "#1f2128"     # 面板
PANEL2 = "#262933"    # 面板内控件
LINE = "#33374a"      # 描边
TEXT = "#e8eaf0"      # 主文字
MUTED = "#8b93a7"     # 次要文字
ACCENT = "#3d8bfd"    # 高亮蓝
GREEN = "#2ecc71"
RED = "#e74c3c"
ORANGE = "#f39c12"
FONT = ("Microsoft YaHei UI", 10)
FONT_B = ("Microsoft YaHei UI", 10, "bold")
FONT_S = ("Microsoft YaHei UI", 9)
FONT_M = ("Consolas", 9)
FONT_T = ("Microsoft YaHei UI", 13, "bold")

CAM_W, CAM_H = 620, 465       # 相机画面显示尺寸


class Dashboard:
    def __init__(self, cfg, use_viewer=True):
        self.cfg = cfg
        self.world = SimWorld(cfg)
        self.lock = threading.Lock()
        self.world.reset_robot()
        self.use_viewer = use_viewer
        if use_viewer:
            self.world.open_viewer()

        c = self.controller = self.world.controller
        if c is None:
            raise RuntimeError("配置缺少 controller 段,上位机无法运行")
        self.joint_names = [n.replace("_joint", "") for n in c.joint_names]

        import mujoco
        self.jnt_range_deg = []
        for name in c.joint_names:
            jid = mujoco.mj_name2id(self.world.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            lo, hi = self.world.model.jnt_range[jid]
            self.jnt_range_deg.append((lo * RAD2DEG - 2, hi * RAD2DEG + 2))
        self.default_deg = [float(v) * RAD2DEG for v in c.q_default]

        # YOLO(可选)
        self.detector = None
        self.yolo_cam = None
        self._last_yolo_t = -1e9
        self._cam_frame = None            # 物理线程写,UI 线程读
        self._dets = []
        if cfg.yolo.enabled and cfg.yolo.camera in self.world.cameras:
            self.yolo_cam = self.world.cameras[cfg.yolo.camera]
            from yolo import YoloWorldDetector
            try:
                from pathlib import Path as _P
                w = _P(cfg.yolo.weights)
                if not w.is_absolute():
                    w = Path(__file__).resolve().parent / w
                self.detector = YoloWorldDetector(str(w), cfg.yolo_classes(), conf=cfg.yolo.conf)
                print(f"[yolo] 已启用: {w.name}, 相机 {cfg.yolo.camera}")
            except Exception as e:
                print(f"[yolo] 初始化失败({e}),相机只显示原始画面")

        self._pending_reset = False
        self._running = True
        self._photo = None                # 防 PhotoImage 被回收
        self._phys = threading.Thread(target=self._physics_loop, daemon=True)
        self._phys.start()
        self._build_ui()

    # ================= 物理线程 =================
    def _physics_loop(self):
        step_start = time.perf_counter()
        dt = self.world.model.opt.timestep
        while self._running and (not self.use_viewer or self.world.viewer_running()):
            with self.lock:
                if self._pending_reset:
                    self.world.reset_robot()
                    self._pending_reset = False
                self.world.step()
                self._maybe_detect()
            step_start += dt
            delay = step_start - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                step_start = time.perf_counter()

    def _maybe_detect(self):
        """物理线程里按 yolo.interval 周期抓相机 + 检测。"""
        if self.yolo_cam is None:
            return
        interval = self.cfg.yolo.interval if self.detector else 0.2
        if self.world.sim_time - self._last_yolo_t < interval:
            return
        self._last_yolo_t = self.world.sim_time
        rgb = self.yolo_cam.render_rgb()
        dets = []
        if self.detector is not None:
            from yolo import measure_detections
            dets = measure_detections(self.detector.detect(rgb), self.yolo_cam)
            self._dets = [(d.name, d.conf, d.distance) for d in dets]
        else:
            self._dets = []
        self._cam_frame = rgb if self.detector is None else self.detector.annotate(rgb, dets)

    # ================= 线程安全操作 =================
    def request_reset(self):
        with self.lock:
            self._pending_reset = True

    def set_mode(self, mode: str):
        with self.lock:
            self.controller.mode = mode
            self.mode_var.set(MODE_LABELS[mode])
            self._highlight_modes()
            self._show_panel()

    def emergency_stop(self):
        """急停:速度清零 + 切回 PD站立。"""
        with self.lock:
            c = self.controller
            c.update_commands([0.0, 0.0, 0.0])
            for v in self.cmd_vars:
                v.set(0.0)
            c.mode = "pd_stand"
        self.mode_var.set(MODE_LABELS["pd_stand"])
        self._highlight_modes()
        self._show_panel()

    # ================= UI =================
    def _build_ui(self):
        self.root = tk.Tk()
        self.root.title(f"MuJoCo 机器人上位机 — {self.cfg.robot.upper()}")
        self.root.configure(bg=BG)
        w, h = 1280, 760
        self.root.geometry(f"{w}x{h}")
        self.root.minsize(1180, 700)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=PANEL, foreground=TEXT, font=FONT)
        style.configure("Muted.TLabel", background=PANEL, foreground=MUTED, font=FONT_S)
        style.configure("Title.TLabel", background=BG, foreground=TEXT, font=FONT_T)
        style.configure("Badge.TLabel", background=PANEL, foreground=ACCENT, font=FONT_B)
        style.configure("Horizontal.TScale", background=PANEL, troughcolor=PANEL2,
                        bordercolor=PANEL, lightcolor=ACCENT, darkcolor=ACCENT)

        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        # ---- 顶栏 ----
        head = ttk.Frame(outer)
        head.pack(fill="x", pady=(0, 8))
        ttk.Label(head, text="🤖 MuJoCo 机器人上位机", style="Title.TLabel").pack(side="left")
        self.time_badge = ttk.Label(head, text="t = 0.0 s", style="Badge.TLabel")
        self.time_badge.pack(side="right")

        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)

        # ---- 左列:模式 + 相机 ----
        left = ttk.Frame(body, style="Panel.TFrame", padding=10)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 8))

        ttk.Label(left, text="控制模式", style="Muted.TLabel").pack(anchor="w", pady=(0, 4))
        self.mode_var = tk.StringVar(value=MODE_LABELS.get(self.controller.mode, ""))
        self.mode_btns = {}
        icon = {"damping": "🫧", "force": "🖐", "pd_stand": "🧍", "position": "🎮", "policy": "🚶"}
        for mode in MODES:
            b = tk.Button(left, text=f"  {icon[mode]}  {MODE_LABELS[mode]}  ",
                          font=FONT, bd=0, cursor="hand2",
                          bg=PANEL2, fg=TEXT, activebackground=ACCENT, activeforeground="white",
                          command=lambda m=mode: self.set_mode(m))
            b.pack(fill="x", pady=3, ipady=6)
            self.mode_btns[mode] = b
        self._highlight_modes()

        row = tk.Frame(left, bg=PANEL)
        row.pack(fill="x", pady=(12, 0))
        tk.Button(row, text="⚠ 急停", font=FONT_B, bd=0, cursor="hand2",
                  bg=RED, fg="white", activebackground="#c0392b",
                  command=self.emergency_stop).pack(side="left", expand=True, fill="x", ipady=6, padx=(0, 4))
        tk.Button(row, text="⟲ 复位", font=FONT_B, bd=0, cursor="hand2",
                  bg=PANEL2, fg=TEXT, activebackground=ACCENT,
                  command=self.request_reset).pack(side="left", expand=True, fill="x", ipady=6)

        # 状态区
        st = tk.Frame(left, bg=PANEL)
        st.pack(fill="both", expand=True, pady=(12, 0))
        tk.Label(st, text="状态", bg=PANEL, fg=MUTED, font=FONT_S).pack(anchor="w")
        self.status_var = tk.StringVar(value="…")
        tk.Label(st, textvariable=self.status_var, bg=PANEL, fg=TEXT,
                 font=FONT_M, justify="left", anchor="nw").pack(anchor="nw")

        # ---- 中列:相机 + YOLO ----
        camf = tk.Frame(body, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
        camf.grid(row=0, column=1, sticky="nsew", padx=(0, 8))
        title = tk.Label(camf, text="相机视角 / YOLO 识别", bg=PANEL, fg=MUTED, font=FONT_S,
                         anchor="w", padx=8, pady=4)
        title.pack(fill="x")
        self.cam_label = tk.Label(camf, bg="#0c0d10",
                                  text=self._camera_placeholder(), fg=MUTED, font=FONT)
        self.cam_label.pack(padx=6, pady=(0, 4))
        self.det_var = tk.StringVar(value="")
        tk.Label(camf, textvariable=self.det_var, bg=PANEL, fg=GREEN,
                 font=FONT_S, justify="left", anchor="nw").pack(fill="x", padx=8, pady=(0, 6))

        # ---- 右列:速度遥控 / 关节滑条 ----
        self.right = tk.Frame(body, bg=PANEL, highlightbackground=LINE, highlightthickness=1)
        self.right.grid(row=0, column=2, sticky="nsew")
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self.cmd_vars = [tk.DoubleVar(value=0.0) for _ in range(3)]
        self._cmd_val_labels = []
        self.joint_slider_vars = []
        self._show_panel()

        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.after(150, self._refresh)

    def _camera_placeholder(self):
        if self.yolo_cam is None:
            return "配置里没有可用的相机\n(cameras 段 / yolo.camera)"
        return "\n\n等待第一帧…\n"

    # ---- 右列各面板 ----
    def _clear_right(self):
        for w in self.right.winfo_children():
            w.destroy()
        self._cmd_val_labels = []
        self.joint_slider_vars = []

    def _panel_title(self, text):
        tk.Label(self.right, text=text, bg=PANEL, fg=MUTED, font=FONT_S,
                 anchor="w").pack(fill="x", padx=10, pady=(10, 2))

    def _build_velocity_panel(self):
        self._clear_right()
        self._panel_title("RL行走 · 速度遥控")
        labels = ["前进 vx (m/s)", "侧移 vy (m/s)", "转向 wz (rad/s)"]
        for i, (lab, var) in enumerate(zip(labels, self.cmd_vars)):
            row = tk.Frame(self.right, bg=PANEL)
            row.pack(fill="x", padx=10, pady=(4, 0))
            tk.Label(row, text=lab, bg=PANEL, fg=TEXT, font=FONT_S, width=13,
                     anchor="w").pack(side="left")
            val = tk.Label(row, text=f"{var.get():+.2f}", bg=PANEL, fg=ACCENT,
                           font=FONT_M, width=6)
            val.pack(side="right")
            self._cmd_val_labels.append(val)
            ttk.Scale(row, from_=-1.0, to=1.0, variable=var,
                      command=lambda _, i=i: self._on_cmd()).pack(
                side="left", fill="x", expand=True, padx=6)

        presets = tk.Frame(self.right, bg=PANEL)
        presets.pack(fill="x", padx=10, pady=(10, 0))
        tk.Label(presets, text="快捷速度:", bg=PANEL, fg=MUTED, font=FONT_S).pack(side="left")
        for v, name in [(0.0, "停"), (0.3, "0.3"), (0.5, "0.5"), (0.8, "0.8"), (1.0, "1.0")]:
            tk.Button(presets, text=name, font=FONT_S, bd=0, cursor="hand2", padx=8,
                      bg=PANEL2, fg=TEXT, activebackground=ACCENT,
                      command=lambda vv=v: self._set_vx(vv)).pack(side="left", padx=3, pady=2)

        tk.Label(self.right, text="拖动滑条或点快捷速度;wz 正值左转。\n检测到机器人偏航时用 wz 修正。",
                 bg=PANEL, fg=MUTED, font=FONT_S, justify="left").pack(anchor="w", padx=10, pady=(10, 0))

    def _set_vx(self, v):
        self.cmd_vars[0].set(v)
        self._on_cmd()

    def _show_panel(self):
        mode = self.controller.mode
        if mode == "policy":
            self._build_velocity_panel()
        elif mode in ("position", "force"):
            self._build_joint_sliders(mode)
        else:                                   # damping / pd_stand 无滑条
            self._clear_right()
            self._panel_title(MODE_LABELS[mode])
            tk.Label(self.right, text="该模式无调节参数。\n\n力控/位控模式提供 12 关节滑条;\nRL行走 提供速度遥控。",
                     bg=PANEL, fg=MUTED, font=FONT_S, justify="left").pack(anchor="w", padx=10, pady=6)

    def _build_joint_sliders(self, mode):
        self._clear_right()
        if mode == "position":
            self._panel_title("位控模式 · 关节目标角(度)")
            init, lo_hi = self.default_deg, self.jnt_range_deg
            fmt = "{:+.0f}"
        else:
            self._panel_title("力控模式 · 关节力矩 (N·m)")
            init, lo_hi = [0.0] * 12, [(-25, 25)] * 12
            fmt = "{:+.1f}"
        for i in range(12):
            row = tk.Frame(self.right, bg=PANEL)
            row.pack(fill="x", padx=10, pady=(2, 0))
            v = tk.DoubleVar(value=float(init[i]))
            self.joint_slider_vars.append(v)
            lab = tk.Label(row, text=f"{self.joint_names[i]:>9s} {fmt.format(init[i])}",
                           bg=PANEL, fg=TEXT, font=FONT_M, width=18, anchor="w")
            lab.pack(side="left")
            ttk.Scale(row, from_=lo_hi[i][0], to=lo_hi[i][1], variable=v,
                      command=lambda _, i=i, l=lab, f=fmt: self._joint_moved(i, l, f)
                      ).pack(side="left", fill="x", expand=True, padx=6)

    def _joint_moved(self, i, label, fmt):
        v = self.joint_slider_vars[i].get()
        label.config(text=f"{self.joint_names[i]:>9s} {fmt.format(v)}")
        with self.lock:
            if self.controller.mode == "position":
                self.controller.q_des_ui[i] = v * DEG2RAD
            elif self.controller.mode == "force":
                self.controller.tau_ui[i] = v

    # ---- 回调 ----
    def _on_cmd(self):
        self.controller.update_commands([v.get() for v in self.cmd_vars])

    def _highlight_modes(self):
        mode = self.controller.mode
        for m, b in self.mode_btns.items():
            if m == mode:
                b.config(bg=ACCENT, fg="white")
            else:
                b.config(bg=PANEL2, fg=TEXT)

    # ---- 状态刷新(UI 线程) ----
    def _refresh(self):
        if not self._running:
            return
        c = self.controller
        self.time_badge.config(text=f"t = {self.world.sim_time:6.1f} s")
        pos, quat, _, _, _ = c.base_state()
        wq, x, y, z = quat
        roll = np.degrees(np.arctan2(2 * (wq * x + y * z), 1 - 2 * (x * x + y * y)))
        pitch = np.degrees(np.arcsin(np.clip(2 * (wq * y - z * x), -1, 1)))
        yaw = np.degrees(np.arctan2(2 * (wq * z + x * y), 1 - 2 * (y * y + z * z)))
        q, _ = c.joint_state()

        lines = [f"base高度  {pos[2]:6.3f} m",
                 f"俯仰/横滚 {pitch:6.1f} {roll:6.1f} °",
                 f"偏航      {yaw:6.1f} °",
                 f"指令      vx={c.cmd_target[0]:+.2f} vy={c.cmd_target[1]:+.2f} wz={c.cmd_target[2]:+.2f}"]
        self.status_var.set("\n".join(lines))
        for i, val in enumerate(self._cmd_val_labels):
            val.config(text=f"{self.cmd_vars[i].get():+.2f}")

        # 相机帧
        frame = self._cam_frame
        if frame is not None:
            try:
                from PIL import Image, ImageTk
                img = Image.fromarray(frame).resize((CAM_W, CAM_H))
                self._photo = ImageTk.PhotoImage(img)
                self.cam_label.config(image=self._photo, text="", width=CAM_W, height=CAM_H)
            except Exception:
                pass
        if self._dets:
            self.det_var.set("识别: " + "   ".join(
                f"{n} {conf:.2f}" + (f" {dist:.2f}m" if dist else "") for n, conf, dist in self._dets[:4]))
        else:
            self.det_var.set("")

        self.root.after(120, self._refresh)

    def _close(self):
        self._running = False
        self.world.close()
        self.root.destroy()

    def run(self):
        self.root.mainloop()
        self._running = False
        self.world.close()


def main():
    ap = argparse.ArgumentParser(description="MuJoCo 机器人上位机")
    ap.add_argument("-c", "--config", default=None)
    ap.add_argument("--robot", default=None)
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, robot=args.robot)
    cfg.yolo.enabled = True                      # 上位机默认开相机识别
    app = Dashboard(cfg, use_viewer=not args.no_viewer)
    app.run()


if __name__ == "__main__":
    main()
