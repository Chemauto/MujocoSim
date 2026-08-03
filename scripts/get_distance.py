"""Open the MuJoCo RGB view; click any point to read its distance from the camera.

Mirrors examples/detect_box.py: RGB + depth shown side by side, the distance at
the clicked pixel is printed and drawn on screen. Depth math lives in utils.depth.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import mujoco

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.depth import get_intrinsics, deproject_pixel, pixel_distance, depth_to_colormap

XML = "/data/camera/Mujoco/scene.xml"
CAMERA = "cam"
WIDTH, HEIGHT = 640, 480
STEPS = 50


def render_frame():
    """Render one RGB + depth pair. Returns (rgb, depth, K, depth_colormap)."""
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA)
    for _ in range(STEPS):
        mujoco.mj_step(model, data)

    r = mujoco.Renderer(model, HEIGHT, WIDTH)
    r.update_scene(data, camera=CAMERA)
    rgb = r.render()
    r.enable_depth_rendering()
    depth = r.render()

    K = get_intrinsics(model, cam_id, WIDTH, HEIGHT)
    return rgb, depth, K, depth_to_colormap(depth)


def measure_at(depth, u, v, K):
    """Return (X, Y, Z, R) in meters at pixel (u, v), or None if depth invalid."""
    p = deproject_pixel(depth, u, v, K)
    if p is None:
        return None
    X, Y, Z = p
    R = pixel_distance(depth, u, v, K)
    return X, Y, Z, R


def main():
    rgb, depth, K, depth_vis = render_frame()
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    state = {"last": None}

    def on_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        m = measure_at(depth, x, y, K)
        if m is None:
            print(f"({x},{y}) -> no valid depth")
            state["last"] = (x, y, None)
            return
        X, Y, Z, R = m
        print(f"({x},{y}) -> X={X:.3f}m Y={Y:.3f}m Z={Z:.3f}m R={R:.3f}m")
        state["last"] = (x, y, m)

    title = "RGB - click to measure (q/ESC to quit)"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, on_mouse)

    print("Click on the RGB image to measure distance. Press 'q' or ESC to quit.")
    while True:
        panel = rgb_bgr.copy()
        if state["last"]:
            x, y, m = state["last"]
            cv2.drawMarker(panel, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            if m is not None:
                X, Y, Z, R = m
                cv2.putText(panel, f"X={X:.2f} Y={Y:.2f} Z={Z:.2f} m   R={R:.2f} m",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(title, np.hstack((panel, depth_vis)))
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
