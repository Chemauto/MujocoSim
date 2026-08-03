import mujoco
import numpy as np
import matplotlib.pyplot as plt

XML = "/data/camera/Mujoco/scene.xml"
CAMERA = "cam"


def render(model, data, camera=CAMERA, H=240, W=320):
    """Return (rgb uint8[H,W,3], depth float[H,W] in meters) from a camera."""
    r = mujoco.Renderer(model, H, W)
    r.update_scene(data, camera=camera)
    rgb = r.render()
    r.enable_depth_rendering()
    depth = r.render()
    return rgb, depth


def show(rgb, depth, path="/data/camera/Mujoco/shot.png"):
    """Display RGB + depth side by side; also save to PNG (works headless)."""
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].imshow(rgb)
    ax[0].set_title("RGB"); ax[0].axis("off")
    ax[1].imshow(depth, cmap="turbo")
    ax[1].set_title("Depth (m)"); ax[1].axis("off")
    plt.tight_layout(); plt.savefig(path); plt.show()
    print("saved:", path)


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    for _ in range(50):
        mujoco.mj_step(model, data)
    show(*render(model, data))
