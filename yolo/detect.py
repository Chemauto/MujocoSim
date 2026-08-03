import numpy as np
import mujoco
import matplotlib.pyplot as plt
from ultralytics import YOLOWorld

XML = "/data/camera/Mujoco/scene.xml"
WEIGHTS = "/home/xcj/work/YoloProject/yolov8m-world.pt"
OUT = "/data/camera/Mujoco/yolo/detect.png"

CAMERA = "cam"
CLASSES = ["cube", "box", "block", "red cube"]

# ---- camera intrinsics (adjustable) ----
W, H = 640, 480          # resolution
FOVY = 45.0              # vertical FOV [deg] -> sets focal length

# ---- 1) MuJoCo: render RGB with the chosen intrinsics ----
model = mujoco.MjModel.from_xml_path(XML)
data = mujoco.MjData(model)

cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA)
model.cam_fovy[cam_id] = FOVY                 # adjust focal length via fovy
focal_px = H / (2 * np.tan(np.deg2rad(FOVY) / 2))
print(f"camera intrinsics: {W}x{H}, fovy={FOVY}deg, focal~{focal_px:.1f}px")

for _ in range(50):
    mujoco.mj_step(model, data)

renderer = mujoco.Renderer(model, H, W)
renderer.update_scene(data, camera=CAMERA)
rgb = renderer.render()                        # HxWx3 uint8, RGB

# ---- 2) YOLO-World: detect on the rendered image ----
yolo = YOLOWorld(WEIGHTS)
yolo.set_classes(CLASSES)
bgr = np.ascontiguousarray(rgb[..., ::-1])     # RGB -> BGR (ultralytics convention)
result = yolo.predict(bgr, conf=0.15)[0]

print(f"detected {len(result.boxes)} object(s):")
for b in result.boxes:
    cls = yolo.names[int(b.cls)]
    print(f"  {cls}: conf={float(b.conf):.3f}, xyxy={b.xyxy[0].tolist()}")

# ---- 3) save side-by-side ----
annotated = result.plot()[..., ::-1]           # BGR -> RGB for display
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].imshow(rgb);       ax[0].set_title("MuJoCo RGB");  ax[0].axis("off")
ax[1].imshow(annotated); ax[1].set_title("YOLO-World");  ax[1].axis("off")
plt.tight_layout(); plt.savefig(OUT)
print("saved:", OUT)
