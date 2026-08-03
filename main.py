import time
import mujoco
import mujoco.viewer

XML = "/data/camera/Mujoco/scene.xml"

# Open the MuJoCo simulator window and run physics in real time.
model = mujoco.MjModel.from_xml_path(XML)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)   # keep roughly real-time
