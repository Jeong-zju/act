import time
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from mujoco_lidar import MjLidarWrapper
from mujoco_lidar import scan_gen

from piper_sim_env import MobileDualPiperEnvironment
from piper_sim_task import MobileDualPiperTaskPiper
from constants import DT

# Create MuJoCo model and data
mj_model = mujoco.MjModel.from_xml_path("/home/jeong/zeno/wholebody-teleop/act/assets/mobile_piper.xml")
mj_data = mujoco.MjData(mj_model)
task = MobileDualPiperTaskPiper()
env = MobileDualPiperEnvironment(mj_model, mj_data, task)
ts = env.reset()
t = 0

ax = plt.subplot(1, 3, 1)
plt_img_top = ax.imshow(ts.observation['images']['top'])
ax = plt.subplot(1, 3, 2)
plt_img_left = ax.imshow(ts.observation['images']['left'])
ax = plt.subplot(1, 3, 3)
plt_img_right = ax.imshow(ts.observation['images']['right'])
plt.ion()

while env.viewer.is_running():
    action = np.array([0.0] * 17)
    action[0] = 1
    action[2] = 1
    action[9] = (np.sin(np.pi * t) + 1) / 2
    action[16] = (np.cos(np.pi * t) + 1) / 2
    ts = env.step(action)
    # ts = env.step(np.array([0, 0, 0]))
    plt_img_top.set_data(ts.observation['images']['top'])
    plt_img_left.set_data(ts.observation['images']['left'])
    plt_img_right.set_data(ts.observation['images']['right'])
    plt.pause(0.0001)
    print("=" * 50)
    print(f"Time: {t:.3f}s")
    print(ts.observation["qpos"][:3])
    print(ts.observation["qvel"][:3])
    time.sleep(DT)
    t += DT

# # Generate scan pattern
# rays_theta, rays_phi = scan_gen.generate_grid_scan_pattern(num_ray_cols=64, num_ray_rows=16)

# # Get body ID to exclude (avoid LiDAR detecting itself)
# exclude_body_id = mj_model.body("lidar_base").id

# # Create CPU backend LiDAR sensor
# lidar = MjLidarWrapper(
#     mj_model,
#     site_name="lidar_site",
#     backend="cpu",  # Use CPU backend
#     cutoff_dist=50.0,  # Maximum detection distance of 50 meters
#     args={'bodyexclude': exclude_body_id}  # CPU backend specific parameter: exclude body
# )

# idx = 0

# # Set up matplotlib for interactive plotting
# plt.ion()
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlabel('X (m)')
# ax.set_ylabel('Y (m)')
# ax.set_zlabel('Z (m)')
# ax.set_title('LiDAR Point Cloud')

# # Use in simulation loop
# with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
#     while viewer.is_running():
#         mujoco.mj_step(mj_model, mj_data)
#         viewer.sync()

#         # Perform ray tracing (Wrapper automatically handles pose updates)
#         lidar.trace_rays(mj_data, rays_theta, rays_phi)

#         # Get point cloud data (in local coordinate system)
#         points = lidar.get_hit_points()  # shape: (N, 3)
#         distances = lidar.get_distances()  # shape: (N,)

#         print(f"Frame {idx}: {len(points)} points detected")

#         # Plot points
#         if len(points) > 0:
#             ax.clear()
#             ax.set_xlabel('X (m)')
#             ax.set_ylabel('Y (m)')
#             ax.set_zlabel('Z (m)')
#             ax.set_title(f'LiDAR Point Cloud - Frame {idx}')

#             # Color points by distance
#             scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
#                                c=distances, cmap='viridis', s=10, alpha=0.6)
#             # plt.colorbar(scatter, ax=ax, label='Distance (m)')

#             # Set equal aspect ratio
#             max_range = np.array([points[:, 0].max() - points[:, 0].min(),
#                                  points[:, 1].max() - points[:, 1].min(),
#                                  points[:, 2].max() - points[:, 2].min()]).max() / 2.0
#             mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
#             mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
#             mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
#             ax.set_xlim(mid_x - max_range, mid_x + max_range)
#             ax.set_ylim(mid_y - max_range, mid_y + max_range)
#             ax.set_zlim(mid_z - max_range, mid_z + max_range)

#             plt.draw()
#             plt.pause(0.001)

#         idx += 1
#         time.sleep(1./60.)
