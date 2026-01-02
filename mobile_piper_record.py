import time
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import h5py

from mujoco_lidar import MjLidarWrapper
from mujoco_lidar import scan_gen

from piper_sim_env import MobileDualPiperEnvironment
from piper_sim_task import MobileDualPiperTaskPiper
from constants import DT
from piper_policy_scripted import MobileDualPiperPickAndTransferPolicyPiper

# Create MuJoCo model and data
mj_model = mujoco.MjModel.from_xml_path("/home/jeong/zeno/wholebody-teleop/act/assets/mobile_piper.xml")
mj_data = mujoco.MjData(mj_model)

num_beams = 360
rays_theta, rays_phi = scan_gen.create_lidar_single_line(num_beams)
exclude_body_id = mj_model.body("lidar_base").id
lidar = MjLidarWrapper(
    mj_model,
    site_name="lidar_site",
    backend="cpu",  # Use CPU backend
    cutoff_dist=50.0,  # Maximum detection distance of 50 meters
    args={'bodyexclude': exclude_body_id}  # CPU backend specific parameter: exclude body
)

task = MobileDualPiperTaskPiper()
env = MobileDualPiperEnvironment(mj_model, mj_data, task, lidar=lidar, rays_theta=rays_theta, rays_phi=rays_phi)
policy = MobileDualPiperPickAndTransferPolicyPiper()

# **************************************************
# sim code
# **************************************************

episode_idx = 0
while True:
    print("=" * 100)
    print(f"Episode {episode_idx}")
    print("=" * 100)
    ts = env.reset()
    policy = MobileDualPiperPickAndTransferPolicyPiper()
    episode = [ts]
    for idx in range(1500-1):
        action = policy(ts)
        ts = env.step(action)
        episode.append(ts)
    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/observations/images/top': [],
        '/observations/images/left': [],
        # '/observations/images/right': [],
        '/observations/lidar/points': [],
        '/observations/lidar/distances': [],
        '/action': [],
        '/reward': [],
        '/timestamp': [],
    }
    for time_step, ts in enumerate(episode):
        # combined_qpos = np.concatenate([ts.observation['qvel'][:3], ts.observation['qpos'][3:]])
        # data_dict['/observations/qpos'].append(combined_qpos)
        data_dict['/observations/qpos'].append(ts.observation['qpos'])
        data_dict['/observations/qvel'].append(ts.observation['qvel'])
        data_dict['/observations/images/top'].append(ts.observation['images']['top'])
        data_dict['/observations/images/left'].append(ts.observation['images']['left'])
        # data_dict['/observations/images/right'].append(ts.observation['images']['right'])
        data_dict['/observations/lidar/points'].append(ts.observation['lidar']['points'])
        data_dict['/observations/lidar/distances'].append(ts.observation['lidar']['distances'])
        data_dict['/action'].append(ts.action)
        data_dict['/reward'].append(ts.reward)
        data_dict['/timestamp'].append(time_step * DT)
    
    print("*" * 50)
    print(f"Total reward: {ts.reward}")

    if ts.reward < 4:
        print(f"Episode {episode_idx} failed. Reward: {ts.reward}")
        continue

    max_timesteps = len(episode)
    dataset_dir = f"/home/jeong/zeno/wholebody-teleop/act/dataset/sim_mobile_transfer_cube"
    os.makedirs(dataset_dir, exist_ok=True)
    dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}')
    print(f"Saving episode {episode_idx} to {dataset_path}, {max_timesteps} timesteps...")
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = True
        obs = root.create_group('observations')
        image = obs.create_group('images')
        image.create_dataset('top', (max_timesteps, 480, 640, 3), dtype='uint8', chunks=(1, 480, 640, 3))
        image.create_dataset('left', (max_timesteps, 480, 640, 3), dtype='uint8', chunks=(1, 480, 640, 3))
        # image.create_dataset('right', (max_timesteps, 480, 640, 3), dtype='uint8', chunks=(1, 480, 640, 3))
        qpos = obs.create_dataset('qpos', (max_timesteps, 17))
        qvel = obs.create_dataset('qvel', (max_timesteps, 17))
        lidar_group = obs.create_group('lidar')
        lidar_group.create_dataset('points', (max_timesteps, num_beams, 3))
        lidar_group.create_dataset('distances', (max_timesteps, num_beams))
        action = root.create_dataset('action', (max_timesteps, 17))
        reward = root.create_dataset('reward', (max_timesteps,))
        timestamp = root.create_dataset('timestamp', (max_timesteps,))

        for name, array in data_dict.items():
            root[name][...] = array

    print("*" * 50)
    episode_idx += 1
    if episode_idx >= 50:
        break

exit()

# **************************************************
# test code
# **************************************************

rays_theta, rays_phi = scan_gen.create_lidar_single_line()
exclude_body_id = mj_model.body("lidar_base").id
lidar = MjLidarWrapper(
    mj_model,
    site_name="lidar_site",
    backend="cpu",  # Use CPU backend
    cutoff_dist=50.0,  # Maximum detection distance of 50 meters
    args={'bodyexclude': exclude_body_id}  # CPU backend specific parameter: exclude body
)

ts = env.reset()
t = 0
reset = False
break_point = True

# Create 2D figure for laser scatter points
plt.figure(figsize=(10, 8))
ax_lidar = plt.subplot(1, 1, 1)
ax_lidar.set_title('Laser Scatter Points (2D)')
ax_lidar.set_xlabel('X (meters)')
ax_lidar.set_ylabel('Y (meters)')
ax_lidar.grid(True)
ax_lidar.set_aspect('equal')

# ax = plt.subplot(1, 3, 1)
# plt_img_top = ax.imshow(ts.observation['images']['top'])
# ax = plt.subplot(1, 3, 2)
# plt_img_left = ax.imshow(ts.observation['images']['left'])
# ax = plt.subplot(1, 3, 3)
# plt_img_right = ax.imshow(ts.observation['images']['right'])
plt.ion()

while env.viewer.is_running():
    if reset:
        ts = env.reset()
        reset = False
        t = 0
        break_point = True
    action = policy(ts)
    ts = env.step(action)

    lidar.trace_rays(mj_data, rays_theta, rays_phi)
    # Get point cloud data (in local coordinate system)
    points = lidar.get_hit_points()  # shape: (N, 3)
    distances = lidar.get_distances()  # shape: (N,)

    # Update 2D scatter plot
    ax_lidar.clear()
    ax_lidar.set_title('Laser Scatter Points (2D)')
    ax_lidar.set_xlabel('X (meters)')
    ax_lidar.set_ylabel('Y (meters)')
    ax_lidar.grid(True)
    ax_lidar.set_aspect('equal')

    # Plot valid points (where distance < cutoff_dist)
    valid_mask = distances < 50.0  # Only plot points within range
    if np.any(valid_mask):
        valid_points = points[valid_mask]
        ax_lidar.scatter(valid_points[:, 0], valid_points[:, 1], c=distances[valid_mask],
                        cmap='viridis', s=1, alpha=0.7)
        ax_lidar.set_xlim(-4, 4)
        ax_lidar.set_ylim(-4, 4)
    plt.draw()
    plt.pause(0.001)

    # plt_img_top.set_data(ts.observation['images']['top'])
    # plt_img_left.set_data(ts.observation['images']['left'])
    # plt_img_right.set_data(ts.observation['images']['right'])
    # plt.pause(0.0001)
    time.sleep(DT)
    t += DT

    # if t > 600 * DT and break_point:
    #     break_point = False
    #     input("Press Enter to continue break point...")

    if t < 1500 * DT:
        print(f"Time: {t:.3f}s, Reward: {ts.reward}, Basevel: {ts.observation['qvel'][:3]}, Basepos: {ts.observation['qpos'][:3]}")
    else:
        input("Press Enter to continue...")
        reset = True
