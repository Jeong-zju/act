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
task = MobileDualPiperTaskPiper()
env = MobileDualPiperEnvironment(mj_model, mj_data, task)
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
        '/action': [],
        '/reward': [],
        '/timestamp': [],
    }
    for time_step, ts in enumerate(episode):
        data_dict['/observations/qpos'].append(ts.observation['qpos'])
        data_dict['/observations/qvel'].append(ts.observation['qvel'])
        data_dict['/observations/images/top'].append(ts.observation['images']['top'])
        data_dict['/observations/images/left'].append(ts.observation['images']['left'])
        # data_dict['/observations/images/right'].append(ts.observation['images']['right'])
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
    action = policy(ts)
    ts = env.step(action)
    plt_img_top.set_data(ts.observation['images']['top'])
    plt_img_left.set_data(ts.observation['images']['left'])
    plt_img_right.set_data(ts.observation['images']['right'])
    plt.pause(0.0001)
    t += DT
    if t < 1500 * DT:
        print(f"Time: {t:.3f}s, Reward: {ts.reward}")
