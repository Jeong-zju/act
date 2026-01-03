import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

import IPython
e = IPython.embed

def pad_qpos(qpos, target_dim=17):
    """
    Pad qpos array/tensor with zeros at the front if dimension is less than target_dim.
    If qpos is 14D, pad with 3 zeros at the front to make it 17D.
    """
    if isinstance(qpos, torch.Tensor):
        qpos_np = qpos.cpu().numpy()
        is_tensor = True
        device = qpos.device if hasattr(qpos, 'device') else None
    else:
        qpos_np = np.asarray(qpos)
        is_tensor = False
        device = None
    
    original_shape = qpos_np.shape
    current_dim = original_shape[-1]
    
    if current_dim < target_dim:
        pad_size = target_dim - current_dim
        # Pad along the last dimension
        pad_shape = list(original_shape)
        pad_shape[-1] = pad_size
        padding = np.zeros(pad_shape, dtype=qpos_np.dtype)
        padded = np.concatenate([padding, qpos_np], axis=-1)
    elif current_dim == target_dim:
        padded = qpos_np
    else:
        raise ValueError(f"qpos dimension {current_dim} is greater than target_dim {target_dim}")
    
    if is_tensor:
        result = torch.from_numpy(padded).float()
        if device is not None:
            result = result.to(device)
        return result
    else:
        return padded

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, use_qtor=False, use_lidar=False):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.use_qtor = use_qtor
        self.use_lidar = use_lidar
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            # Load qtor if available and use_qtor is True, otherwise use zeros
            if self.use_qtor and '/observations/qtor' in root:
                qtor = root['/observations/qtor'][start_ts]
            else:
                qtor = np.zeros_like(qpos)
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # Load lidar_scan if available and use_lidar is True, otherwise use zeros
            if self.use_lidar and '/observations/lidar/distances' in root:
                lidar_scan = root['/observations/lidar/distances'][start_ts]
            else:
                # Default to 1080 beams if we need to create zeros (common LiDAR size)
                lidar_scan = np.zeros(1080, dtype=np.float32)
            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        qtor_data = torch.from_numpy(qtor).float()
        lidar_scan_data = torch.from_numpy(lidar_scan).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # pad qpos if needed (14D -> 17D)
        qpos_data = pad_qpos(qpos_data, target_dim=17)
        # pad qtor if needed (14D -> 17D)
        qtor_data = pad_qpos(qtor_data, target_dim=17)
        # pad action if needed (14D -> 17D)
        action_data = pad_qpos(action_data, target_dim=17)

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        qtor_data = (qtor_data - self.norm_stats["qtor_mean"]) / self.norm_stats["qtor_std"]
        # Note: lidar_scan_data is not normalized here - LiDAREncoder handles preprocessing

        return image_data, qpos_data, qtor_data, lidar_scan_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_qtor_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            if '/observations/qtor' in root:
                qtor = root['/observations/qtor'][()]
            else:
                qtor = np.zeros_like(qpos)
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_qtor_data.append(torch.from_numpy(qtor))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_qtor_data = torch.stack(all_qtor_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # pad qpos data if needed (14D -> 17D)
    # Check the last dimension
    if all_qpos_data.shape[-1] < 17:
        # Pad along the last dimension
        pad_size = 17 - all_qpos_data.shape[-1]
        padding = torch.zeros(*all_qpos_data.shape[:-1], pad_size, dtype=all_qpos_data.dtype)
        all_qpos_data = torch.cat([padding, all_qpos_data], dim=-1)

    # pad qtor data if needed (14D -> 17D)
    # Check the last dimension
    if all_qtor_data.shape[-1] < 17:
        # Pad along the last dimension
        pad_size = 17 - all_qtor_data.shape[-1]
        padding = torch.zeros(*all_qtor_data.shape[:-1], pad_size, dtype=all_qtor_data.dtype)
        all_qtor_data = torch.cat([padding, all_qtor_data], dim=-1)

    # pad action data if needed (14D -> 17D)
    # Check the last dimension
    if all_action_data.shape[-1] < 17:
        # Pad along the last dimension
        pad_size = 17 - all_action_data.shape[-1]
        padding = torch.zeros(*all_action_data.shape[:-1], pad_size, dtype=all_action_data.dtype)
        all_action_data = torch.cat([padding, all_action_data], dim=-1)

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    # normalize qtor data
    qtor_mean = all_qtor_data.mean(dim=[0, 1], keepdim=True)
    qtor_std = all_qtor_data.std(dim=[0, 1], keepdim=True)
    qtor_std = torch.clip(qtor_std, 1e-2, np.inf) # clipping

    # Pad example_qpos if needed
    example_qpos = qpos
    if example_qpos.shape[-1] < 17:
        example_qpos = pad_qpos(example_qpos, target_dim=17)
    example_qtor = qtor
    if example_qtor.shape[-1] < 17:
        example_qtor = pad_qpos(example_qtor, target_dim=17)

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "qtor_mean": qtor_mean.numpy().squeeze(), "qtor_std": qtor_std.numpy().squeeze(),
             "example_qpos": example_qpos}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, use_qtor=False, use_lidar=False):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, use_qtor=use_qtor, use_lidar=use_lidar)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, use_qtor=use_qtor, use_lidar=use_lidar)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
