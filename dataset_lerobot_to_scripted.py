#!/usr/bin/env python3
"""
Convert LeRobot dataset to scripted HDF5 format.

This script loads data from a LeRobot-formatted dataset (e.g., push_block_dual_lerobot21)
and converts it to the HDF5 format used by scripted datasets (e.g., sim_transfer_cube_scripted).

Usage:
    python dataset_lerobot_to_scripted.py --input_dataset_dir dataset/push_block_dual_lerobot21 --output_dataset_dir dataset/push_block_dual_scripted --camera_names main:top,secondary_0:left,secondary_1:right
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np


def parse_camera_mapping(camera_mapping_str: str) -> Dict[str, str]:
    """Parse camera name mapping from string format 'lerobot_name:scripted_name,lerobot_name:scripted_name'"""
    mapping = {}
    if camera_mapping_str:
        for pair in camera_mapping_str.split(','):
            lerobot_name, scripted_name = pair.split(':')
            mapping[lerobot_name.strip()] = scripted_name.strip()
    return mapping


def load_episode_data(lerobot_dataset_dir: str, episode_index: int, camera_names: List[str]):
    """Load all data for a single episode from LeRobot dataset"""
    dataset_root = Path(lerobot_dataset_dir)

    # Load episode metadata
    episodes_jsonl = dataset_root / "meta" / "episodes.jsonl"
    episode_length = None
    with episodes_jsonl.open("r") as f:
        for line in f:
            item = json.loads(line.strip())
            if item["episode_index"] == episode_index:
                episode_length = item["length"]
                break

    if episode_length is None:
        raise ValueError(f"Episode {episode_index} not found in episodes.jsonl")

    # Load info.json for data paths
    info_path = dataset_root / "meta" / "info.json"
    with info_path.open("r") as f:
        info = json.load(f)

    chunks_size = info.get("chunks_size", 1000)
    data_path_template = info["data_path"]
    video_path_template = info["video_path"]

    # Load parquet data
    episode_chunk = episode_index // chunks_size
    data_rel_path = data_path_template.format(episode_chunk=episode_chunk, episode_index=episode_index)
    parquet_path = dataset_root / data_rel_path

    # Try to load with pyarrow first, then pandas
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(parquet_path)
        action_col = table.column("action").to_pylist()
        state_col = table.column("observation.state").to_pylist()
        frame_col = table.column("frame_index").to_pylist() if "frame_index" in table.column_names else None
    except ImportError:
        try:
            import pandas as pd
            df = pd.read_parquet(parquet_path)
            action_col = df["action"].to_list()
            state_col = df["observation.state"].to_list()
            frame_col = df["frame_index"].to_list() if "frame_index" in df.columns else None
        except ImportError:
            raise RuntimeError("Need either pyarrow or pandas to read parquet files")

    # Convert to numpy arrays
    action = np.array(action_col, dtype=np.float32)
    state = np.array(state_col, dtype=np.float32)

    if action.ndim == 1:
        action = np.stack([np.asarray(v) for v in action], axis=0)
    if state.ndim == 1:
        state = np.stack([np.asarray(v) for v in state], axis=0)

    if frame_col is None:
        frame_index = np.arange(len(action), dtype=np.int64)
    else:
        frame_index = np.array(frame_col, dtype=np.int64)
        if frame_index.ndim == 2 and frame_index.shape[1] == 1:
            frame_index = frame_index[:, 0]

    # Load video frames for each camera
    images = {}
    for cam_name in camera_names:
        video_key = f"observation.images.{cam_name}" if not cam_name.startswith("observation.images.") else cam_name
        video_rel_path = video_path_template.format(
            episode_chunk=episode_chunk,
            episode_index=episode_index,
            video_key=video_key
        )
        video_path = dataset_root / video_rel_path

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Load all frames from video
        frames = []
        try:
            import av
            container = av.open(str(video_path))
            try:
                for frame in container.decode(video=0):
                    frames.append(frame.to_ndarray(format="rgb24"))
            finally:
                container.close()
        except ImportError:
            try:
                import imageio.v3 as iio
                # This might be slow for long videos, but works
                for frame_idx in range(len(action)):
                    frame = iio.imread(str(video_path), index=frame_idx)
                    if frame.ndim == 3 and frame.shape[-1] >= 3:
                        frames.append(frame[:, :, :3].astype(np.uint8))
                    else:
                        raise RuntimeError(f"Unexpected frame shape: {frame.shape}")
            except ImportError:
                raise RuntimeError("Need either av or imageio to read video frames")

        images[cam_name] = np.stack(frames, axis=0)

    return {
        'action': action,
        'qpos': state,  # observation.state is joint positions
        'images': images,
        'frame_index': frame_index
    }


def save_episode_hdf5(data: Dict, output_path: str, camera_names: List[str], is_sim: bool = False):
    """Save episode data to HDF5 format matching scripted dataset structure"""
    max_timesteps = len(data['action'])

    with h5py.File(output_path, 'w', rdcc_nbytes=1024**2*2) as root:
        root.attrs['sim'] = is_sim

        # Create groups
        obs = root.create_group('observations')
        image_group = obs.create_group('images')

        # Create datasets
        for cam_name in camera_names:
            image_data = data['images'][cam_name]
            if image_data.shape[0] != max_timesteps:
                raise ValueError(f"Image sequence length {image_data.shape[0]} doesn't match action length {max_timesteps}")
            _ = image_group.create_dataset(
                cam_name, (max_timesteps, image_data.shape[1], image_data.shape[2], image_data.shape[3]),
                dtype='uint8', chunks=(1, image_data.shape[1], image_data.shape[2], image_data.shape[3]),
                compression='gzip', compression_opts=6
            )
            image_group[cam_name][...] = image_data

        qpos = obs.create_dataset('qpos', (max_timesteps, data['qpos'].shape[1]), dtype=np.float32, compression='gzip', compression_opts=6)
        qvel = obs.create_dataset('qvel', (max_timesteps, data['qpos'].shape[1]), dtype=np.float32, compression='gzip', compression_opts=6)
        action = root.create_dataset('action', (max_timesteps, data['action'].shape[1]), dtype=np.float32, compression='gzip', compression_opts=6)

        # Fill data
        qpos[...] = data['qpos']
        # For qvel, compute as differences or set to zeros (LeRobot doesn't have qvel directly)
        if max_timesteps > 1:
            qvel[1:, :] = np.diff(data['qpos'], axis=0)
        qvel[0, :] = 0  # First timestep velocity is 0
        action[...] = data['action']


def main():
    parser = argparse.ArgumentParser(description="Convert LeRobot dataset to scripted HDF5 format")
    parser.add_argument('--input_dataset_dir', type=str, required=True,
                        help='Path to LeRobot dataset directory (e.g., dataset/push_block_dual_lerobot21)')
    parser.add_argument('--output_dataset_dir', type=str, required=True,
                        help='Path to output directory for scripted dataset')
    parser.add_argument('--camera_names', type=str, required=True,
                        help='Camera name mapping: "lerobot_name:scripted_name,lerobot_name:scripted_name" '
                             '(e.g., "main:top,secondary_0:left,secondary_1:right")')
    parser.add_argument('--max_episodes', type=int, default=None,
                        help='Maximum number of episodes to convert (default: all)')
    parser.add_argument('--start_episode', type=int, default=0,
                        help='Starting episode index (default: 0)')

    args = parser.parse_args()

    # Parse camera mapping
    camera_mapping = parse_camera_mapping(args.camera_names)
    lerobot_camera_names = list(camera_mapping.keys())
    scripted_camera_names = list(camera_mapping.values())

    print(f"Input dataset: {args.input_dataset_dir}")
    print(f"Output dataset: {args.output_dataset_dir}")
    print(f"Camera mapping: {camera_mapping}")

    # Load dataset info
    info_path = Path(args.input_dataset_dir) / "meta" / "info.json"
    with info_path.open("r") as f:
        info = json.load(f)

    total_episodes = info.get("total_episodes", 0)
    if args.max_episodes is not None:
        num_episodes = min(args.max_episodes, total_episodes - args.start_episode)
    else:
        num_episodes = total_episodes - args.start_episode

    print(f"Converting {num_episodes} episodes (starting from episode {args.start_episode})")

    # Create output directory
    os.makedirs(args.output_dataset_dir, exist_ok=True)

    # Convert episodes
    for i in range(num_episodes):
        episode_idx = args.start_episode + i
        output_episode_idx = i  # Always start output numbering from 0
        print(f"Converting episode {episode_idx}...")

        try:
            # Load episode data
            data = load_episode_data(args.input_dataset_dir, episode_idx, lerobot_camera_names)

            # Remap camera names
            remapped_images = {}
            for lerobot_name, scripted_name in camera_mapping.items():
                remapped_images[scripted_name] = data['images'][lerobot_name]
            data['images'] = remapped_images

            # Save as HDF5
            output_path = os.path.join(args.output_dataset_dir, f'episode_{output_episode_idx}.hdf5')
            save_episode_hdf5(data, output_path, scripted_camera_names, is_sim=True)

            print(f"  Saved episode {episode_idx} as episode_{output_episode_idx}.hdf5 with {len(data['action'])} timesteps")

        except Exception as e:
            print(f"  Error converting episode {episode_idx}: {e}")
            continue

    print(f"Conversion complete. Output saved to {args.output_dataset_dir}")


if __name__ == '__main__':
    main()
