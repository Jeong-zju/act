#!/usr/bin/env python3
"""
Simple script to visualize ACT dataset episodes

This script provides easy-to-use functions for visualizing episode data.
"""

import os
import sys
import argparse
from interactive_visualizer import InteractiveVisualizer, create_video

def visualize_episode(dataset_dir, episode_idx):
    """Launch interactive visualization for a specific episode"""
    print(f"Visualizing episode {episode_idx} from {dataset_dir}")
    viz = InteractiveVisualizer(dataset_dir, episode_idx)
    viz.show()

def create_episode_video(dataset_dir, episode_idx, output_path=None):
    """Create video from episode data"""
    create_video(dataset_dir, episode_idx, output_path)

def list_available_episodes(dataset_dir):
    """List available episodes in dataset directory"""
    if not os.path.exists(dataset_dir):
        print(f"Directory {dataset_dir} does not exist")
        return

    episodes = [f for f in os.listdir(dataset_dir) if f.endswith('.hdf5')]
    episodes.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    print(f"Available episodes in {dataset_dir}:")
    for ep in episodes:
        idx = ep.split('_')[1].split('.')[0]
        print(f"  Episode {idx}: {ep}")

def main():
    parser = argparse.ArgumentParser(description='ACT Dataset Visualizer')
    parser.add_argument('--dataset_dir', type=str,
                       help='Path to dataset directory')
    parser.add_argument('--episode_idx', type=int,
                       help='Episode index to visualize')
    parser.add_argument('--list', action='store_true',
                       help='List available episodes in dataset directory')
    parser.add_argument('--video', action='store_true',
                       help='Create video instead of interactive visualization')
    parser.add_argument('--output', type=str,
                       help='Output path for video')

    # Default dataset directory
    default_dataset = 'dataset/sim_transfer_cube_pyb'

    args = parser.parse_args()

    if args.list:
        dataset_dir = args.dataset_dir or default_dataset
        list_available_episodes(dataset_dir)
        return

    if not args.dataset_dir or args.episode_idx is None:
        print("Usage examples:")
        print("  # List available episodes")
        print(f"  python {sys.argv[0]} --dataset_dir {default_dataset} --list")
        print("")
        print("  # Interactive visualization")
        print(f"  python {sys.argv[0]} --dataset_dir {default_dataset} --episode_idx 0")
        print("")
        print("  # Create video")
        print(f"  python {sys.argv[0]} --dataset_dir {default_dataset} --episode_idx 0 --video --output episode_0.mp4")
        return

    if args.video:
        create_episode_video(args.dataset_dir, args.episode_idx, args.output)
    else:
        visualize_episode(args.dataset_dir, args.episode_idx)

if __name__ == '__main__':
    main()
