#!/usr/bin/env python3
"""
Interactive Visualizer for ACT Dataset Episodes

This script provides an interactive visualization of episode data containing:
- Joint configurations (qpos, qvel, actions)
- Multiple camera views
- Timestamp information

Usage:
    python interactive_visualizer.py --dataset_dir dataset/sim_transfer_cube_pyb --episode_idx 0
"""

import os
import numpy as np
import cv2
import h5py
import argparse
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.gridspec as gridspec
from constants import DT, JOINT_NAMES
import warnings

warnings.filterwarnings("ignore")

# Joint names for dual-arm setup
LEFT_JOINT_NAMES = [f"left_{name}" for name in JOINT_NAMES] + ["left_gripper"]
RIGHT_JOINT_NAMES = [f"right_{name}" for name in JOINT_NAMES] + ["right_gripper"]
ALL_JOINT_NAMES = LEFT_JOINT_NAMES + RIGHT_JOINT_NAMES
ALL_JOINT_NAMES = ["left_1", "left_2", "left_3", "left_4", "left_5", "left_6", "left_gripper", "right_1", "right_2", "right_3", "right_4", "right_5", "right_6", "right_gripper"]


class InteractiveVisualizer:
    def __init__(self, dataset_dir, episode_idx):
        self.dataset_dir = dataset_dir
        self.episode_idx = episode_idx
        self.dataset_name = f"episode_{episode_idx}"

        # Load data
        self.qpos, self.qvel, self.qtor, self.action, self.image_dict, self.timestamps = self.load_hdf5_data()
        # self.qpos, self.qvel, self.action, self.image_dict = self.load_hdf5_data()

        # Setup visualization
        self.current_frame = 0
        self.num_frames = len(self.qpos)

        # Create figure and layout
        self.setup_figure()

        # Initialize plots
        self.update_plots()

    def load_hdf5_data(self):
        """Load data from HDF5 file"""
        dataset_path = os.path.join(self.dataset_dir, self.dataset_name + ".hdf5")
        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(f"Dataset does not exist at {dataset_path}")

        with h5py.File(dataset_path, "r") as root:
            qpos = root["/observations/qpos"][()]
            qvel = root["/observations/qvel"][()]
            if '/observations/qtor' in root:
                qtor = root["/observations/qtor"][()]
            else:
                qtor = np.zeros_like(qpos)
            action = root["/action"][()]
            timestamps = root["/timestamp"][()]

            image_dict = dict()
            for cam_name in root[f"/observations/images/"].keys():
                image_dict[cam_name] = root[f"/observations/images/{cam_name}"][()]

        return qpos, qvel, qtor, action, image_dict, timestamps
        # return qpos, qvel, action, image_dict

    def setup_figure(self):
        """Setup the matplotlib figure with subplots"""
        # Create figure with custom grid
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle(f"Episode {self.episode_idx} - Interactive Visualization", fontsize=16)

        # Create gridspec for layout
        # Check if torque data is available (not all zeros)
        self.has_torque_data = np.any(np.abs(self.qtor) > 1e-6)
        
        if self.has_torque_data:
            # Layout with separate torque plot: cameras, positions, torques, slider
            gs = gridspec.GridSpec(4, 4, figure=self.fig, height_ratios=[2, 1, 1, 0.1], hspace=0.3, wspace=0.3)
        else:
            # Layout without torque plot: cameras, positions, slider
            gs = gridspec.GridSpec(3, 4, figure=self.fig, height_ratios=[2, 1, 0.1], hspace=0.3, wspace=0.3)

        # Camera views (top row, 3 cameras)
        self.ax_cameras = []
        camera_names = list(self.image_dict.keys())
        for i, cam_name in enumerate(camera_names):
            ax = self.fig.add_subplot(gs[0, i])
            ax.set_title(f'{cam_name.replace("_", " ").title()} Camera')
            ax.axis("off")
            self.ax_cameras.append(ax)

        # Joint position/action plots (second row, spans all columns)
        self.ax_joints = self.fig.add_subplot(gs[1, :])
        self.ax_joints.set_title("Joint Positions and Actions")
        self.ax_joints.set_xlabel("Timestep")
        self.ax_joints.set_ylabel("Joint Value")
        self.ax_joints.grid(True, alpha=0.3)

        # Torque plots (third row, spans all columns) - only if torque data exists
        if self.has_torque_data:
            self.ax_torque = self.fig.add_subplot(gs[2, :])
            self.ax_torque.set_title("Joint Torques")
            self.ax_torque.set_xlabel("Timestep")
            self.ax_torque.set_ylabel("Torque Value")
            self.ax_torque.grid(True, alpha=0.3)
            slider_row = 3
        else:
            self.ax_torque = None
            slider_row = 2

        # Slider (bottom row)
        self.ax_slider = self.fig.add_subplot(gs[slider_row, :])
        self.slider = Slider(self.ax_slider, "Timestep", 0, self.num_frames - 1, valinit=0, valstep=1, valfmt="%d")
        self.slider.on_changed(self.on_slider_change)

        # Initialize camera images
        self.camera_images = []
        for ax in self.ax_cameras:
            img = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
            self.camera_images.append(img)

        # Initialize joint lines
        self.qpos_lines = []
        self.action_lines = []
        self.qtor_lines = []
        colors = plt.cm.tab20(np.linspace(0, 1, len(ALL_JOINT_NAMES)))

        for i, joint_name in enumerate(ALL_JOINT_NAMES):
            color = colors[i % len(colors)]
            # Position and action lines on ax_joints
            (qpos_line,) = self.ax_joints.plot(
                [], [], color=color, linewidth=2, label=f"{joint_name} (qpos)", alpha=0.8
            )
            (action_line,) = self.ax_joints.plot(
                [], [], color=color, linewidth=1, linestyle="--", label=f"{joint_name} (action)", alpha=0.6
            )
            self.qpos_lines.append(qpos_line)
            self.action_lines.append(action_line)
            
            # Torque lines on separate ax_torque (if available)
            if self.has_torque_data:
                (qtor_line,) = self.ax_torque.plot(
                    [], [], color=color, linewidth=2, label=f"{joint_name} (qtor)", alpha=0.8
                )
                self.qtor_lines.append(qtor_line)
            else:
                self.qtor_lines.append(None)

        # Add legend for position/action plot (only show subset to avoid clutter)
        legend_elements_pos = []
        legend_labels_pos = []
        num_legend_joints = min(4, len(ALL_JOINT_NAMES))  # Show first 4 joints
        
        for i in range(num_legend_joints):
            legend_elements_pos.append(self.qpos_lines[i])
            legend_labels_pos.append(f"{ALL_JOINT_NAMES[i]} (qpos)")
            legend_elements_pos.append(self.action_lines[i])
            legend_labels_pos.append(f"{ALL_JOINT_NAMES[i]} (action)")
        
        self.ax_joints.legend(
            legend_elements_pos,
            legend_labels_pos,
            loc="upper right",
            fontsize=8,
            ncol=2,
        )
        
        # Add legend for torque plot (if available)
        if self.has_torque_data:
            legend_elements_torque = []
            legend_labels_torque = []
            for i in range(num_legend_joints):
                if self.qtor_lines[i] is not None:
                    legend_elements_torque.append(self.qtor_lines[i])
                    legend_labels_torque.append(f"{ALL_JOINT_NAMES[i]} (qtor)")
            
            self.ax_torque.legend(
                legend_elements_torque,
                legend_labels_torque,
                loc="upper right",
                fontsize=8,
                ncol=2,
            )

    def update_plots(self):
        """Update all plots for current frame"""
        # Update camera images
        for i, (cam_name, ax, img) in enumerate(zip(self.image_dict.keys(), self.ax_cameras, self.camera_images)):
            frame_image = self.image_dict[cam_name][self.current_frame]
            # Images are already in RGB format from HDF5
            frame_image_rgb = frame_image

            # Add timestamp overlay
            timestamp = self.timestamps[self.current_frame]
            cv2.putText(frame_image_rgb, ".3f", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            img.set_array(frame_image_rgb)
            ax.set_title(".3f")

        # Update joint plots
        time_range = np.arange(self.num_frames)

        # Plot vertical line at current frame for position/action plot
        if hasattr(self, "current_line"):
            self.current_line.remove()
        self.current_line = self.ax_joints.axvline(
            x=self.current_frame, color="red", linestyle="--", alpha=0.8, linewidth=2
        )

        # Plot vertical line at current frame for torque plot (if available)
        if self.has_torque_data:
            if hasattr(self, "current_line_torque"):
                self.current_line_torque.remove()
            self.current_line_torque = self.ax_torque.axvline(
                x=self.current_frame, color="red", linestyle="--", alpha=0.8, linewidth=2
            )

        # Update position and action lines on ax_joints
        for i in range(len(ALL_JOINT_NAMES)):
            self.qpos_lines[i].set_data(time_range, self.qpos[:, i])
            self.action_lines[i].set_data(time_range, self.action[:, i])

        # Update torque lines on ax_torque (if available)
        if self.has_torque_data:
            for i in range(len(ALL_JOINT_NAMES)):
                if self.qtor_lines[i] is not None:
                    self.qtor_lines[i].set_data(time_range, self.qtor[:, i])

        # Update axis limits for position/action plot
        self.ax_joints.set_xlim(0, self.num_frames - 1)
        y_min_pos = min(np.min(self.qpos), np.min(self.action))
        y_max_pos = max(np.max(self.qpos), np.max(self.action))
        margin_pos = (y_max_pos - y_min_pos) * 0.1
        self.ax_joints.set_ylim(y_min_pos - margin_pos, y_max_pos + margin_pos)

        # Update axis limits for torque plot (if available)
        if self.has_torque_data:
            self.ax_torque.set_xlim(0, self.num_frames - 1)
            torque_data = np.concatenate([self.qtor[:, i] for i in range(len(ALL_JOINT_NAMES)) if self.qtor_lines[i] is not None])
            if len(torque_data) > 0:
                y_min_torque = np.min(torque_data)
                y_max_torque = np.max(torque_data)
                margin_torque = (y_max_torque - y_min_torque) * 0.1
                self.ax_torque.set_ylim(y_min_torque - margin_torque, y_max_torque + margin_torque)

        self.fig.canvas.draw_idle()

    def on_slider_change(self, val):
        """Callback for slider value change"""
        self.current_frame = int(val)
        self.update_plots()

    def show(self):
        """Display the interactive visualization"""
        plt.tight_layout()
        plt.show()

    def save_snapshot(self, filename=None):
        """Save current frame as image"""
        if filename is None:
            filename = f"episode_{self.episode_idx}_frame_{self.current_frame:04d}.png"
        self.fig.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"Saved snapshot to {filename}")


def create_video(dataset_dir, episode_idx, output_path=None, fps=None):
    """Create video from episode data"""
    if fps is None:
        fps = int(1.0 / DT)

    # Load data
    dataset_name = f"episode_{episode_idx}"
    dataset_path = os.path.join(dataset_dir, dataset_name + ".hdf5")

    with h5py.File(dataset_path, "r") as root:
        image_dict = dict()
        for cam_name in root[f"/observations/images/"].keys():
            image_dict[cam_name] = root[f"/observations/images/{cam_name}"][()]

    if output_path is None:
        output_path = os.path.join(dataset_dir, f"{dataset_name}_video.mp4")

    # Create video
    cam_names = list(image_dict.keys())
    num_frames = image_dict[cam_names[0]].shape[0]

    # Calculate canvas size
    h, w = image_dict[cam_names[0]].shape[1:3]
    canvas_w = w * len(cam_names)
    canvas_h = h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (canvas_w, canvas_h))

    for frame_idx in range(num_frames):
        # Concatenate all camera views horizontally
        images = []
        for cam_name in cam_names:
            frame = image_dict[cam_name][frame_idx]
            # Convert RGB to BGR for OpenCV video writer
            frame_bgr = frame[:, :, [2, 1, 0]]
            images.append(frame_bgr)
        combined_frame = np.concatenate(images, axis=1)
        out.write(combined_frame)

    out.release()
    print(f"Saved video to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Interactive ACT Dataset Visualizer")
    parser.add_argument(
        "--dataset_dir", type=str, required=True, help="Path to dataset directory containing HDF5 files"
    )
    parser.add_argument("--episode_idx", type=int, required=True, help="Episode index to visualize")
    parser.add_argument(
        "--mode", type=str, choices=["interactive", "video"], default="interactive", help="Visualization mode"
    )
    parser.add_argument("--output_video", type=str, help="Output path for video (when mode=video)")
    parser.add_argument("--fps", type=int, help="Video FPS (default: 1/DT)")

    args = parser.parse_args()

    try:
        if args.mode == "interactive":
            print(f"Loading episode {args.episode_idx} from {args.dataset_dir}")
            viz = InteractiveVisualizer(args.dataset_dir, args.episode_idx)
            print("Interactive visualization ready. Use the slider to navigate through the episode.")
            print("Close the window to exit.")
            viz.show()
        elif args.mode == "video":
            print(f"Creating video for episode {args.episode_idx}")
            create_video(args.dataset_dir, args.episode_idx, args.output_video, args.fps)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Available episodes in {args.dataset_dir}:")
        if os.path.exists(args.dataset_dir):
            episodes = [f for f in os.listdir(args.dataset_dir) if f.endswith(".hdf5")]
            episodes.sort()
            for ep in episodes[:10]:  # Show first 10
                print(f"  {ep}")
            if len(episodes) > 10:
                print(f"  ... and {len(episodes)-10} more")


if __name__ == "__main__":
    main()
