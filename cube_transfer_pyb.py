"""
Cube Transfer Simulation with Performance Optimizations

This module implements a PyBullet-based simulation for cube transfer tasks with optimized
camera rendering for faster data generation.

Performance Optimizations:
- Camera capture frequency: Reduced from every timestep to every N timesteps (configurable)
- Camera resolution: Reduced from 640x480 to 320x240 by default
- Depth rendering: Disabled by default (uses faster ER_TINY_RENDERER)
- Matrix caching: Camera transformation matrices cached when gripper pose unchanged
- Configurable parameters: Camera settings can be adjusted via constructor parameters

Expected speedup: ~10-50x depending on configuration
"""

from dataclasses import dataclass
import json
import os
import sys
import time
from typing import Optional
from xml.etree import ElementTree
import cv2
from tqdm import tqdm

import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from tracikpy import TracIKSolver

import pp_lite as pp
import pybullet as p

robot_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
gripper_joint_names = ["joint7", "joint8"]


class BasePolicy:
    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.left_trajectory = None
        self.right_trajectory = None
        self.precomputed_trajectory = None

    def generate_trajectory(self):
        raise NotImplementedError

    def pose_to_matrix(self, xyz, quat):
        """Convert position and quaternion to 4x4 transformation matrix"""
        rotation = R.from_quat(quat)
        matrix = np.eye(4)
        matrix[:3, :3] = rotation.as_matrix()
        matrix[:3, 3] = xyz
        return matrix

    def precompute_joint_trajectory(self, left_robot, right_robot, left_ik_solver, right_ik_solver, left_qinit=None, right_qinit=None):
        """Precompute the entire joint trajectory using joint-space interpolation"""
        if self.left_trajectory is None or self.right_trajectory is None:
            self.generate_trajectory()

        # Find the maximum timestep in the trajectory
        max_timestep = max(
            max(wp['t'] for wp in self.left_trajectory),
            max(wp['t'] for wp in self.right_trajectory)
        )

        # Initialize precomputed trajectory
        self.precomputed_trajectory = []

        # Get initial joint positions for IK solvers
        if left_qinit is None:
            left_qinit = left_robot.get_joint_positions()
        if right_qinit is None:
            right_qinit = right_robot.get_joint_positions()

        # First, convert all waypoints to joint space
        left_joint_waypoints = []
        right_joint_waypoints = []

        for wp in self.left_trajectory:
            # Convert Cartesian waypoint to joint position
            left_pose_matrix = self.pose_to_matrix(wp['xyz'], wp['quat'])

            try:
                joint_pos = left_ik_solver.ik(left_pose_matrix, qinit=left_qinit)
                if joint_pos is None:
                    print(f"Left IK failed for waypoint at t={wp['t']}, using fallback")
                    joint_pos = [0.0] * 6
                left_qinit = joint_pos  # Update for next waypoint
            except Exception as e:
                print(f"Left IK failed for waypoint at t={wp['t']}: {e}")
                joint_pos = [0.0] * 6

            left_joint_waypoints.append({
                't': wp['t'],
                'joints': joint_pos,
                'gripper': wp['gripper']
            })

        for wp in self.right_trajectory:
            # Convert Cartesian waypoint to joint position
            right_pose_matrix = self.pose_to_matrix(wp['xyz'], wp['quat'])

            try:
                joint_pos = right_ik_solver.ik(right_pose_matrix, qinit=right_qinit)
                if joint_pos is None:
                    print(f"Right IK failed for waypoint at t={wp['t']}, using fallback")
                    joint_pos = [0.0] * 6
                right_qinit = joint_pos  # Update for next waypoint
            except Exception as e:
                print(f"Right IK failed for waypoint at t={wp['t']}: {e}")
                joint_pos = [0.0] * 6

            right_joint_waypoints.append({
                't': wp['t'],
                'joints': joint_pos,
                'gripper': wp['gripper']
            })

        # Now interpolate in joint space for each timestep
        for t in range(max_timestep + 1):
            # Get interpolated joint positions and gripper values
            left_joints, left_gripper = self._get_interpolated_joints(t, left_joint_waypoints)
            right_joints, right_gripper = self._get_interpolated_joints(t, right_joint_waypoints)

            # Add noise if requested
            if self.inject_noise:
                scale = 0.01
                left_joints = np.array(left_joints) + np.random.uniform(-scale, scale, len(left_joints))
                right_joints = np.array(right_joints) + np.random.uniform(-scale, scale, len(right_joints))
                left_joints = left_joints.tolist()
                right_joints = right_joints.tolist()

            # Store the complete state for this timestep
            self.precomputed_trajectory.append({
                'left_joints': left_joints,
                'right_joints': right_joints,
                'left_gripper': left_gripper,
                'right_gripper': right_gripper,
                'left_xyz': None,  # Not computed in joint-space interpolation
                'right_xyz': None
            })

        print(f"Precomputed joint-space trajectory with {len(self.precomputed_trajectory)} timesteps")

    def _get_interpolated_pose(self, t, arm):
        """Get interpolated pose for a specific timestep and arm"""
        trajectory = self.left_trajectory if arm == 'left' else self.right_trajectory

        # Find the appropriate waypoint interval
        curr_waypoint = None
        next_waypoint = None

        for i, wp in enumerate(trajectory):
            if wp['t'] <= t:
                curr_waypoint = wp
                if i + 1 < len(trajectory):
                    next_waypoint = trajectory[i + 1]
                else:
                    next_waypoint = wp  # Stay at final position
            else:
                next_waypoint = wp
                break

        if curr_waypoint is None:
            curr_waypoint = trajectory[0]

        if next_waypoint is None:
            next_waypoint = trajectory[-1]

        # Interpolate
        if curr_waypoint['t'] == next_waypoint['t']:
            # No interpolation needed
            return curr_waypoint['xyz'], curr_waypoint['quat'], curr_waypoint['gripper']
        else:
            t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
            curr_xyz = curr_waypoint['xyz']
            curr_quat = curr_waypoint['quat']
            curr_grip = curr_waypoint['gripper']
            next_xyz = next_waypoint['xyz']
            next_quat = next_waypoint['quat']
            next_grip = next_waypoint['gripper']
            xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
            quat = curr_quat + (next_quat - curr_quat) * t_frac
            gripper = curr_grip + (next_grip - curr_grip) * t_frac
            return xyz, quat, gripper

    def _get_interpolated_joints(self, t, joint_trajectory):
        """Get interpolated joint positions for a specific timestep using joint-space interpolation"""
        # Find the appropriate waypoint interval
        curr_waypoint = None
        next_waypoint = None

        for i, wp in enumerate(joint_trajectory):
            if wp['t'] <= t:
                curr_waypoint = wp
                if i + 1 < len(joint_trajectory):
                    next_waypoint = joint_trajectory[i + 1]
                else:
                    next_waypoint = wp  # Stay at final position
            else:
                next_waypoint = wp
                break

        if curr_waypoint is None:
            curr_waypoint = joint_trajectory[0]

        if next_waypoint is None:
            next_waypoint = joint_trajectory[-1]

        # Interpolate in joint space
        if curr_waypoint['t'] == next_waypoint['t']:
            # No interpolation needed
            return curr_waypoint['joints'], curr_waypoint['gripper']
        else:
            t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
            curr_joints = np.array(curr_waypoint['joints'])
            curr_grip = curr_waypoint['gripper']
            next_joints = np.array(next_waypoint['joints'])
            next_grip = next_waypoint['gripper']

            # Linear interpolation in joint space
            joints = curr_joints + (next_joints - curr_joints) * t_frac
            gripper = curr_grip + (next_grip - curr_grip) * t_frac
            return joints.tolist(), gripper

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_xyz = curr_waypoint['xyz']
        curr_quat = curr_waypoint['quat']
        curr_grip = curr_waypoint['gripper']
        next_xyz = next_waypoint['xyz']
        next_quat = next_waypoint['quat']
        next_grip = next_waypoint['gripper']
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        quat = curr_quat + (next_quat - curr_quat) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def __call__(self):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory()

        # obtain left and right waypoints
        if self.left_trajectory[0]['t'] == self.step_count:
            self.curr_left_waypoint = self.left_trajectory.pop(0)
        next_left_waypoint = self.left_trajectory[0]

        if self.right_trajectory[0]['t'] == self.step_count:
            self.curr_right_waypoint = self.right_trajectory.pop(0)
        next_right_waypoint = self.right_trajectory[0]

        # interpolate between waypoints to obtain current pose and gripper command
        left_xyz, left_quat, left_gripper = self.interpolate(self.curr_left_waypoint, next_left_waypoint, self.step_count)
        right_xyz, right_quat, right_gripper = self.interpolate(self.curr_right_waypoint, next_right_waypoint, self.step_count)

        # Inject noise
        if self.inject_noise:
            scale = 0.01
            left_xyz = left_xyz + np.random.uniform(-scale, scale, left_xyz.shape)
            right_xyz = right_xyz + np.random.uniform(-scale, scale, right_xyz.shape)

        self.step_count += 1
        return left_xyz, left_quat, left_gripper, right_xyz, right_quat, right_gripper

    def get_precomputed_step(self, step):
        """Get precomputed joint positions for a specific step"""
        if self.precomputed_trajectory is None:
            raise ValueError("Trajectory not precomputed. Call precompute_joint_trajectory() first.")

        if step >= len(self.precomputed_trajectory):
            # Return final position if step exceeds trajectory length
            return self.precomputed_trajectory[-1]

        return self.precomputed_trajectory[step]


class CubeTransferPolicy(BasePolicy):

    def __init__(self, cube_pose_left_local, cube_pose_right_local, meeting_point=None, inject_noise=False):
        super().__init__(inject_noise)
        self.cube_pose_left_local = cube_pose_left_local  # Cube pose in left arm's local frame
        self.cube_pose_right_local = cube_pose_right_local  # Cube pose in right arm's local frame
        self.meeting_point = meeting_point  # Should be a tuple of (left_meeting_pose_local, right_meeting_pose_local)

    def generate_trajectory(self):
        # Extract positions and orientations from local cube poses
        box_left_xyz = np.array(self.cube_pose_left_local[0])
        box_left_quat = np.array(self.cube_pose_left_local[1])
        box_right_xyz = np.array(self.cube_pose_right_local[0])
        box_right_quat = np.array(self.cube_pose_right_local[1])

        # For simplicity, use the left arm's view of the cube for left arm trajectory
        # and right arm's view for right arm trajectory
        box_xyz = box_left_xyz  # Use left arm's local frame for left arm trajectory

        # Use provided meeting poses or fall back to defaults
        if self.meeting_point is not None:
            meeting_pose_left, meeting_pose_right = self.meeting_point
            meet_left_xyz = np.array(meeting_pose_left[0])
            meet_left_quat = np.array(meeting_pose_left[1])
            meet_right_xyz = np.array(meeting_pose_right[0])
            meet_right_quat = np.array(meeting_pose_right[1])
        else:
            # Fallback to hardcoded values if meeting point not provided
            meet_xyz = np.array([0.1, 0.0, 0.3])  # Center meeting point
            meet_left_xyz = meet_xyz + np.array([0.0, 0.0, -0.12])  # Left meeting point
            meet_right_xyz = meet_xyz + np.array([0.0, 0.0, 0.12])  # Right meeting point

            # Define orientations
            meeting_left_euler = (np.pi/2, 0, -np.pi/2 + np.pi/2)  # meeting_pose_left_delta
            meet_left_quat = R.from_euler('xyz', meeting_left_euler).as_quat()
            meeting_right_euler = (np.pi/2 + np.pi, 0, 0)  # meeting_pose_right_delta
            meet_right_quat = R.from_euler('xyz', meeting_right_euler).as_quat()

        # Define orientations for cube reaching
        cube_reach_euler = (0, np.pi, 0)  # For left arm reaching to cube
        cube_reach_quat = R.from_euler('xyz', cube_reach_euler).as_quat()

        zero_quat = np.array([0, 0, 0, 1])  # Identity quaternion for zero position

        gripper_close = 0.02 - 0.001
        gripper_open = 0.035

        # Left arm waypoints (7 steps as requested)
        self.left_trajectory = [
            {"t": 0, "xyz": np.array([0, 0, 0]), "quat": zero_quat, "gripper": 0.035},           # Reach to cube
            {"t": 50, "xyz": box_xyz + np.array([0, 0, 0.09]), "quat": cube_reach_quat, "gripper": 0.035},           # Reach to cube
            {"t": 100, "xyz": box_xyz + np.array([0, 0, 0.09]), "quat": cube_reach_quat, "gripper": gripper_close},        # Close gripper tightly
            {"t": 400, "xyz": meet_left_xyz + np.array([0, 0.05, 0]), "quat": meet_left_quat, "gripper": gripper_close},   # Move to premeeting point left
            {"t": 500, "xyz": meet_left_xyz, "quat": meet_left_quat, "gripper": gripper_close},                            # Move to meeting point left
            {"t": 600, "xyz": meet_left_xyz, "quat": meet_left_quat, "gripper": gripper_close},                            # Wait for right arm
            {"t": 750, "xyz": meet_left_xyz, "quat": meet_left_quat, "gripper": 0.035},                             # Open gripper
            {"t": 1000, "xyz": np.array([0, 0, 0]), "quat": zero_quat, "gripper": 0.035},                            # Move to zero
        ]

        # Right arm waypoints
        self.right_trajectory = [
            {"t": 0, "xyz": np.array([0, 0, 0]), "quat": zero_quat, "gripper": 0.035},                             # Initial position
            {"t": 100, "xyz": np.array([0, 0, 0]), "quat": zero_quat, "gripper": 0.035},                             # Initial position
            {"t": 400, "xyz": meet_right_xyz + np.array([0, -0.05, 0]), "quat": meet_right_quat, "gripper": 0.035},      # Move to premeeting point right
            {"t": 500, "xyz": meet_right_xyz, "quat": meet_right_quat, "gripper": 0.035},                               # Move to meeting point right
            {"t": 600, "xyz": meet_right_xyz, "quat": meet_right_quat, "gripper": gripper_close},                              # Close gripper tightly
            {"t": 750, "xyz": meet_right_xyz, "quat": meet_right_quat, "gripper": gripper_close},                              # Wait for left arm
            {"t": 1000, "xyz": np.array([0, 0, 0]), "quat": zero_quat, "gripper": gripper_close},                                # Move to zero
        ]


@dataclass(frozen=True)
class DisabledCollision:
    link1: str
    link2: str
    reason: str = ""


@dataclass
class CalibrationData:
    R_gripper2cam: np.ndarray
    t_gripper2cam: np.ndarray
    R_base2top: np.ndarray
    t_base2top: np.ndarray


@dataclass
class VirtualCamera:
    """Virtual camera with no physical representation."""

    image_width: int = 320  # Reduced from 640 for faster rendering
    image_height: int = 240  # Reduced from 480 for faster rendering
    fov: float = 120.0  # Field of view in degrees
    near_plane: float = 0.01
    far_plane: float = 100.0
    # Camera pose relative to robot base (will be computed from calibration)
    position: Optional[np.ndarray] = None
    orientation: Optional[np.ndarray] = None  # Rotation matrix


def load_calibration_data(calib_path: str) -> CalibrationData:
    """Load hand-eye calibration data from JSON file."""
    with open(calib_path, "r") as f:
        data = json.load(f)

    calib_result = data["calib_result"]
    R_gripper2cam = np.array(data["R_gripper2cam"])
    t_gripper2cam = np.array(data["t_gripper2cam"])
    R_base2top = np.array(data["R_base2top"])
    t_base2top = np.array(data["t_base2top"])

    return CalibrationData(
        R_gripper2cam=R_gripper2cam, t_gripper2cam=t_gripper2cam, R_base2top=R_base2top, t_base2top=t_base2top
    )


def load_disabled_collisions(semantics_path: str) -> list[DisabledCollision]:
    try:
        tree = ElementTree.parse(semantics_path)
    except (OSError, ElementTree.ParseError):
        return []

    collisions: list[DisabledCollision] = []
    for elem in tree.getroot().iter("disable_collisions"):
        link1 = elem.attrib.get("link1")
        link2 = elem.attrib.get("link2")
        if not link1 or not link2:
            continue
        collisions.append(DisabledCollision(link1, link2, elem.attrib.get("reason", "")))
    return collisions


class Piper:
    # Shared class variable for the single global client_id
    _client_id: Optional[int] = None

    def __init__(
        self,
        robot_model_path: str,
        semantics_path: Optional[str] = None,
        *,
        gui: bool = False,
        calibration_path: Optional[str] = None,
    ):
        # Initialize the shared client_id only once
        if Piper._client_id is None:
            Piper._client_id = pp.connect(gui=gui)

        self.client_id = Piper._client_id
        self.robot = pp.load_model(robot_model_path, physicsClientId=self.client_id)
        self.robot_joint_ids = [
            pp.joint_from_name(self.robot, name, physicsClientId=self.client_id) for name in robot_joint_names
        ]
        self.gripper_joint_ids = [
            pp.joint_from_name(self.robot, name, physicsClientId=self.client_id) for name in gripper_joint_names
        ]

        if semantics_path is not None:
            disabled = load_disabled_collisions(semantics_path)
            self.disabled_collisions = pp.get_disabled_collisions(self.robot, disabled, physicsClientId=self.client_id)
        else:
            self.disabled_collisions = set()

        self.joint_pos_sampler = None
        self.gripper_pos_sampler = None
        self.collision_fn = None

        # Camera calibration data
        self.calibration: Optional[CalibrationData] = None
        if calibration_path is not None:
            self.calibration = load_calibration_data(calibration_path)

        # Find gripper_base link ID for camera attachment
        try:
            self.gripper_link_id = pp.link_from_name(self.robot, "gripper_base", physicsClientId=self.client_id)
        except KeyError:
            print(f"Warning: gripper_base link not found for robot, using base link")
            self.gripper_link_id = pp.BASE_LINK

        # Virtual camera (no physical representation)
        self.virtual_camera: Optional[VirtualCamera] = None

        # Camera pose caching for performance
        self._last_gripper_pos: Optional[np.ndarray] = None
        self._last_gripper_ori: Optional[np.ndarray] = None
        self._cached_view_matrix: Optional[np.ndarray] = None
        self._cached_projection_matrix: Optional[np.ndarray] = None

    def close(self) -> None:
        pp.disconnect(self.client_id)

    def set_joint_positions(self, joint_positions: list[float]) -> None:
        pp.set_joint_positions(
            self.robot,
            self.robot_joint_ids,
            joint_positions,
            physicsClientId=self.client_id,
        )

    def set_gripper_positions(self, gripper_positions: list[float]) -> None:
        pp.set_joint_positions(
            self.robot,
            self.gripper_joint_ids,
            gripper_positions,
            physicsClientId=self.client_id,
        )

    def set_gripper_position(self, gripper_position: float) -> None:
        self.set_gripper_positions([gripper_position, -gripper_position])

    def get_joint_positions(self) -> list[float]:
        return pp.get_joint_positions(self.robot, self.robot_joint_ids, physicsClientId=self.client_id)

    def get_collision_fn(self, obs: list[int] = []) -> callable:
        self.collision_fn = pp.get_collision_fn(
            self.robot,
            self.robot_joint_ids,
            obstacles=obs,
            disabled_collisions=self.disabled_collisions,
            physicsClientId=self.client_id,
        )
        return self.collision_fn

    def sample_joint_positions(self, dim: int = 1, *, seed: Optional[int] = None) -> np.ndarray:
        if self.joint_pos_sampler is None:
            self.joint_pos_sampler = pp.get_sample_fn(
                self.robot,
                self.robot_joint_ids,
                seed=seed,
                physicsClientId=self.client_id,
            )

        if dim == 1:
            return np.array(self.joint_pos_sampler(), dtype=float)

        samples = [self.joint_pos_sampler() for _ in range(dim)]
        return np.array(samples, dtype=float)

    def sample_gripper_positions(self, dim: int = 1, *, seed: Optional[int] = None) -> np.ndarray:
        if self.gripper_pos_sampler is None:
            self.gripper_pos_sampler = pp.get_sample_fn(
                self.robot,
                self.gripper_joint_ids,
                seed=seed,
                physicsClientId=self.client_id,
            )

        if dim == 1:
            return np.array(self.gripper_pos_sampler(), dtype=float)

        samples = [self.gripper_pos_sampler() for _ in range(dim)]
        return np.array(samples, dtype=float)

    def check_collision(self, joint_positions: list[float]) -> bool:
        if self.collision_fn is None:
            self.get_collision_fn()
        return bool(self.collision_fn(joint_positions))

    def get_joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        lower, upper = pp.get_custom_limits(
            self.robot,
            self.robot_joint_ids,
            circular_limits=pp.CIRCULAR_LIMITS,
            physicsClientId=self.client_id,
        )
        return np.array(lower, dtype=float), np.array(upper, dtype=float)

    def setup_virtual_camera(self) -> None:
        """Set up virtual camera using calibration data."""
        if self.calibration is None:
            raise ValueError("Calibration data required to setup virtual camera")

        self.virtual_camera = VirtualCamera()

        # Camera pose will be computed relative to gripper_base link
        # The calibration data gives us the transform from gripper to camera
        # So camera_pose = gripper_world_pose * gripper_to_camera_transform

        # Store the gripper-to-camera transform from calibration
        self.virtual_camera.position = self.calibration.t_gripper2cam
        self.virtual_camera.orientation = self.calibration.R_gripper2cam

        # Clear any cached camera matrices
        self._clear_camera_cache()

    def _clear_camera_cache(self) -> None:
        """Clear cached camera transformation matrices."""
        self._last_gripper_pos = None
        self._last_gripper_ori = None
        self._cached_view_matrix = None
        self._cached_projection_matrix = None

    def get_camera_image(self, include_depth=True) -> tuple[np.ndarray, np.ndarray]:
        """Get RGB and optionally depth images from the virtual camera.

        Args:
            include_depth: Whether to compute and return depth image (default: True)
        """
        if self.virtual_camera is None:
            raise ValueError("Virtual camera not set up")

        # Get current gripper_base link pose
        gripper_pos, gripper_ori = pp.get_link_pose(self.robot, self.gripper_link_id, physicsClientId=self.client_id)

        # Check if we can use cached matrices (if gripper pose hasn't changed significantly)
        pose_changed = (self._last_gripper_pos is None or
                       self._last_gripper_ori is None or
                       not np.allclose(gripper_pos, self._last_gripper_pos, atol=1e-4) or
                       not np.allclose(gripper_ori, self._last_gripper_ori, atol=1e-4))

        if pose_changed or self._cached_view_matrix is None:
            # Compute camera pose and matrices
            # Convert quaternion to rotation matrix for transformation composition
            # PyBullet uses quaternion [x,y,z,w], scipy uses [w,x,y,z]
            gripper_rot = R.from_quat([gripper_ori[0], gripper_ori[1], gripper_ori[2], gripper_ori[3]])

            # Camera pose in gripper frame (from calibration)
            cam_rot_gripper = R.from_matrix(self.virtual_camera.orientation)
            cam_pos_gripper = self.virtual_camera.position

            # Compose transformations: gripper_world * gripper_to_camera
            cam_rot_world = gripper_rot * cam_rot_gripper
            cam_pos_world = gripper_pos + gripper_rot.apply(cam_pos_gripper)

            # Convert back to quaternion for PyBullet
            cam_ori_world = cam_rot_world.as_quat()  # Returns [x,y,z,w]

            # pp.draw_frame((cam_pos_world, cam_ori_world), 0.2, physicsClientId=self.client_id)

            # For camera view matrix, we need eye position and a target direction
            # Use the camera's forward direction to determine target
            forward_dir = cam_rot_world.apply(np.array([0, 0, 1]))  # +Z is forward in camera frame
            camera_target = cam_pos_world + forward_dir * 1  # Look 1 unit forward

            # Create view matrix
            self._cached_view_matrix = pp.compute_view_matrix(
                cameraEyePosition=cam_pos_world,
                cameraTargetPosition=camera_target,
                cameraUpVector=cam_rot_world.apply(np.array([0, -1, 0])),  # Y down in camera frame
                physicsClientId=self.client_id,
            )

            # Cache projection matrix (this doesn't change unless camera intrinsics change)
            if self._cached_projection_matrix is None:
                self._cached_projection_matrix = pp.compute_projection_matrix_fov(
                    fov=self.virtual_camera.fov,
                    aspect=self.virtual_camera.image_width / self.virtual_camera.image_height,
                    nearVal=self.virtual_camera.near_plane,
                    farVal=self.virtual_camera.far_plane,
                )

            # Update cached gripper pose
            self._last_gripper_pos = gripper_pos.copy()
            self._last_gripper_ori = gripper_ori.copy()

        view_matrix = self._cached_view_matrix

        # Get camera image - only request depth if needed
        renderer = p.ER_BULLET_HARDWARE_OPENGL if include_depth else p.ER_TINY_RENDERER
        images = pp.get_camera_image(
            width=self.virtual_camera.image_width,
            height=self.virtual_camera.image_height,
            viewMatrix=view_matrix,
            projectionMatrix=self._cached_projection_matrix,
            renderer=renderer,
            physicsClientId=self.client_id,
        )

        # Extract RGB image
        rgb_image = np.array(images[2]).reshape(self.virtual_camera.image_height, self.virtual_camera.image_width, 4)[
            :, :, :3
        ]  # Remove alpha channel

        if include_depth:
            depth_image = np.array(images[3]).reshape(self.virtual_camera.image_height, self.virtual_camera.image_width)
        else:
            depth_image = None

        return rgb_image, depth_image

    def get_top_camera_image(self, top_camera: VirtualCamera, include_depth=True) -> tuple[np.ndarray, np.ndarray]:
        """Get RGB and optionally depth images from the top camera positioned relative to left arm base.

        Args:
            top_camera: VirtualCamera configuration
            include_depth: Whether to compute and return depth image (default: True)
        """
        # Top camera is positioned relative to left arm base
        left_base_pos, left_base_ori = pp.get_base_pose(self.robot, physicsClientId=self.client_id)

        # Convert quaternion to rotation matrix for transformation composition
        from scipy.spatial.transform import Rotation as R

        left_base_rot = R.from_quat([left_base_ori[0], left_base_ori[1], left_base_ori[2], left_base_ori[3]])

        # Top camera pose relative to left base
        top_rot_left = R.from_matrix(top_camera.orientation)
        top_pos_left = top_camera.position

        # Compose transformations: left_base_world * left_base_to_top
        top_rot_world = left_base_rot * top_rot_left
        top_pos_world = left_base_pos + left_base_rot.apply(top_pos_left)

        # Convert back to quaternion
        top_ori_world = top_rot_world.as_quat()  # [x,y,z,w]

        # For camera view matrix, we need eye position and a target direction
        # Top camera looks down (negative Z direction in camera frame)
        forward_dir = top_rot_world.apply(np.array([0, 0, 1]))  # +Z is forward in camera frame
        camera_target = top_pos_world + forward_dir * 1.0

        # Create view matrix
        view_matrix = pp.compute_view_matrix(
            cameraEyePosition=top_pos_world,
            cameraTargetPosition=camera_target,
            cameraUpVector=top_rot_world.apply(np.array([0, -1, 0])),  # Y down in camera frame
            physicsClientId=self.client_id,
        )

        # Get camera image - only request depth if needed
        renderer = p.ER_BULLET_HARDWARE_OPENGL if include_depth else p.ER_TINY_RENDERER
        images = pp.get_camera_image(
            width=top_camera.image_width,
            height=top_camera.image_height,
            viewMatrix=view_matrix,
            projectionMatrix=pp.compute_projection_matrix_fov(
                fov=top_camera.fov,
                aspect=top_camera.image_width / top_camera.image_height,
                nearVal=top_camera.near_plane,
                farVal=top_camera.far_plane,
            ),
            renderer=renderer,
            physicsClientId=self.client_id,
        )

        # Extract RGB image
        rgb_image = np.array(images[2]).reshape(top_camera.image_height, top_camera.image_width, 4)[
            :, :, :3
        ]  # Remove alpha channel

        if include_depth:
            depth_image = np.array(images[3]).reshape(top_camera.image_height, top_camera.image_width)
        else:
            depth_image = None

        return rgb_image, depth_image


def command_gripper(robot_id, joint_ids, targets, max_force=80, physics_client_id=0):
    """Control gripper joints using motor control with position control mode.

    Args:
        robot_id: The robot body ID
        joint_ids: List of joint IDs to control
        targets: List of target positions for each joint
        max_force: Maximum force to apply (default 80)
        physics_client_id: PyBullet physics client ID
    """
    import pybullet as p
    for jid, tgt in zip(joint_ids, targets):
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=jid,
            controlMode=p.POSITION_CONTROL,
            targetPosition=tgt,
            force=max_force,
            physicsClientId=physics_client_id,
        )

def command_joints(robot_id, joint_ids, targets, max_force=80, physics_client_id=0):
    """Control joints using motor control with position control mode.

    Args:
        robot_id: The robot body ID
        joint_ids: List of joint IDs to control
        targets: List of target positions for each joint
        max_force: Maximum force to apply (default 80)
        physics_client_id: PyBullet physics client ID
    """
    import pybullet as p
    for jid, tgt in zip(joint_ids, targets):
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=jid,
            controlMode=p.POSITION_CONTROL,
            targetPosition=tgt,
            force=max_force,
            physicsClientId=physics_client_id,
        )


class CubeTransferSimulator:
    def __init__(self, robot_model_path=None, semantics_path=None, left_calib_path=None, right_calib_path=None,
                 gui=True, camera_width=320, camera_height=240, camera_capture_interval=10):
        """Initialize cube transfer simulator with configurable camera settings.

        Args:
            robot_model_path: Path to robot URDF file
            semantics_path: Path to robot semantics file
            left_calib_path: Path to left camera calibration file
            right_calib_path: Path to right camera calibration file
            gui: Whether to show GUI
            camera_width: Camera image width (default: 320 for speed)
            camera_height: Camera image height (default: 240 for speed)
            camera_capture_interval: Capture camera images every N simulation steps (default: 10)
        """
        # File paths (can be overridden)
        self.robot_model_path = robot_model_path or "/home/jeong/zeno/wholebody-teleop/act/assets/piper_description.urdf"
        self.semantics_path = semantics_path or "/home/jeong/zeno/wholebody-teleop/act/assets/piper.srdf"
        self.left_calib_path = left_calib_path or (
            "/home/jeong/zeno/wholebody-teleop/teleop/src/zeno-wholebody-teleop/"
            "robot_side/cam_calibration/data/samples/left/handeye_data_20251216_135728.json"
        )
        self.right_calib_path = right_calib_path or (
            "/home/jeong/zeno/wholebody-teleop/teleop/src/zeno-wholebody-teleop/"
            "robot_side/cam_calibration/data/samples/right/handeye_data_20251216_135743.json"
        )

        # Camera configuration
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_capture_interval = camera_capture_interval

        # Initialize robots
        self.piper_left = Piper(self.robot_model_path, self.semantics_path, calibration_path=self.left_calib_path, gui=gui)
        self.piper_right = Piper(self.robot_model_path, self.semantics_path, calibration_path=self.right_calib_path, gui=gui)

        # Set up physics environment
        pp.load_ground_plane(physicsClientId=0)
        pp.disable_side_panels(physicsClientId=self.piper_left.client_id)
        pp.set_gravity(physicsClientId=self.piper_left.client_id)

        # Set left arm pose to (0, 0, 0) position and (0, 0, 0, 1) orientation
        pp.set_base_pose(self.piper_left.robot, ((0, 0, 0), (0, 0, 0, 1)), physicsClientId=self.piper_left.client_id)

        # Load ground plane
        self.ground_id = pp.load_ground_plane(physicsClientId=self.piper_left.client_id)
        print("Ground plane loaded")

        # Set up virtual cameras for both arms with configurable resolution
        self.piper_left.setup_virtual_camera()
        self.piper_right.setup_virtual_camera()

        # Override default camera resolutions with configured values
        self.piper_left.virtual_camera.image_width = self.camera_width
        self.piper_left.virtual_camera.image_height = self.camera_height
        self.piper_right.virtual_camera.image_width = self.camera_width
        self.piper_right.virtual_camera.image_height = self.camera_height

        # Create top camera using left arm calibration: left_base --> top
        self.top_camera = VirtualCamera(
            image_width=self.camera_width,
            image_height=self.camera_height
        )
        if self.piper_left.calibration is not None:
            # Position top camera relative to left arm base using calibration
            self.top_camera.position = self.piper_left.calibration.t_base2top + np.array([0, 0, 0.3])
            self.top_camera.orientation = self.piper_left.calibration.R_base2top

        # pp.draw_frame(
        #     (self.top_camera.position, R.from_matrix(self.top_camera.orientation).as_quat()),
        #     0.1,
        #     physicsClientId=self.piper_left.client_id,
        # )

        # Set up the relative poses: left arm <--> top cam <--> right arm
        # Compute right arm pose using the chain: left_base --> top + (right_base --> top)^{-1}
        if self.piper_left.calibration is not None and self.piper_right.calibration is not None:
            # Get left arm base pose (already set to (0,0,0) with identity orientation)
            left_base_pos = np.array([0, 0, 0])
            left_base_ori = np.array([0, 0, 0, 1])  # Identity quaternion

            # T_left_base_to_top: transform from left base to top camera
            T_left_to_top_pos = self.piper_left.calibration.t_base2top
            T_left_to_top_rot = R.from_matrix(self.piper_left.calibration.R_base2top)

            # T_right_base_to_top_inv: inverse transform from right base to top camera
            T_right_to_top_pos = self.piper_right.calibration.t_base2top
            T_right_to_top_rot = R.from_matrix(self.piper_right.calibration.R_base2top)

            # Compute inverse transformation
            T_right_to_top_rot_inv = T_right_to_top_rot.inv()
            T_right_to_top_pos_inv = -T_right_to_top_rot_inv.apply(T_right_to_top_pos)

            # Compose transformations: left_base --> top + (right_base --> top)^{-1}
            # First: left_base * T_left_to_top
            intermediate_pos = left_base_pos + T_left_to_top_pos
            intermediate_rot = T_left_to_top_rot

            # Then: intermediate * T_right_to_top_inv
            right_base_pos = intermediate_pos + intermediate_rot.apply(T_right_to_top_pos_inv)
            right_base_rot = intermediate_rot * T_right_to_top_rot_inv

            # Convert rotation back to quaternion
            right_base_ori = right_base_rot.as_quat()  # [x,y,z,w] format

            # Set right arm pose
            pp.set_base_pose(self.piper_right.robot, (right_base_pos, right_base_ori), physicsClientId=self.piper_right.client_id)

            self.right_base_pos = right_base_pos

        print("Left arm joint limits:", self.piper_left.get_joint_limits())
        print("Right arm joint limits:", self.piper_right.get_joint_limits())
        print("Calibration loaded for both arms")
        print("Virtual cameras set up for both arms")

        # Create IK solver
        self.ik_solver = TracIKSolver(self.robot_model_path, "base_link", "gripper_base")

        # Store initial joint positions
        self.left_joint_positions_init = self.piper_left.get_joint_positions()
        self.right_joint_positions_init = self.piper_right.get_joint_positions()

        # Simulation state
        self.cube_id = None
        self.policy = None
        self.current_step = 0
        self.is_recording = False
        self.recorded_data = None
        self.episode_data = []

    def create_cube(self, arm_choice=None):
        """Create a cube randomly in front of left or right arm"""
        if arm_choice is None:
            arm_choice = np.random.choice(['left', 'right'])

        cube_x = np.random.uniform(0.1, 0.3)
        cube_y = np.random.uniform(-0.1, 0.1)
        cube_z = 0.05  # Height above ground
        cube_position = np.array([cube_x, cube_y, cube_z])
        cube_quaternion = np.array([0, 0, 0, 1])  # Identity quaternion

        # Create red cube with high friction for grasping
        self.cube_id = pp.create_shape(
            shape="box",
            size=[0.02, 0.02, 0.02],
            pose=(cube_position, cube_quaternion),
            color=[1, 0, 0, 1],
            mass=0.01,
            friction=100.0,
            physicsClientId=self.piper_left.client_id
        )
        print(f"Created red cube at position: {cube_position}, in front of {arm_choice} arm")

        # Set high friction on gripper fingers for better grasping
        left_gripper_links = [
            pp.link_from_name(self.piper_left.robot, "gripper_base", physicsClientId=self.piper_left.client_id),
            pp.link_from_name(self.piper_left.robot, "link7", physicsClientId=self.piper_left.client_id),
            pp.link_from_name(self.piper_left.robot, "link8", physicsClientId=self.piper_left.client_id)
        ]
        right_gripper_links = [
            pp.link_from_name(self.piper_right.robot, "gripper_base", physicsClientId=self.piper_right.client_id),
            pp.link_from_name(self.piper_right.robot, "link7", physicsClientId=self.piper_right.client_id),
            pp.link_from_name(self.piper_right.robot, "link8", physicsClientId=self.piper_right.client_id)
        ]

        # Set high friction on gripper links
        for link_id in left_gripper_links:
            pp.set_friction(self.piper_left.robot, link_id, 100.0, physicsClientId=self.piper_left.client_id)

        for link_id in right_gripper_links:
            pp.set_friction(self.piper_right.robot, link_id, 100.0, physicsClientId=self.piper_right.client_id)

        print(f"Set high friction on left gripper links: {left_gripper_links}")
        print(f"Set high friction on right gripper links: {right_gripper_links}")

        # Enable better contact handling for grasping
        import pybullet as p
        p.setPhysicsEngineParameter(enableConeFriction=1, physicsClientId=self.piper_left.client_id)
        p.setPhysicsEngineParameter(contactBreakingThreshold=0.001, physicsClientId=self.piper_left.client_id)
        p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=self.piper_left.client_id)

        return cube_position, cube_quaternion

    def setup_meeting_point(self, cube_position):
        """Set up meeting point for cube transfer"""
        meeting_point = self.right_base_pos / 2
        meeting_pose = pp.pose([0.0, meeting_point[1], 0.55], [np.pi/2, 0, -np.pi/2])
        meeting_pose_left_delta = pp.pose([0.0, 0.0, -0.12], [0, 0, np.pi/2])
        meeting_pose_right_delta = pp.pose([0.0, 0.0, 0.12], [np.pi, 0, 0])
        meeting_pose_left = pp.multiply(meeting_pose, meeting_pose_left_delta)
        meeting_pose_right = pp.multiply(meeting_pose, meeting_pose_right_delta)

        # pp.draw_frame(meeting_pose, 0.1, physicsClientId=self.piper_left.client_id)
        # pp.draw_frame(meeting_pose_left, 0.1, physicsClientId=self.piper_left.client_id)
        # pp.draw_frame(meeting_pose_right, 0.1, physicsClientId=self.piper_right.client_id)

        left_base_pose = pp.get_base_pose(self.piper_left.robot, physicsClientId=self.piper_left.client_id)
        right_base_pose = pp.get_base_pose(self.piper_right.robot, physicsClientId=self.piper_right.client_id)

        meeting_pose_left_local = pp.multiply(pp.inverse(left_base_pose), meeting_pose_left)
        meeting_pose_right_local = pp.multiply(pp.inverse(right_base_pose), meeting_pose_right)

        # Transform cube position to local frames for each arm
        cube_pose_world = pp.pose(cube_position, [0, 0, 0])
        cube_pose_left_local = pp.multiply(pp.inverse(left_base_pose), cube_pose_world)
        cube_pose_right_local = pp.multiply(pp.inverse(right_base_pose), cube_pose_world)

        return meeting_pose_left_local, meeting_pose_right_local, cube_pose_left_local, cube_pose_right_local

    def initialize_policy(self, cube_pose_left_local, cube_pose_right_local, meeting_pose_left_local, meeting_pose_right_local):
        """Initialize and precompute the cube transfer policy"""
        # Initialize policy
        self.policy = CubeTransferPolicy(
            cube_pose_left_local,
            cube_pose_right_local,
            meeting_point=(meeting_pose_left_local, meeting_pose_right_local),
            inject_noise=False
        )

        # Precompute the entire joint trajectory using TracIKSolver
        print("Precomputing joint trajectory...")
        self.policy.precompute_joint_trajectory(
            self.piper_left, self.piper_right,
            self.ik_solver, self.ik_solver,
            left_qinit=self.left_joint_positions_init,
            right_qinit=self.right_joint_positions_init
        )

    def record_episode(self, episode_idx=0, dataset_dir="episodes", camera_capture_interval=10):
        """Record an episode of simulation data

        Args:
            episode_idx: Episode index for naming
            dataset_dir: Directory to save episodes
            camera_capture_interval: Capture camera images every N simulation steps (default: 10)
                                   At 240Hz simulation, this gives ~24 FPS camera capture
        """
        if self.policy is None:
            raise ValueError("Policy not initialized. Call initialize_policy() first.")

        if not os.path.isdir(dataset_dir):
            os.makedirs(dataset_dir, exist_ok=True)

        # Reset recording state
        self.current_step = 0
        self.is_recording = True
        self.episode_data = []

        # Camera names for this simulation
        camera_names = ['left_arm', 'right_arm', 'top']

        # Data structure similar to record_sim_episodes.py
        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
            '/timestamp': [],
        }
        # Image data collection is currently disabled
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []

        dt = 1.0 / 240
        max_steps = len(self.policy.precomputed_trajectory)

        print(f"Recording episode {episode_idx} with {max_steps} steps (camera capture every {camera_capture_interval} steps)...")

        with tqdm(total=max_steps-1, desc=f"Episode {episode_idx}") as pbar:
            for step in range(max_steps-1):
                # Get precomputed joint positions and gripper states
                step_data = self.policy.get_precomputed_step(step)

                # Set joint positions
                command_joints(self.piper_left.robot, self.piper_left.robot_joint_ids,
                              step_data['left_joints'], max_force=1000, physics_client_id=self.piper_left.client_id)
                command_joints(self.piper_right.robot, self.piper_right.robot_joint_ids,
                              step_data['right_joints'], max_force=1000, physics_client_id=self.piper_right.client_id)

                # Set gripper positions
                left_gripper_angle = step_data['left_gripper']
                right_gripper_angle = step_data['right_gripper']

                command_gripper(self.piper_left.robot, self.piper_left.gripper_joint_ids,
                               [left_gripper_angle, -left_gripper_angle],
                               max_force=1000, physics_client_id=self.piper_left.client_id)
                command_gripper(self.piper_right.robot, self.piper_right.gripper_joint_ids,
                               [right_gripper_angle, -right_gripper_angle],
                               max_force=1000, physics_client_id=self.piper_right.client_id)

                # Step simulation to update positions
                pp.step_simulation(self.piper_left.client_id)
                time.sleep(dt)

                # Check if cube should be grasped (after gripper closes at step 100)
                # If gripper is closed but cube is not grasped, discard this episode
                if step >= 150 and (left_gripper_angle < 0.021 or right_gripper_angle < 0.021):
                    if not self.check_cube_grasped():
                        print(f"\nEpisode {episode_idx} discarded: Cube not grasped at step {step}")
                        pbar.set_postfix({'status': 'FAILED - Cube not grasped'})
                        return None  # Return None to indicate episode was discarded

                # Get observations
                left_qpos = self.piper_left.get_joint_positions()
                right_qpos = self.piper_right.get_joint_positions()
                qpos = np.concatenate([left_qpos, [left_gripper_angle], right_qpos, [right_gripper_angle]])

                # For qvel, we need to get joint velocities (simplified - using zeros for now)
                # In a real implementation, you'd track velocity over time
                qvel = np.zeros_like(qpos)

                # Get action (joint positions + gripper)
                action = np.concatenate([
                    step_data['left_joints'] + [step_data['left_gripper']],
                    step_data['right_joints'] + [step_data['right_gripper']]
                ])

                # Store basic data (always captured)
                timestamp = step * dt
                data_dict['/observations/qpos'].append(qpos)
                data_dict['/observations/qvel'].append(qvel)
                data_dict['/action'].append(action)
                data_dict['/timestamp'].append(timestamp)

                # Get camera images only every N steps to reduce computational load
                # Skip depth computation for speed
                if step % camera_capture_interval == 0:
                    left_camera_image, _ = self.piper_left.get_camera_image(include_depth=False)
                    right_camera_image, _ = self.piper_right.get_camera_image(include_depth=False)
                    top_camera_image, _ = self.piper_left.get_top_camera_image(self.top_camera, include_depth=False)

                    data_dict['/observations/images/left_arm'].append(left_camera_image)
                    data_dict['/observations/images/right_arm'].append(right_camera_image)
                    data_dict['/observations/images/top'].append(top_camera_image)
                else:
                    # Append None or duplicate previous frame to maintain array length
                    # For now, we'll duplicate the last frame (this could be optimized further)
                    if len(data_dict['/observations/images/left_arm']) > 0:
                        last_left = data_dict['/observations/images/left_arm'][-1]
                        last_right = data_dict['/observations/images/right_arm'][-1]
                        last_top = data_dict['/observations/images/top'][-1]
                        data_dict['/observations/images/left_arm'].append(last_left)
                        data_dict['/observations/images/right_arm'].append(last_right)
                        data_dict['/observations/images/top'].append(last_top)
                    else:
                        # First step, capture anyway (skip depth for speed)
                        left_camera_image, _ = self.piper_left.get_camera_image(include_depth=False)
                        right_camera_image, _ = self.piper_right.get_camera_image(include_depth=False)
                        top_camera_image, _ = self.piper_left.get_top_camera_image(self.top_camera, include_depth=False)
                        data_dict['/observations/images/left_arm'].append(left_camera_image)
                        data_dict['/observations/images/right_arm'].append(right_camera_image)
                        data_dict['/observations/images/top'].append(top_camera_image)

                # Update tqdm postfix with gripper values
                pbar.set_postfix({
                    'left_gripper': f"{step_data['left_gripper']:.2f}",
                    'right_gripper': f"{step_data['right_gripper']:.2f}"
                })
                pbar.update(1)

                if step == max_steps-1:
                    print(f"Episode {episode_idx} completed successfully")
                    break

        # Only save episode if it completed successfully (cube was grasped)
        if len(data_dict['/observations/qpos']) > 0:
            # Save episode data to HDF5
            import h5py
            t0 = time.time()
            dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}')
            actual_steps = len(data_dict['/observations/qpos'])
            with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
                root.attrs['sim'] = True
                obs = root.create_group('observations')
                image = obs.create_group('images')
                for cam_name in camera_names:
                    _ = image.create_dataset(cam_name, (actual_steps, 480, 640, 3), dtype='uint8', chunks=(1, 480, 640, 3))
                qpos_dataset = obs.create_dataset('qpos', (actual_steps, 14))
                qvel_dataset = obs.create_dataset('qvel', (actual_steps, 14))
                action_dataset = root.create_dataset('action', (actual_steps, 14))
                timestamp_dataset = root.create_dataset('timestamp', (actual_steps,))

                for name, array in data_dict.items():
                    root[name][...] = array

            print(f'Saving episode {episode_idx}: {timestamp:.3f} secs')
            self.is_recording = False
            return dataset_path + '.hdf5'
        else:
            # Episode was discarded
            self.is_recording = False
            return None

    def restart(self):
        """Restart the simulation environment"""
        # Reset robot positions to initial state
        pp.set_base_pose(self.piper_left.robot, ((0, 0, 0), (0, 0, 0, 1)), physicsClientId=self.piper_left.client_id)
        if hasattr(self, 'right_base_pos'):
            right_base_rot = R.from_quat([0, 0, 0, 1])  # Identity quaternion
            pp.set_base_pose(self.piper_right.robot, (self.right_base_pos, right_base_rot.as_quat()),
                           physicsClientId=self.piper_right.client_id)

        # Reset joint positions
        self.piper_left.set_joint_positions(self.left_joint_positions_init)
        self.piper_right.set_joint_positions(self.right_joint_positions_init)

        # Reset grippers
        self.piper_left.set_gripper_position(0.035)
        self.piper_right.set_gripper_position(0.035)

        # Remove existing cube if it exists
        if self.cube_id is not None:
            pp.remove_body(self.cube_id, physicsClientId=self.piper_left.client_id)
            self.cube_id = None

        # Clear camera caches
        self.piper_left._clear_camera_cache()
        self.piper_right._clear_camera_cache()

        # Reset simulation state
        self.policy = None
        self.current_step = 0
        self.is_recording = False
        self.episode_data = []

        print("Simulation restarted")

    def reset(self):
        """Reset the simulation to initial state and create new cube"""
        self.restart()

        # Create new cube
        cube_position, _ = self.create_cube()

        # Set up meeting point and initialize policy
        meeting_pose_left_local, meeting_pose_right_local, cube_pose_left_local, cube_pose_right_local = self.setup_meeting_point(cube_position)
        self.initialize_policy(cube_pose_left_local, cube_pose_right_local, meeting_pose_left_local, meeting_pose_right_local)

        print("Simulation reset with new cube")

    def check_cube_grasped(self):
        """Check if the cube is grasped by any gripper"""
        if self.cube_id is None:
            return False

        # Get gripper finger link IDs
        left_gripper_links = [
            pp.link_from_name(self.piper_left.robot, "gripper_base", physicsClientId=self.piper_left.client_id),
            pp.link_from_name(self.piper_left.robot, "link7", physicsClientId=self.piper_left.client_id),
            pp.link_from_name(self.piper_left.robot, "link8", physicsClientId=self.piper_left.client_id)
        ]
        right_gripper_links = [
            pp.link_from_name(self.piper_right.robot, "gripper_base", physicsClientId=self.piper_right.client_id),
            pp.link_from_name(self.piper_right.robot, "link7", physicsClientId=self.piper_right.client_id),
            pp.link_from_name(self.piper_right.robot, "link8", physicsClientId=self.piper_right.client_id)
        ]

        # Check contacts between cube and gripper fingers
        import pybullet as p
        for link_id in left_gripper_links:
            # Get contact points between this link and the cube
            contact_points = p.getContactPoints(
                bodyA=self.piper_left.robot,
                bodyB=self.cube_id,
                linkIndexA=link_id,
                physicsClientId=self.piper_left.client_id
            )
            if contact_points:  # If there are any contact points
                return True

        for link_id in right_gripper_links:
            # Get contact points between this link and the cube
            contact_points = p.getContactPoints(
                bodyA=self.piper_right.robot,
                bodyB=self.cube_id,
                linkIndexA=link_id,
                physicsClientId=self.piper_right.client_id
            )
            if contact_points:  # If there are any contact points
                return True

        return False

    def close(self):
        """Clean up resources"""
        if hasattr(self, 'piper_left'):
            self.piper_left.close()


if __name__ == "__main__":
    # Create simulator instance
    simulator = CubeTransferSimulator(gui=True, camera_width=640, camera_height=480, camera_capture_interval=4)
    index = 0
    while True:

        print("=" * 100)
        print(f"Resetting simulator and recording episode {index}...")
        print("=" * 100)

        # Reset to create initial cube and setup
        simulator.reset()

        # Record an episode with configured camera settings
        episode_path = simulator.record_episode(episode_idx=index, dataset_dir="dataset/sim_transfer_cube_pyb",
                                               camera_capture_interval=simulator.camera_capture_interval)
        if episode_path is not None:
            print(f"Episode recorded to: {episode_path}")
            index += 1
        else:
            print("Episode discarded due to failed grasping")
            
        if index > 100:
            break

    # Clean up
    simulator.close()
