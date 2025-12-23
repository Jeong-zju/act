import time
import numpy as np
import collections
from scipy.spatial.transform import Rotation as R
from types import SimpleNamespace

# Import constants from sim_env.py
import sys
import os
sys.path.append(os.path.dirname(__file__))
from constants import PIPER_GRIPPER_POSITION_NORMALIZE_FN
from constants import PIPER_GRIPPER_POSITION_UNNORMALIZE_FN
from constants import PIPER_GRIPPER_VELOCITY_NORMALIZE_FN
from constants import PIPER_GRIPPER_POSITION_OPEN
from constants import PIPER_GRIPPER_POSITION_CLOSE
from constants import XML_DIR

import matplotlib.pyplot as plt
import h5py
render_cam_name = 'top'

# Try to import MuJoCo and TracIKSolver (optional for testing)
try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    print("MuJoCo not available - running in test mode")

try:
    from tracikpy import TracIKSolver
    TRACIK_AVAILABLE = True
except ImportError:
    TRACIK_AVAILABLE = False
    print("TracIKSolver not available - IK will not work")

class BasePolicy:
    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.left_trajectory = None
        self.right_trajectory = None

    def generate_trajectory(self, ts_first):
        raise NotImplementedError

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

    def __call__(self, ts):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory(ts)

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

        action_left = np.concatenate([left_xyz, left_quat, [left_gripper]])
        action_right = np.concatenate([right_xyz, right_quat, [right_gripper]])

        self.step_count += 1
        return np.concatenate([action_left, action_right])


class PickAndTransferPolicyPiper(BasePolicy):
    """Policy for piper robots using joint-space interpolation with IK solver"""
    
    def __init__(self, inject_noise=False, robot_model_path=None):
        super().__init__(inject_noise)
        self.precomputed_trajectory = None
        
        # Set up IK solvers if available
        if TRACIK_AVAILABLE:
            robot_model_path = robot_model_path or "/home/jeong/zeno/wholebody-teleop/act/assets/piper_description.urdf"
            self.ik_solver = TracIKSolver(robot_model_path, "base_link", "gripper_base")
        else:
            self.ik_solver = None
            print("Warning: TracIKSolver not available. PickAndTransferPolicyPiper requires IK solver.")

    def pose_to_matrix(self, xyz, quat):
        """Convert position and quaternion to 4x4 transformation matrix
        Handles both pyquaternion format [w, x, y, z] and scipy format [x, y, z, w]
        """
        quat = np.array(quat)
        quat_scipy = np.array([quat[1], quat[2], quat[3], quat[0]])
        rot_matrix = R.from_quat(quat_scipy).as_matrix()
        matrix = np.eye(4)
        matrix[:3, :3] = rot_matrix
        matrix[:3, 3] = xyz
        return matrix

    def generate_trajectory(self, ts_first):
        """Generate Cartesian waypoint trajectory (same as PickAndTransferPolicy)"""
        # init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        # init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        world2left_pos = np.array([0, 0, 0])
        world2left_quat = np.array([1, 0, 0, 0])
        self.world2left_mat = self.pose_to_matrix(world2left_pos, world2left_quat)

        world2right_pos = np.array([0, -0.56376582, 0])
        world2right_quat = np.array([1, 0, 0, 0])
        self.world2right_mat = self.pose_to_matrix(world2right_pos, world2right_quat)

        box_info = np.array(ts_first.observation['env_state'])
        box_xyz = box_info[:3]
        box_quat = box_info[3:]
        # print(f"Generate trajectory for {box_xyz=}")

        # Meeting points matching piper_mj_record.py
        meet_xyz = np.array([0, -0.28188, 0.3])  # Center meeting point
        meet_left_xyz = np.array([0, -0.28188, 0.3]) + np.array([0.0, 0.125, 0.0])  # Left meeting point
        meet_right_xyz = np.array([0, -0.28188, 0.3]) + np.array([0.0, -0.125, 0.0])  # Right meeting point

        # Define orientations matching piper_mj_record.py
        # Left: euler (np.pi/2, 0, 0) -> quaternion [w, x, y, z] for pyquaternion
        meeting_left_euler = (np.pi/2, 0, 0)
        meet_left_quat_scipy = R.from_euler('xyz', meeting_left_euler).as_quat()  # [x, y, z, w]
        meet_left_quat = np.array([meet_left_quat_scipy[3], meet_left_quat_scipy[0], meet_left_quat_scipy[1], meet_left_quat_scipy[2]])  # Convert to [w, x, y, z]

        world2leftmeet_mat = self.pose_to_matrix(meet_left_xyz, meet_left_quat)
        left2leftmeet_mat = np.linalg.inv(self.world2left_mat) @ world2leftmeet_mat
        left2leftmeet_xyz = np.array([left2leftmeet_mat[0, 3], left2leftmeet_mat[1, 3], left2leftmeet_mat[2, 3]])
        left2leftmeet_quat_scipy = R.from_matrix(left2leftmeet_mat[:3, :3]).as_quat()
        left2leftmeet_quat = np.array([left2leftmeet_quat_scipy[3], left2leftmeet_quat_scipy[0], left2leftmeet_quat_scipy[1], left2leftmeet_quat_scipy[2]])  # Convert to [w, x, y, z]
        
        # Right: euler (0, np.pi/2, np.pi/2) -> quaternion [w, x, y, z] for pyquaternion
        meeting_right_euler = (0, np.pi/2, np.pi/2)
        meet_right_quat_scipy = R.from_euler('xyz', meeting_right_euler).as_quat()  # [x, y, z, w]
        meet_right_quat = np.array([meet_right_quat_scipy[3], meet_right_quat_scipy[0], meet_right_quat_scipy[1], meet_right_quat_scipy[2]])  # Convert to [w, x, y, z]

        world2rightmeet_mat = self.pose_to_matrix(meet_right_xyz, meet_right_quat)
        right2rightmeet_mat = np.linalg.inv(self.world2right_mat) @ world2rightmeet_mat
        right2rightmeet_xyz = np.array([right2rightmeet_mat[0, 3], right2rightmeet_mat[1, 3], right2rightmeet_mat[2, 3]])
        right2rightmeet_quat_scipy = R.from_matrix(right2rightmeet_mat[:3, :3]).as_quat()
        right2rightmeet_quat = np.array([right2rightmeet_quat_scipy[3], right2rightmeet_quat_scipy[0], right2rightmeet_quat_scipy[1], right2rightmeet_quat_scipy[2]])  # Convert to [w, x, y, z]

        # Define orientations for cube reaching (matching piper_mj_record.py)
        cube_reach_euler = (0, np.pi, 0)  # For left arm reaching to cube
        cube_reach_quat_scipy = R.from_euler('xyz', cube_reach_euler).as_quat()  # [x, y, z, w]
        cube_reach_quat = np.array([cube_reach_quat_scipy[3], cube_reach_quat_scipy[0], cube_reach_quat_scipy[1], cube_reach_quat_scipy[2]])  # Convert to [w, x, y, z]

        gripper_close = 0.005
        gripper_open = 0.035

        # Left arm waypoints matching piper_mj_record.py exactly
        self.left_trajectory = [
            {"t": 0, "joint": [0, 0, 0, 0, 0, 0], "gripper": gripper_open},
            {"t": 100, "xyz": box_xyz + np.array([0, 0, 0.195]), "quat": cube_reach_quat, "gripper": gripper_open},
            {"t": 200, "xyz": box_xyz + np.array([0, 0, 0.095]), "quat": cube_reach_quat, "gripper": gripper_open},
            {"t": 300, "xyz": box_xyz + np.array([0, 0, 0.095]), "quat": cube_reach_quat, "gripper": gripper_close},
            {"t": 600, "xyz": left2leftmeet_xyz + np.array([0, 0.05, 0]), "quat": left2leftmeet_quat, "gripper": gripper_close},
            {"t": 700, "xyz": left2leftmeet_xyz, "quat": left2leftmeet_quat, "gripper": gripper_close},
            {"t": 800, "xyz": left2leftmeet_xyz, "quat": left2leftmeet_quat, "gripper": gripper_close},
            {"t": 900, "xyz": left2leftmeet_xyz, "quat": left2leftmeet_quat, "gripper": gripper_open},
            {"t": 1200, "joint": [0, 0, 0, 0, 0, 0], "gripper": gripper_open},
        ]

        # Right arm waypoints matching piper_mj_record.py exactly
        self.right_trajectory = [
            {"t": 0, "joint": [0, 0, 0, 0, 0, 0], "gripper": gripper_open},
            {"t": 600, "xyz": right2rightmeet_xyz + np.array([0, -0.05, 0]), "quat": right2rightmeet_quat, "gripper": gripper_open},
            {"t": 700, "xyz": right2rightmeet_xyz, "quat": right2rightmeet_quat, "gripper": gripper_open},
            {"t": 800, "xyz": right2rightmeet_xyz, "quat": right2rightmeet_quat, "gripper": gripper_close},
            {"t": 900, "xyz": right2rightmeet_xyz, "quat": right2rightmeet_quat, "gripper": gripper_close},
            {"t": 1200, "joint": [0, 0, 0, 0, 0, 0], "gripper": gripper_close}
        ]

        self.precompute_joint_trajectory(ts_first)

        # self.precompute_cart_trajectory(ts_first)

    def precompute_joint_trajectory(self, ts_first):
        """Precompute the entire joint trajectory using joint-space interpolation"""
        if self.left_trajectory is None or self.right_trajectory is None:
            raise ValueError("Trajectory not generated. Call generate_trajectory first.")

        # Find the maximum timestep in the trajectory
        max_timestep = max(
            max(wp['t'] for wp in self.left_trajectory),
            max(wp['t'] for wp in self.right_trajectory)
        )

        # Initialize precomputed trajectory
        self.precomputed_trajectory = []

        # Get initial joint positions from observation
        qpos = ts_first.observation['qpos']
        left_qinit = qpos[:6].tolist()  # First 6 joints for left arm
        right_qinit = qpos[7:13].tolist()  # Joints 7-12 for right arm (skip gripper at index 6)

        # First, convert all waypoints to joint space
        left_joint_waypoints = []
        right_joint_waypoints = []

        for wp in self.left_trajectory:
            if "joint" in wp:
                left_joint_waypoints.append({
                    't': wp['t'],
                    'joints': wp['joint'],
                    'gripper': wp['gripper']
                })
                continue

            # Convert Cartesian waypoint to joint position
            left_pose_matrix = self.pose_to_matrix(wp['xyz'], wp['quat'])

            try:
                joint_pos = self.ik_solver.ik(left_pose_matrix, qinit=left_qinit)
                if joint_pos is None:
                    print(f"Left IK failed for waypoint at t={wp['t']}, using fallback")
                    joint_pos = left_qinit.copy()  # Use previous position as fallback
                else:
                    left_qinit = joint_pos.tolist()  # Update for next waypoint
            except Exception as e:
                print(f"Left IK failed for waypoint at t={wp['t']}: {e}")
                joint_pos = np.array(left_qinit)  # Use previous position as fallback

            left_joint_waypoints.append({
                't': wp['t'],
                'joints': joint_pos.tolist() if isinstance(joint_pos, np.ndarray) else joint_pos,
                'gripper': wp['gripper']
            })

        for wp in self.right_trajectory:
            if "joint" in wp:
                right_joint_waypoints.append({
                    't': wp['t'],
                    'joints': wp['joint'],
                    'gripper': wp['gripper']
                })
                continue
            # Convert Cartesian waypoint to joint position
            right_pose_matrix = self.pose_to_matrix(wp['xyz'], wp['quat'])

            try:
                joint_pos = self.ik_solver.ik(right_pose_matrix, qinit=right_qinit)
                if joint_pos is None:
                    print(f"Right IK failed for waypoint at t={wp['t']}, using fallback")
                    joint_pos = right_qinit.copy()  # Use previous position as fallback
                else:
                    right_qinit = joint_pos.tolist()  # Update for next waypoint
            except Exception as e:
                print(f"Right IK failed for waypoint at t={wp['t']}: {e}")
                joint_pos = np.array(right_qinit)  # Use previous position as fallback

            right_joint_waypoints.append({
                't': wp['t'],
                'joints': joint_pos.tolist() if isinstance(joint_pos, np.ndarray) else joint_pos,
                'gripper': wp['gripper']
            })

        # Now interpolate in joint space for each timestep
        for t in range(max_timestep + 1):
            # Get interpolated joint positions and gripper values
            left_joints, left_gripper = self._get_interpolated_joints(t, left_joint_waypoints)
            right_joints, right_gripper = self._get_interpolated_joints(t, right_joint_waypoints)

            # Convert joint positions back to Cartesian using FK
            try:
                left_pose_matrix = self.ik_solver.fk(left_joints)
                left_xyz = left_pose_matrix[:3, 3]
                # scipy returns [x, y, z, w], convert to MuJoCo format [w, x, y, z]
                left_quat_scipy = R.from_matrix(left_pose_matrix[:3, :3]).as_quat()
                left_quat = np.array([left_quat_scipy[3], left_quat_scipy[0], left_quat_scipy[1], left_quat_scipy[2]])
            except Exception as e:
                # Fallback to Cartesian interpolation if FK fails
                left_xyz, left_quat, _ = self._get_interpolated_pose(t, 'left')

            try:
                right_pose_matrix = self.ik_solver.fk(right_joints)
                right_xyz = right_pose_matrix[:3, 3]
                # scipy returns [x, y, z, w], convert to MuJoCo format [w, x, y, z]
                right_quat_scipy = R.from_matrix(right_pose_matrix[:3, :3]).as_quat()
                right_quat = np.array([right_quat_scipy[3], right_quat_scipy[0], right_quat_scipy[1], right_quat_scipy[2]])
            except Exception as e:
                # Fallback to Cartesian interpolation if FK fails
                right_xyz, right_quat, _ = self._get_interpolated_pose(t, 'right')

            # Add noise if requested
            if self.inject_noise:
                scale = 0.01
                left_joints = np.array(left_joints) + np.random.uniform(-scale, scale, len(left_joints))
                right_joints = np.array(right_joints) + np.random.uniform(-scale, scale, len(right_joints))
                left_joints = left_joints.tolist()
                right_joints = right_joints.tolist()
                left_xyz = left_xyz + np.random.uniform(-scale, scale, left_xyz.shape)
                right_xyz = right_xyz + np.random.uniform(-scale, scale, right_xyz.shape)

            # Store the complete state for this timestep
            self.precomputed_trajectory.append({
                'left_joints': left_joints,
                'right_joints': right_joints,
                'left_gripper': left_gripper,
                'right_gripper': right_gripper,
                'left_xyz': left_xyz,
                'right_xyz': right_xyz,
                'left_quat': left_quat,
                'right_quat': right_quat
            })

        print(f"Precomputed joint-space trajectory with {len(self.precomputed_trajectory)} timesteps")

    def precompute_cart_trajectory(self, ts_first):
        max_timestep = max(
            max(wp['t'] for wp in self.left_trajectory),
            max(wp['t'] for wp in self.right_trajectory)
        )
        self.precomputed_trajectory = []
        for t in range(max_timestep + 1):
            left_xyz, left_quat, left_gripper = self._get_interpolated_pose(t, 'left')
            right_xyz, right_quat, right_gripper = self._get_interpolated_pose(t, 'right')
            self.precomputed_trajectory.append({'left_xyz': left_xyz, 'left_quat': left_quat, 'left_gripper': left_gripper, 'right_xyz': right_xyz, 'right_quat': right_quat, 'right_gripper': right_gripper})
        print(f"Precomputed Cartesian trajectory with {len(self.precomputed_trajectory)} timesteps")

    def _get_interpolated_pose(self, t, arm):
        """Get interpolated pose for a specific timestep and arm (Cartesian interpolation)"""
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
            curr_quat = np.array(curr_waypoint['quat'])
            curr_grip = curr_waypoint['gripper']
            next_xyz = next_waypoint['xyz']
            next_quat = np.array(next_waypoint['quat'])
            next_grip = next_waypoint['gripper']
            
            # Linear interpolation for position and gripper
            xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
            gripper = curr_grip + (next_grip - curr_grip) * t_frac
            curr_quat_scipy = np.array([curr_quat[1], curr_quat[2], curr_quat[3], curr_quat[0]])
            next_quat_scipy = np.array([next_quat[1], next_quat[2], next_quat[3], next_quat[0]])
            rotations = R.from_quat([curr_quat_scipy, next_quat_scipy])
            slerp = Slerp([0, 1], rotations)
            interp_rot = slerp(t_frac)
            quat_scipy = interp_rot.as_quat()
            quat = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])
            
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

    def get_precomputed_step(self, step):
        """Get precomputed joint positions for a specific step"""
        if self.precomputed_trajectory is None:
            raise ValueError("Trajectory not precomputed. Call precompute_joint_trajectory() first.")

        if step >= len(self.precomputed_trajectory):
            # Return final position if step exceeds trajectory length
            return self.precomputed_trajectory[-1]

        return self.precomputed_trajectory[step]

    def __call__(self, ts):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory(ts)

        # Use precomputed joint-space trajectory if available, otherwise fall back to Cartesian interpolation
        if self.precomputed_trajectory is not None and self.step_count < len(self.precomputed_trajectory):
            # Get precomputed joint positions and gripper values
            step_data = self.get_precomputed_step(self.step_count)

            # Return Cartesian poses from the joint-space interpolated trajectory
            left_xyz = step_data['left_xyz']
            right_xyz = step_data['right_xyz']
            left_quat = step_data['left_quat']
            right_quat = step_data['right_quat']
            left_gripper = step_data['left_gripper']
            right_gripper = step_data['right_gripper']
            left_joints = step_data['left_joints']
            right_joints = step_data['right_joints']
        else:
            # Fall back to Cartesian interpolation between waypoints
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
        # if self.inject_noise:
        #     scale = 0.01
        #     left_xyz = left_xyz + np.random.uniform(-scale, scale, left_xyz.shape)
        #     right_xyz = right_xyz + np.random.uniform(-scale, scale, right_xyz.shape)

        # action_left = np.concatenate([left_xyz, left_quat, [left_gripper]])
        # action_right = np.concatenate([right_xyz, right_quat, [right_gripper]])
        action_left = np.concatenate([left_joints, [left_gripper]])
        action_right = np.concatenate([right_joints, [right_gripper]])

        self.step_count += 1
        return np.concatenate([action_left, action_right])


class TransferCubeEETaskPiper():
    def __init__(self, random=None):
        self.max_reward = 4

    def initialize_robots(self, data):
        """Initialize piper robots to home position"""
        # Reset joint positions to home (all zeros for piper)
        # Piper has 16 joints total: 8 per arm (6 arm joints + 2 gripper joints)
        data.qpos[:16] = [0.0] * 16
        
        # Reset gripper control to open position (piper gripper values)
        # Piper gripper: joint7 [0, 0.035], joint8 [-0.035, 0]
        # Order: left joints (1-6), left gripper (7,8), right joints (1-6), right gripper (7,8)
        open_gripper_control = np.array([
            # Left arm joints (already set via qpos, but actuators need control values)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # Left joints 1-6
            PIPER_GRIPPER_POSITION_OPEN, -PIPER_GRIPPER_POSITION_OPEN,  # Left gripper joint7, joint8
            # Right arm joints
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # Right joints 1-6
            PIPER_GRIPPER_POSITION_OPEN, -PIPER_GRIPPER_POSITION_OPEN,  # Right gripper joint7, joint8
        ])
        np.copyto(data.ctrl, open_gripper_control)

    @staticmethod
    def get_qpos(data):
        qpos_raw = data.qpos.copy()
        left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[8:16]
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        # For piper, joint7 is the gripper position [0, 0.035], joint8 is the mirror [-0.035, 0]
        # Normalize joint7 only (index 6) for gripper position
        left_gripper_qpos = [PIPER_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[6])]
        right_gripper_qpos = [PIPER_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[6])]
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(data):
        qvel_raw = data.qvel.copy()
        left_qvel_raw = qvel_raw[:8]
        right_qvel_raw = qvel_raw[8:16]
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        # For piper, normalize joint7 velocity (index 6) for gripper velocity
        left_gripper_qvel = [PIPER_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[6])]
        right_gripper_qvel = [PIPER_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[6])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    @staticmethod
    def get_env_state(data):
        env_state = data.qpos.copy()[16:]
        return env_state

    def get_observation(self, data, renderer=None):
        # note: it is important to do .copy()
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(data)
        obs['qvel'] = self.get_qvel(data)
        obs['env_state'] = self.get_env_state(data)
        obs['images'] = dict()
        # obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        if renderer is not None:
            renderer.update_scene(data, camera="top")
            obs['images']['top'] = renderer.render()
        else:
            obs['images']['top'] = np.random.randint(0, 255, (480, 640, 3))

        return obs

    def get_reward(self, model, data):
        # return whether grippers are holding the box (piper-specific contact checking)
        all_contact_pairs = []
        for i_contact in range(data.ncon):
            id_geom_1 = data.contact[i_contact].geom1
            id_geom_2 = data.contact[i_contact].geom2
            name_geom_1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, id_geom_1)
            name_geom_2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, id_geom_2)
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        # For piper robots, check contact with either gripper finger (link7 or link8)
        touch_left_gripper = ("red_box", "piper_left/7_link7") in all_contact_pairs or \
                            ("red_box", "piper_left/8_link8") in all_contact_pairs
        touch_right_gripper = ("red_box", "piper_right/7_link7") in all_contact_pairs or \
                             ("red_box", "piper_right/8_link8") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        reward = 0
        if touch_left_gripper:
            reward = 1
        if touch_left_gripper and not touch_table: # lifted
            reward = 2
        if touch_right_gripper: # attempted transfer
            reward = 3
        if touch_right_gripper and not touch_table: # successful transfer
            reward = 4
        return reward



class PiperEnvironment():
    def __init__(self, task, model, data):
        self.model = model
        self.data = data
        self.task: TransferCubeEETaskPiper = task
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.renderer = mujoco.Renderer(model, height=480, width=640)

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        cube_position_x_lim = [0.15, 0.35]
        cube_position_y_lim = [-0.1, 0.1]
        cube_position_x = np.random.uniform(cube_position_x_lim[0], cube_position_x_lim[1])
        cube_position_y = np.random.uniform(cube_position_y_lim[0], cube_position_y_lim[1])
        cube_position = np.array([cube_position_x, cube_position_y, 0.05])
        box_start_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "red_box_joint")
        np.copyto(self.data.qpos[box_start_idx : box_start_idx + 3], cube_position)
        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()
        observation = self.task.get_observation(self.data)

        ts = SimpleNamespace(observation=observation, reward=0, action=np.zeros(14))
        return ts
    
    def step(self, action):
        """Step the environment with given action"""
        a_len = len(action) // 2
        action_left = action[:a_len]
        action_right = action[a_len:]
        env_action = np.array([
            # Left arm joints
            action_left[0], action_left[1], action_left[2], action_left[3], action_left[4], action_left[5],
            # Left gripper
            action_left[6], -action_left[6],  # joint7, joint8 (mirrored)
            # Right arm joints
            action_right[0], action_right[1], action_right[2], action_right[3], action_right[4], action_right[5],
            # Right gripper
            action_right[6], -action_right[6]  # joint7, joint8 (mirrored)
        ])
        # env_action = np.array([ 1, 0, 0, 0, 0, 0, 0, 0,
        #                         1, 0, 0, 0, 0, 0, 0, 0])
        self.data.ctrl[:] = env_action
        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()
        observation = self.task.get_observation(self.data, renderer=self.renderer)
        reward = self.task.get_reward(self.model, self.data)
        ts = SimpleNamespace(observation=observation, reward=reward, action=action)
        return ts

if __name__ == "__main__":

    model = mujoco.MjModel.from_xml_path(os.path.join(XML_DIR, f'test_piper.xml'))
    data = mujoco.MjData(model)

    task = TransferCubeEETaskPiper()
    env = PiperEnvironment(task, model, data)

    # for _ in range(10):
    #     ts = env.reset()
    #     input("Press Enter to continue...")

    # ts = env.reset()
    
    # ax = plt.subplot()
    # plt_img = ax.imshow(ts.observation['images'][render_cam_name])
    # plt.ion()

    episode_idx = 0
    while True:
        print("=" * 100)
        print(f"Episode {episode_idx}")
        print("=" * 100)
        ts = env.reset()
        policy = PickAndTransferPolicyPiper(inject_noise=False)
        episode = [ts]
        for _ in range(1200-1):
            action = policy(ts)
            ts = env.step(action)
            episode.append(ts)
            # plt_img.set_data(ts.observation['images'][render_cam_name])
            # # plt.pause(0.002)

        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/observations/images/top': [],
            '/action': [],
            '/reward': [],
            '/timestamp': [],
        }
        for time_step, ts in enumerate(episode):
            data_dict['/observations/qpos'].append(ts.observation['qpos'])
            data_dict['/observations/qvel'].append(ts.observation['qvel'])
            data_dict['/observations/images/top'].append(ts.observation['images']['top'])
            data_dict['/action'].append(ts.action)
            data_dict['/reward'].append(ts.reward)
            data_dict['/timestamp'].append(time_step * 0.02)

        print("*" * 50)
        print(f"Total reward: {ts.reward}")

        if ts.reward < 4:
            print(f"Episode {episode_idx} failed. Reward: {ts.reward}")
            continue

        max_timesteps = len(episode)
        dataset_dir = "/home/jeong/zeno/wholebody-teleop/act/dataset/sim_transfer_cube_piper"
        os.makedirs(dataset_dir, exist_ok=True)
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}')
        print(f"Saving episode {episode_idx} to {dataset_path}, {max_timesteps} timesteps...")
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            image.create_dataset('top', (max_timesteps, 480, 640, 3), dtype='uint8', chunks=(1, 480, 640, 3))
            qpos = obs.create_dataset('qpos', (max_timesteps, 14))
            qvel = obs.create_dataset('qvel', (max_timesteps, 14))
            action = root.create_dataset('action', (max_timesteps, 14))
            reward = root.create_dataset('reward', (max_timesteps,))
            timestamp = root.create_dataset('timestamp', (max_timesteps,))

            for name, array in data_dict.items():
                root[name][...] = array

        print("*" * 50)

        episode_idx += 1

        if episode_idx > 50:
            break