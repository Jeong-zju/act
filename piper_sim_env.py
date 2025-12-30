import numpy as np
from types import SimpleNamespace

import sys
import os

sys.path.append(os.path.dirname(__file__))

from piper_sim_task import TransferCubeTaskPiper, InsertionTaskPiper, MobileDualPiperTaskPiper
from constants import DT, PIPER_GRIPPER_POSITION_UNNORMALIZE_FN, PIPER_GRIPPER_POSITION_OPEN

import mujoco
import mujoco.viewer


class PiperCubeTransferEnvironment:

    def __init__(self, task, model, data):
        self.model = model
        self.data = data
        self.task: TransferCubeTaskPiper = task
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
        env_action = np.array(
            [
                # Left arm joints
                action_left[0],
                action_left[1],
                action_left[2],
                action_left[3],
                action_left[4],
                action_left[5],
                # Left gripper
                action_left[6],
                -action_left[6],  # joint7, joint8 (mirrored)
                # Right arm joints
                action_right[0],
                action_right[1],
                action_right[2],
                action_right[3],
                action_right[4],
                action_right[5],
                # Right gripper
                action_right[6],
                -action_right[6],  # joint7, joint8 (mirrored)
            ]
        )
        # env_action = np.array([ 1, 0, 0, 0, 0, 0, 0, 0,
        #                         1, 0, 0, 0, 0, 0, 0, 0])
        self.data.ctrl[:] = env_action
        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()
        observation = self.task.get_observation(self.data, renderer=self.renderer)
        reward = self.task.get_reward(self.model, self.data)
        ts = SimpleNamespace(observation=observation, reward=reward, action=action)
        return ts


class PiperInsertionEnvironment(PiperCubeTransferEnvironment):

    def __init__(self, task, model, data):
        self.model = model
        self.data = data
        self.task: InsertionTaskPiper = task
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.renderer = mujoco.Renderer(model, height=480, width=640)

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        # peg_position_x_lim = [0.15, 0.35]
        # peg_position_y_lim = [-0.01, 0.01]
        # peg_position_x = np.random.uniform(peg_position_x_lim[0], peg_position_x_lim[1])
        # peg_position_y = np.random.uniform(peg_position_y_lim[0], peg_position_y_lim[1])
        # peg_position = np.array([peg_position_x, peg_position_y, 0.05])
        # peg_start_idx = 16
        # np.copyto(self.data.qpos[peg_start_idx : peg_start_idx + 3], peg_position)

        # -0.56376582
        socket_position_x_lim = [0.15, 0.35]
        socket_position_y_lim = [-0.1, 0.1]
        socket_position_x = np.random.uniform(socket_position_x_lim[0], socket_position_x_lim[1])
        socket_position_y = np.random.uniform(socket_position_y_lim[0], socket_position_y_lim[1])
        socket_position = np.array([socket_position_x, socket_position_y, 0.1])
        socket_start_idx = 23
        np.copyto(self.data.qpos[socket_start_idx : socket_start_idx + 3], socket_position)

        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()
        observation = self.task.get_observation(self.data)

        ts = SimpleNamespace(observation=observation, reward=0, action=np.zeros(14))
        return ts


class MobileDualPiperEnvironment:

    def __init__(self, model, data, task):
        self.model = model
        self.data = data
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.renderer = mujoco.Renderer(model, height=480, width=640)
        self.task: MobileDualPiperTaskPiper = task

        # Get mocap body ID for base_mocap and find its mocap index
        base_mocap_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_mocap")
        self.mocap_id = self.model.body_mocapid[base_mocap_body_id]
        if self.mocap_id < 0:
            # If not found, assume it's the first mocap (index 0)
            self.mocap_id = 0

        # Initial z position from XML (0.08)
        self.base_z = 0.08

        # Track odometry state
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0

        # Stabilization parameters
        self.stabilization_steps = 1000  # Maximum steps to wait for stabilization
        self.velocity_threshold = 0.01  # Velocity threshold for considering stable
        self.viewer_update_interval = 10  # Update viewer every N steps

        # Pose interpolation parameters
        self.interpolation_steps = 50  # Number of steps for pose interpolation

    def reset(self):
        """Reset environment and set robot base to sampled initial pose"""
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        # Generate random initial position within U-shape map bounds
        # U-shape map valid area: x: -1.1 to 3.1, y: -3.1 to 1.9
        x_min, x_max = -0.6, 0.0
        y_min, y_max = 0.6, 1.6

        # Sample random initial position
        init_x = np.random.uniform(x_min, x_max)
        init_y = np.random.uniform(y_min, y_max)
        init_yaw = 0.0  # Keep yaw at 0 for simplicity

        print(f"Environment reset: sampled initial pose ({init_x:.3f}, {init_y:.3f}, {init_yaw:.3f})")

        # Get current pose after keyframe reset
        current_pos = self.data.mocap_pos[self.mocap_id].copy()
        current_quat = self.data.mocap_quat[self.mocap_id].copy()

        # Target pose
        target_pos = np.array([init_x, init_y, self.base_z])
        quat_w = np.cos(init_yaw / 2.0)
        quat_z = np.sin(init_yaw / 2.0)
        target_quat = np.array([quat_w, 0.0, 0.0, quat_z])

        # Linearly interpolate to target pose to avoid instability
        self._interpolate_to_pose(current_pos, current_quat, target_pos, target_quat)

        # Update odometry tracking to final position
        self.odom_x = init_x
        self.odom_y = init_y
        self.odom_yaw = init_yaw

        # Wait for robot to stabilize
        self._wait_for_stabilization()

        observation = self.task.get_observation(self.data, renderer=self.renderer)
        ts = SimpleNamespace(observation=observation, reward=0, action=np.zeros(17))
        return ts

    def _wait_for_stabilization(self):
        """Wait for the robot (base, arms, gripper) to stabilize after pose reset"""
        print("Waiting for robot stabilization...")

        for step in range(self.stabilization_steps):
            # Send control commands: arms to zero, grippers open
            self._send_reset_control_commands()

            # Step the simulation
            mujoco.mj_step(self.model, self.data)

            # Update viewer periodically for visualization
            if step % self.viewer_update_interval == 0:
                self.viewer.sync()

            # Check if velocities are below threshold (indicating stabilization)
            # Check maximum absolute velocity across all joints
            max_velocity = np.max(np.abs(self.data.qvel))

            if max_velocity < self.velocity_threshold:
                print(f"Robot stabilized after {step+1} steps (max velocity: {max_velocity:.6f})")
                break
        else:
            print(f"Robot did not fully stabilize within {self.stabilization_steps} steps "
                  f"(final max velocity: {max_velocity:.6f})")

        # Final viewer sync
        self.viewer.sync()

    def _interpolate_to_pose(self, start_pos, start_quat, target_pos, target_quat):
        """Linearly interpolate from start pose to target pose over several steps"""
        print("Interpolating to target pose...")

        for step in range(self.interpolation_steps):
            # Linear interpolation factor (0 to 1)
            t = (step + 1) / self.interpolation_steps

            # Interpolate position
            current_pos = start_pos + t * (target_pos - start_pos)

            # Interpolate quaternion (simplified - for small rotations around z-axis)
            # For more complex rotations, would need proper quaternion interpolation
            current_quat = start_quat + t * (target_quat - start_quat)
            # Normalize quaternion
            current_quat = current_quat / np.linalg.norm(current_quat)

            # Set mocap to interpolated pose
            np.copyto(self.data.mocap_pos[self.mocap_id], current_pos)
            np.copyto(self.data.mocap_quat[self.mocap_id], current_quat)

            # Send control commands: arms to zero, grippers open
            self._send_reset_control_commands()

            # Step simulation
            mujoco.mj_step(self.model, self.data)

            # Update viewer occasionally
            if step % 10 == 0:
                self.viewer.sync()

        # Final viewer sync
        self.viewer.sync()
        print("Pose interpolation complete")

    def _send_reset_control_commands(self):
        """Send control commands during reset: arms to zero, grippers open"""
        reset_ctrl = np.array([
            # Left arm joints (6 joints) - all zero for home position
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            # Left gripper - normalized open position (1.0)
            PIPER_GRIPPER_POSITION_OPEN,
            -PIPER_GRIPPER_POSITION_OPEN,
            # Right arm joints (6 joints) - all zero for home position
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            # Right gripper - normalized open position (1.0)
            PIPER_GRIPPER_POSITION_OPEN,
            -PIPER_GRIPPER_POSITION_OPEN,
        ])

        # Apply control
        self.data.ctrl[:] = reset_ctrl

    def step(self, action):
        """Step the environment with given action

        Args:
            action: [vx, vy, omega, left_arm_joints (6), left_gripper (1), right_arm_joints (6), right_gripper (1)]
                - vx: forward velocity (m/s)
                - vy: lateral velocity (m/s)
                - omega: angular velocity (rad/s)
                - left_arm_joints: 6 joint positions for left arm
                - left_gripper: normalized gripper position (0: close, 1: open)
                - right_arm_joints: 6 joint positions for right arm
                - right_gripper: normalized gripper position (0: close, 1: open)
        """
        # Parse action components
        vx, vy, omega = action[0], action[1], action[2]

        # Left arm: joints 1-6 and gripper
        left_arm_joints = action[3:9]  # 6 joint positions
        left_gripper_normalized = action[9]  # normalized gripper (0-1)

        # Right arm: joints 1-6 and gripper
        right_arm_joints = action[10:16]  # 6 joint positions
        right_gripper_normalized = action[16]  # normalized gripper (0-1)

        # Unnormalize gripper positions (from normalized 0-1 to actual range 0-0.035)
        left_gripper_pos = PIPER_GRIPPER_POSITION_UNNORMALIZE_FN(left_gripper_normalized)
        right_gripper_pos = PIPER_GRIPPER_POSITION_UNNORMALIZE_FN(right_gripper_normalized)

        # print(f"left_gripper_pos: {left_gripper_pos}, left_gripper_normalized: {left_gripper_normalized}, right_gripper_pos: {right_gripper_pos}, right_gripper_normalized: {right_gripper_normalized}")

        # Transform velocities from robot frame to world frame
        # vx is forward (robot x), vy is left (robot y)
        vx_world = vx * np.cos(self.odom_yaw) - vy * np.sin(self.odom_yaw)
        vy_world = vx * np.sin(self.odom_yaw) + vy * np.cos(self.odom_yaw)

        # Integrate to update odometry
        self.odom_x += vx_world * DT
        self.odom_y += vy_world * DT
        self.odom_yaw += omega * DT
        self.odom_yaw = (self.odom_yaw + np.pi) % (2 * np.pi) - np.pi

        # Convert yaw to quaternion (rotation around z-axis)
        # MuJoCo quaternion format: [w, x, y, z]
        quat_w = np.cos(self.odom_yaw / 2.0)
        quat_z = np.sin(self.odom_yaw / 2.0)
        quat = np.array([quat_w, 0.0, 0.0, quat_z])

        # Update mocap position and orientation
        np.copyto(self.data.mocap_pos[self.mocap_id], [self.odom_x, self.odom_y, self.base_z])
        np.copyto(self.data.mocap_quat[self.mocap_id], quat)

        # Build control array for arms
        # Control order: left arm joints 1-6, left gripper joints 7-8, right arm joints 1-6, right gripper joints 7-8
        env_action = np.array(
            [
                # Left arm joints 1-6
                left_arm_joints[0],
                left_arm_joints[1],
                left_arm_joints[2],
                left_arm_joints[3],
                left_arm_joints[4],
                left_arm_joints[5],
                # Left gripper (joint7, joint8 are mirrored)
                left_gripper_pos,
                -left_gripper_pos,
                # Right arm joints 1-6
                right_arm_joints[0],
                right_arm_joints[1],
                right_arm_joints[2],
                right_arm_joints[3],
                right_arm_joints[4],
                right_arm_joints[5],
                # Right gripper (joint7, joint8 are mirrored)
                right_gripper_pos,
                -right_gripper_pos,
            ]
        )

        # Apply control actions
        self.data.ctrl[:] = env_action

        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()

        observation = self.task.get_observation(self.data, renderer=self.renderer)
        reward = 0
        ts = SimpleNamespace(observation=observation, reward=0, action=action)
        return ts
