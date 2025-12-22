import time
import numpy as np
from scipy.spatial.transform import Rotation as R

# Import constants from sim_env.py
import sys
import os
sys.path.append(os.path.dirname(__file__))
from constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN

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

    def precompute_joint_trajectory(self, left_robot, right_robot, ik_solver, left_qinit=None, right_qinit=None):
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
                joint_pos = ik_solver.ik(left_pose_matrix, qinit=left_qinit)
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
                joint_pos = ik_solver.ik(right_pose_matrix, qinit=right_qinit)
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
            meet_xyz = np.array([0, -0.28188, 0.3])  # Center meeting point
            meet_left_xyz = np.array([0, -0.28188, 0.3]) + np.array([0.0, 0.125, 0.0])  # Left meeting point
            print(f"meet_left_xyz: {meet_left_xyz}")
            meet_right_xyz = np.array([0, 0.28188, 0.3]) + np.array([0.0, -0.125, 0.0])  # Right meeting point
            print(f"meet_right_xyz: {meet_right_xyz}")

            # Define orientations
            meeting_left_euler = (np.pi/2, 0, 0)  # meeting_pose_left_delta
            meet_left_quat = R.from_euler('xyz', meeting_left_euler).as_quat()
            meeting_right_euler = (0, np.pi/2, np.pi/2)  # meeting_pose_right_delta
            meet_right_quat = R.from_euler('xyz', meeting_right_euler).as_quat()

        # Define orientations for cube reaching
        cube_reach_euler = (0, np.pi, 0)  # For left arm reaching to cube
        cube_reach_quat = R.from_euler('xyz', cube_reach_euler).as_quat()

        zero_quat = np.array([0, 0, 0, 1])  # Identity quaternion for zero position

        gripper_close = 0.0
        gripper_open = 0.035

        # Left arm waypoints (7 steps as requested)
        self.left_trajectory = [
            {"t": 0, "xyz": np.array([0, 0, 0]), "quat": zero_quat, "gripper": 0.035},
            {"t": 200, "xyz": box_xyz + np.array([0, 0, 0.195]), "quat": cube_reach_quat, "gripper": 0.035},
            {"t": 250, "xyz": box_xyz + np.array([0, 0, 0.095]), "quat": cube_reach_quat, "gripper": 0.035},
            {"t": 300, "xyz": box_xyz + np.array([0, 0, 0.095]), "quat": cube_reach_quat, "gripper": gripper_close},
            {"t": 600, "xyz": meet_left_xyz + np.array([0, 0.05, 0]), "quat": meet_left_quat, "gripper": gripper_close},
            {"t": 700, "xyz": meet_left_xyz, "quat": meet_left_quat, "gripper": gripper_close},
            {"t": 800, "xyz": meet_left_xyz, "quat": meet_left_quat, "gripper": gripper_close},
            {"t": 900, "xyz": meet_left_xyz, "quat": meet_left_quat, "gripper": 0.035},
            {"t": 1200, "xyz": np.array([0, 0, 0]), "quat": zero_quat, "gripper": 0.035},
        ]

        # Right arm waypoints
        self.right_trajectory = [
            {"t": 0, "xyz": np.array([0, 0, 0]), "quat": zero_quat, "gripper": 0.035},
            {"t": 600, "xyz": meet_right_xyz + np.array([0, -0.05, 0]), "quat": meet_right_quat, "gripper": 0.035},
            {"t": 700, "xyz": meet_right_xyz, "quat": meet_right_quat, "gripper": 0.035},
            {"t": 800, "xyz": meet_right_xyz, "quat": meet_right_quat, "gripper": gripper_close},
            {"t": 900, "xyz": meet_right_xyz, "quat": meet_right_quat, "gripper": gripper_close},
            {"t": 1200, "xyz": np.array([0, 0, 0]), "quat": zero_quat, "gripper": gripper_close}
        ]


class MuJoCoController:
    """Controller class to manage MuJoCo simulation with the cube transfer policy"""

    def __init__(self, model, data, policy=None, robot_model_path=None):
        self.model = model
        self.data = data
        self.policy = policy
        self.step_count = 0

        # Find cube joint index
        self.cube_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "red_box_joint")

        # Initialize policy if not provided
        if self.policy is None:
            # Policy will be created later after cube position is randomized
            # For now, create a dummy policy that will be replaced
            self.policy = None

        # Set up IK solvers if available
        if TRACIK_AVAILABLE:
            robot_model_path = robot_model_path or "/home/jeong/zeno/wholebody-teleop/act/assets/piper_description.urdf"
            # Create separate IK solvers for each arm since they're in the same MuJoCo model
            # but have different kinematic chains
            self.ik_solver = TracIKSolver(robot_model_path, "base_link", "gripper_base")

            # Get initial joint positions (home position)
            self.left_qinit = [0.0] * 6  # Start from home position
            self.right_qinit = [0.0] * 6  # Start from home position

            # Precompute the joint trajectory using IK
            # print("Precomputing joint trajectory using IK...")
            # self.policy.precompute_joint_trajectory(
            #     self, self,  # Mock robot objects (we'll override get_joint_positions)
            #     self.ik_solver,
            #     left_qinit=self.left_qinit,
            #     right_qinit=self.right_qinit
            # )
            # print(f"Precomputed trajectory with {len(self.policy.precomputed_trajectory)} steps")
        else:
            print("TracIKSolver not available - using mock trajectory")
            # Create a mock trajectory for testing
            self.policy.precomputed_trajectory = []
            for i in range(1200):  # Same length as original
                self.policy.precomputed_trajectory.append({
                    'left_joints': [0.0] * 6,
                    'right_joints': [0.0] * 6,
                    'left_gripper': 0.035,
                    'right_gripper': 0.035,
                    'left_xyz': None,
                    'right_xyz': None
                })

    def get_joint_positions(self):
        """Mock method to provide joint positions for IK solver"""
        return [0.0] * 6  # Return home position

    def randomize_cube_position(self):
        """Randomly set the cube position in the MuJoCo simulation"""
        # Random position within a reasonable workspace
        # Keep it near the center but with some variation
        x = np.random.uniform(0.1, 0.3)  # X position: 0.1 to 0.3
        y = np.random.uniform(-0.2, 0.2)  # Y position: -0.2 to 0.2
        z = 0.05  # Keep Z fixed at table height

        # Keep orientation as identity (no rotation)
        orientation = np.array([1, 0, 0, 0])  # Identity quaternion

        # Set cube position and orientation in qpos
        self.data.qpos[self.cube_joint_id:self.cube_joint_id+3] = [x, y, z]
        self.data.qpos[self.cube_joint_id+3:self.cube_joint_id+7] = orientation

        print(f"Randomized cube position to: [{x:.3f}, {y:.3f}, {z:.3f}]")

    def recreate_policy_with_current_cube_position(self):
        """Recreate the policy using the current cube position in MuJoCo"""
        # Get current cube pose from MuJoCo data
        cube_position_world = self.data.qpos[self.cube_joint_id:self.cube_joint_id+3]
        cube_orientation_world = self.data.qpos[self.cube_joint_id+3:self.cube_joint_id+7]

        # Transform cube pose to each arm's local coordinate frame
        left_base_pos = np.array([0, 0, 0])
        left_base_ori = np.array([0, 0, 0, 1])  # Identity quaternion

        right_base_pos = np.array([-0.00860732, -0.56376582, 0.00961532])
        right_base_ori = np.array([0.99967745, -0.01113996, 0.0132907, 0.01855407])

        cube_pose_left_local = self._transform_pose_to_local(cube_position_world, cube_orientation_world,
                                                           left_base_pos, left_base_ori)
        cube_pose_right_local = self._transform_pose_to_local(cube_position_world, cube_orientation_world,
                                                            right_base_pos, right_base_ori)

        self.policy = CubeTransferPolicy(cube_pose_left_local, cube_pose_right_local)

        # Re-precompute trajectory if IK is available
        if TRACIK_AVAILABLE and hasattr(self, 'ik_solver'):
            print("Precomputing joint trajectory for first episode...")
            self.policy.precompute_joint_trajectory(
                self, self,
                self.ik_solver,
                left_qinit=self.left_qinit,
                right_qinit=self.right_qinit
            )

    def _transform_pose_to_local(self, world_pos, world_ori, base_pos, base_ori):
        """Transform a pose from world coordinates to local robot base coordinates"""
        # Convert quaternions to rotation matrices
        world_rot = R.from_quat(world_ori)
        base_rot = R.from_quat(base_ori)

        # Transform: local_pose = inverse(base_pose) * world_pose
        # First, transform position: local_pos = base_rot^T * (world_pos - base_pos)
        local_pos = base_rot.inv().apply(world_pos - base_pos)

        # Then, transform orientation: local_ori = base_rot^T * world_rot
        local_rot = base_rot.inv() * world_rot
        local_ori = local_rot.as_quat()

        return (local_pos, local_ori)

    def step_policy(self):
        """Execute one step of the policy and apply to MuJoCo"""
        # Check if policy exists and has a precomputed trajectory
        if self.policy is None or self.policy.precomputed_trajectory is None or self.step_count >= len(self.policy.precomputed_trajectory):
            return False

        # Get precomputed joint positions
        step_data = self.policy.get_precomputed_step(self.step_count)

        # Extract joint positions and gripper values
        left_joints = step_data['left_joints']
        right_joints = step_data['right_joints']
        left_gripper_pos = step_data['left_gripper']
        right_gripper_pos = step_data['right_gripper']

        # Normalize gripper positions for MuJoCo control
        # MuJoCo gripper range based on actuator limits: joint7 [0, 0.035], joint8 [-0.035, 0]
        left_gripper_normalized = (left_gripper_pos - 0.0) / (0.035 - 0.0)  # Normalize to [0, 1]
        right_gripper_normalized = (right_gripper_pos - 0.0) / (0.035 - 0.0)

        # Clamp to valid range
        left_gripper_normalized = np.clip(left_gripper_normalized, 0.0, 1.0)
        right_gripper_normalized = np.clip(right_gripper_normalized, 0.0, 1.0)

        # Convert to actual gripper positions for MuJoCo
        left_gripper_action = left_gripper_normalized * 0.035  # joint7: [0, 0.035]
        right_gripper_action = right_gripper_normalized * 0.035  # joint7: [0, 0.035]

        # Create full action array for MuJoCo (matching the actuator order in test_piper.xml)
        # Order: left joints (1-6), left gripper (7,8), right joints (1-6), right gripper (7,8)
        env_action = np.array([
            # Left arm joints
            left_joints[0], left_joints[1], left_joints[2], left_joints[3], left_joints[4], left_joints[5],
            # Left gripper
            left_gripper_action, -left_gripper_action,  # joint7, joint8 (mirrored)
            # Right arm joints
            right_joints[0], right_joints[1], right_joints[2], right_joints[3], right_joints[4], right_joints[5],
            # Right gripper
            right_gripper_action, -right_gripper_action  # joint7, joint8 (mirrored)
        ])

        # Apply action to MuJoCo
        self.data.ctrl[:] = env_action

        # Debug output
        if self.step_count % 200 == 0:  # Print every 200 steps
            print(f"Step {self.step_count}: Applied control - Left joints: {left_joints[:3]}..., Gripper: {left_gripper_pos:.3f}")

        self.step_count += 1
        return True

    def check_cube_grasped(self):
        """Check if the cube is currently grasped by the left gripper (successful transfer)"""
        # Get all contact pairs
        all_contact_pairs = []
        for i_contact in range(self.data.ncon):
            id_geom_1 = self.data.contact[i_contact].geom1
            id_geom_2 = self.data.contact[i_contact].geom2
            name_geom_1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, id_geom_1)
            name_geom_2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, id_geom_2)
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        # Check contacts based on sim_env.py logic
        # touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        # touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        # For piper robots, check contact with either gripper finger
        touch_left_gripper = ("red_box", "piper_left/7_link7") in all_contact_pairs or ("red_box", "piper_left/8_link8") in all_contact_pairs
        touch_right_gripper = ("red_box", "piper_right/7_link7") in all_contact_pairs or ("red_box", "piper_right/8_link8") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        # Return True if cube is grasped by left gripper and not touching table (successful transfer)
        return touch_right_gripper and not touch_table


if __name__ == "__main__":
    if not MUJOCO_AVAILABLE:
        print("MuJoCo not available - testing policy only")
        # Test policy without MuJoCo
        cube_pose = ([0.2, 0, 0.05], [0, 0, 0, 1])
        policy = CubeTransferPolicy(cube_pose, cube_pose)
        policy.generate_trajectory()
        print(f"Generated trajectory with {len(policy.left_trajectory)} waypoints")
        print("Policy test completed successfully!")
    else:
        # Initialize MuJoCo model and data
        model = mujoco.MjModel.from_xml_path("assets/test_piper.xml")
        data = mujoco.MjData(model)

        print(f"Model loaded with {model.nq} DOF and {model.nu} actuators")
        print(f"Joint names: {[model.joint(i).name for i in range(model.njnt)]}")
        print(f"Actuator names: {[model.actuator(i).name for i in range(model.nu)]}")

        # Initialize controller with policy
        controller = MuJoCoController(model, data)

        # Test control array size
        print(f"Control array size: {len(data.ctrl)} (expected: {model.nu})")
        if len(data.ctrl) != model.nu:
            print("ERROR: Control array size mismatch!")
            exit(1)

        # Reset to initial keyframe pose
        mujoco.mj_resetDataKeyframe(model, data, 0)
        print("Reset to initial keyframe pose")

        # Randomize cube position and recreate policy for first episode
        controller.randomize_cube_position()
        controller.recreate_policy_with_current_cube_position()

        # Run simulation with policy control
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                step_start = time.time()

                # Apply policy control
                if not controller.step_policy():
                    # Policy finished, check if cube was grasped
                    cube_grasped = controller.check_cube_grasped()

                    if not cube_grasped:
                        print("Episode finished. Cube not grasped")
                        break
                    
                    # Policy finished, reset
                    controller.step_count = 0
                    mujoco.mj_resetDataKeyframe(model, data, 0)
                    # Randomize cube position for next episode
                    controller.randomize_cube_position()
                    # Recreate policy with new cube position
                    controller.recreate_policy_with_current_cube_position()
                    print("Reset to initial pose with randomized cube position")

                # Step physics
                mujoco.mj_step(model, data)

                # Optional: modify visualization options
                # with viewer.lock():
                #     viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1

                # Sync and render
                viewer.sync()

                # Sleep to maintain 60 FPS (fixed time step)
                target_frame_time = 1.0 / 60.0  # 60 FPS
                elapsed_time = time.time() - step_start
                sleep_time = target_frame_time - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
