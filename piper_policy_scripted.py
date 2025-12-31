import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import heapq

# Import constants from sim_env.py
import sys
import os
sys.path.append(os.path.dirname(__file__))

from constants import DT

try:
    from tracikpy import TracIKSolver
    TRACIK_AVAILABLE = True
except ImportError:
    TRACIK_AVAILABLE = False
    print("TracIKSolver not available - IK will not work")


class AStarPlanner:
    """A* 路径规划算法实现"""

    def __init__(self, grid_size=0.1, grid_width=50, grid_height=50, obstacles=None):
        """
        初始化A*规划器

        Args:
            grid_size: 网格分辨率 (米)
            grid_width: 网格宽度 (网格单元)
            grid_height: 网格高度 (网格单元)
            obstacles: 障碍物列表，每个障碍物为 (x, y, radius) 元组
        """
        self.grid_size = grid_size
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.obstacles = obstacles or []

        # 创建网格中心在原点的坐标系
        self.origin_x = -grid_width * grid_size / 2
        self.origin_y = -grid_height * grid_size / 2

    def world_to_grid(self, world_x, world_y):
        """将世界坐标转换为网格坐标"""
        grid_x = int((world_x - self.origin_x) / self.grid_size)
        grid_y = int((world_y - self.origin_y) / self.grid_size)
        return grid_x, grid_y

    def grid_to_world(self, grid_x, grid_y):
        """将网格坐标转换为世界坐标"""
        world_x = grid_x * self.grid_size + self.origin_x + self.grid_size / 2
        world_y = grid_y * self.grid_size + self.origin_y + self.grid_size / 2
        return world_x, world_y

    def is_valid_position(self, grid_x, grid_y):
        """检查网格位置是否有效且无碰撞"""
        # 检查边界
        if not (0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height):
            return False

        # 获取机器人中心位置（网格中心点）
        robot_center_x, robot_center_y = self.grid_to_world(grid_x, grid_y)

        # 机器人碰撞球半径 (从mobile_piper.xml中的sphere size="0.5")
        robot_radius = 0.5

        # 检查与所有障碍物的碰撞
        for obs_x, obs_y, obs_radius in self.obstacles:
            # 计算机器人中心到障碍物中心的距离
            distance = np.sqrt((robot_center_x - obs_x)**2 + (robot_center_y - obs_y)**2)
            # 如果距离小于机器人半径+障碍物半径，则发生碰撞
            if distance < (robot_radius + obs_radius):
                return False

        return True

    def get_neighbors(self, grid_x, grid_y):
        """获取有效邻居节点 (8个方向)"""
        neighbors = []
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

        for dx, dy in directions:
            nx, ny = grid_x + dx, grid_y + dy
            if self.is_valid_position(nx, ny):
                neighbors.append((nx, ny))

        return neighbors

    def heuristic(self, grid_x1, grid_y1, grid_x2, grid_y2):
        """启发函数：使用欧几里得距离"""
        return np.sqrt((grid_x1 - grid_x2)**2 + (grid_y1 - grid_y2)**2)

    def plan_path(self, start_world, goal_world):
        """
        规划从起点到终点的路径

        Args:
            start_world: 起点世界坐标 (x, y)
            goal_world: 终点世界坐标 (x, y)

        Returns:
            path_world: 路径点列表 [(x1,y1), (x2,y2), ...]
        """
        start_grid = self.world_to_grid(*start_world)
        goal_grid = self.world_to_grid(*goal_world)

        # 检查起点和终点是否有效
        if not self.is_valid_position(*start_grid):
            print(f"起点无效: {start_world}")
            return [start_world, goal_world]  # 返回直线路径

        if not self.is_valid_position(*goal_grid):
            print(f"终点无效: {goal_world}")
            return [start_world, goal_world]  # 返回直线路径

        # A* 算法实现
        frontier = []
        heapq.heappush(frontier, (0, start_grid))

        came_from = {start_grid: None}
        cost_so_far = {start_grid: 0}

        while frontier:
            current_cost, current = heapq.heappop(frontier)

            if current == goal_grid:
                break

            for neighbor in self.get_neighbors(*current):
                new_cost = cost_so_far[current] + self.heuristic(*current, *neighbor)

                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.heuristic(*neighbor, *goal_grid)
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current

        # 重建路径
        if goal_grid not in came_from:
            print("无法找到路径，返回直线路径")
            return [start_world, goal_world]

        path_grid = []
        current = goal_grid
        while current is not None:
            path_grid.append(current)
            current = came_from[current]
        path_grid.reverse()

        # 转换为世界坐标
        path_world = [self.grid_to_world(*grid_pos) for grid_pos in path_grid]

        return path_world

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

class InsertionPolicyPiper(PickAndTransferPolicyPiper):

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

        peg_info = np.array(ts_first.observation['env_state']["peg"])
        peg_xyz = peg_info[:3]
        peg_quat = peg_info[3:]

        socket_info = np.array(ts_first.observation['env_state']["socket"])
        socket_xyz = socket_info[:3]
        socket_quat = socket_info[3:]
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
        gripper_open = 0.02

        # Left arm waypoints matching piper_mj_record.py exactly
        self.left_trajectory = [
            {"t": 0, "joint": [0, 0, 0, 0, 0, 0], "gripper": gripper_open},
            {"t": 200, "xyz": socket_xyz + np.array([0.015, 0.0001, 0.125]), "quat": cube_reach_quat, "gripper": gripper_open},
            {"t": 250, "xyz": socket_xyz + np.array([0.015, 0.0001, 0.1125]), "quat": cube_reach_quat, "gripper": gripper_open},
            {"t": 300, "xyz": socket_xyz + np.array([0.015, 0.0001, 0.1]), "quat": cube_reach_quat, "gripper": gripper_open},
            {"t": 350, "xyz": socket_xyz + np.array([0.015, 0.0001, 0.0875]), "quat": cube_reach_quat, "gripper": gripper_open},
            {"t": 400, "xyz": socket_xyz + np.array([0.015, 0.0001, 0.075-0.005]), "quat": cube_reach_quat, "gripper": gripper_open},
            {"t": 500, "xyz": socket_xyz + np.array([0.015, 0.0001, 0.075-0.005]), "quat": cube_reach_quat, "gripper": gripper_open},
        ]

        # Right arm waypoints matching piper_mj_record.py exactly
        # self.right_trajectory = [
        #     {"t": 0, "joint": [0, 0, 0, 0, 0, 0], "gripper": gripper_open},
        #     {"t": 600, "xyz": right2rightmeet_xyz + np.array([0, -0.05, 0]), "quat": right2rightmeet_quat, "gripper": gripper_open},
        #     {"t": 700, "xyz": right2rightmeet_xyz, "quat": right2rightmeet_quat, "gripper": gripper_open},
        #     {"t": 800, "xyz": right2rightmeet_xyz, "quat": right2rightmeet_quat, "gripper": gripper_close},
        #     {"t": 900, "xyz": right2rightmeet_xyz, "quat": right2rightmeet_quat, "gripper": gripper_close},
        #     {"t": 1200, "joint": [0, 0, 0, 0, 0, 0], "gripper": gripper_close}
        # ]

        self.right_trajectory = [
            {"t": 0, "joint": [0, 0, 0, 0, 0, 0], "gripper": gripper_open},
            {"t": 500, "joint": [0, 0, 0, 0, 0, 0], "gripper": gripper_open}
        ]

        self.precompute_joint_trajectory(ts_first)


class MobileDualPiperPickAndTransferPolicyPiper(PickAndTransferPolicyPiper):
    """Policy for mobile dual piper robot: base moves to pick column, left arm grasps cube,
    base moves to place column, left arm places cube on top"""

    def __init__(self, inject_noise=False, robot_model_path=None):
        super().__init__(inject_noise, robot_model_path)
        self.base_trajectory = None  # Base velocity commands [vx, vy, omega]
        self.base_positions = None  # Base positions for reference [x, y, yaw]

        # 初始化A*规划器
        # U形地图障碍物配置 (同步自 mobile_piper.xml)
        obstacles = [
            # Left vertical wall (behind_wall_*, x = -1.5)
            (-1.5, -2.5, 0.3),
            (-1.5, -2.0, 0.3),
            (-1.5, -1.5, 0.3),
            (-1.5, -1.0, 0.3),
            (-1.5, -0.5, 0.3),
            (-1.5, 0.0, 0.3),
            (-1.5, 0.5, 0.3),
            (-1.5, 1.0, 0.3),
            (-1.5, 1.5, 0.3),
            (-1.5, 2.0, 0.3),
            (-1.5, 2.5, 0.3),

            # Bottom horizontal wall (right_wall_*, y = -2.5)
            (-1.5, -2.5, 0.3),
            (-1.0, -2.5, 0.3),
            (-0.5, -2.5, 0.3),
            (0.0, -2.5, 0.3),
            (0.5, -2.5, 0.3),
            (1.0, -2.5, 0.3),
            (1.5, -2.5, 0.3),

            # Right vertical wall (front_wall_*, x = 1.5)
            (1.5, -2.5, 0.3),
            (1.5, -2.0, 0.3),
            (1.5, -1.5, 0.3),
            (1.5, -1.0, 0.3),
            (1.5, -0.5, 0.3),
            (1.5, 0.0, 0.3),
            (1.5, 0.5, 0.3),
            (1.5, 1.0, 0.3),
            (1.5, 1.5, 0.3),
            (1.5, 2.0, 0.3),
            (1.5, 2.5, 0.3),

            # Top horizontal wall (left_wall_*, y = 2.5)
            (-1.5, 2.5, 0.3),
            (-1.0, 2.5, 0.3),
            (-0.5, 2.5, 0.3),
            (0.0, 2.5, 0.3),
            (0.5, 2.5, 0.3),
            (1.0, 2.5, 0.3),
            (1.5, 2.5, 0.3),

            # Middle vertical wall (middle_wall_*)
            (1.0, 0.0, 0.1),
            (0.5, 0.0, 0.1),
            (0.0, 0.0, 0.1),
        ]

        self.astar_planner = AStarPlanner(
            grid_size=0.01,  # 1cm分辨率
            grid_width=400,   # 4m x 4m 地图
            grid_height=400,
            obstacles=obstacles
        )
        
    def generate_trajectory(self, ts_first):
        """Generate trajectory for mobile base and left arm using A* path planning"""
        # Read cube position and quaternion from env_state
        env_state = np.array(ts_first.observation['env_state'])
        cube_xyz = env_state[:3]
        cube_quat = env_state[3:7]  # [w, x, y, z] format
        world2cube_mat = self.pose_to_matrix(cube_xyz, cube_quat)
        place_target_xyz = np.array([1.0, -1.0, 1.02])
        place_target_quat = cube_quat
        place_world2place_mat = self.pose_to_matrix(place_target_xyz, place_target_quat)

        # 更新障碍物信息
        # self.update_obstacles_from_env(ts_first.observation.get('env_state', {}))

        left_mount_offset = np.array([0.26, 0.28, 0.57])  # Relative to mobile_base
        base2left_mat = self.pose_to_matrix(left_mount_offset, np.array([1.0, 0.0, 0.0, 0.0]))

        mobile_base_z_offset = 0.3  # mobile_base body z position in world

        # Get actual initial position from observation (set by environment reset)
        actual_init_x, actual_init_y = ts_first.observation['qpos'][:2]
        base_init = np.array([actual_init_x, actual_init_y, 0.0]) # x, y, yaw from environment
        print(f"Policy using environment initial pose: ({actual_init_x:.3f}, {actual_init_y:.3f})")

        base_pick = np.array([0.5-0.26, 1.0-0.28, 0.0]) # x, y, yaw
        base_place = np.array([0.5-0.26, -1.0-0.28, 0.0]) # x, y, yaw

        # 使用A*规划从初始位置到拾取位置的路径
        path_to_pick = self.astar_planner.plan_path(
            (base_init[0], base_init[1]),
            (base_pick[0], base_pick[1])
        )

        # 使用A*规划从拾取位置到放置位置的路径
        path_to_place = self.astar_planner.plan_path(
            (base_pick[0], base_pick[1]),
            (base_place[0], base_place[1])
        )

        # 打印规划的路径
        print(f"A*规划路径到拾取位置: {len(path_to_pick)} 个路径点")
        print(f"A*规划路径到放置位置: {len(path_to_place)} 个路径点")

        # 使用A*路径的端点作为参考位姿
        # 拾取位置使用path_to_pick的最后一个点
        pick_ref_x, pick_ref_y = path_to_pick[-1] if path_to_pick else (base_pick[0], base_pick[1])
        pick_world2base_mat = self.pose_to_matrix(np.array([pick_ref_x, pick_ref_y, mobile_base_z_offset]), np.array([1.0, 0.0, 0.0, 0.0]))
        pick_left2cube_mat = np.linalg.inv(pick_world2base_mat @ base2left_mat) @ world2cube_mat

        pick_left_xyz = np.array([pick_left2cube_mat[0, 3], pick_left2cube_mat[1, 3], pick_left2cube_mat[2, 3]])
        pick_left_quat_scipy = R.from_matrix(pick_left2cube_mat[:3, :3]).as_quat()
        pick_left_quat = np.array([pick_left_quat_scipy[3], pick_left_quat_scipy[0], pick_left_quat_scipy[1], pick_left_quat_scipy[2]])  # Convert to [w, x, y, z]

        # 放置位置使用path_to_place的第一个点（起始点），根据您的要求
        place_ref_x, place_ref_y = path_to_place[-1] if path_to_place else (base_place[0], base_place[1])
        place_world2base_mat = self.pose_to_matrix(np.array([place_ref_x, place_ref_y, mobile_base_z_offset]), np.array([1.0, 0.0, 0.0, 0.0]))
        place_left2cube_mat = np.linalg.inv(place_world2base_mat @ base2left_mat) @ place_world2place_mat

        place_left_xyz = np.array([place_left2cube_mat[0, 3], place_left2cube_mat[1, 3], place_left2cube_mat[2, 3]])
        place_left_quat_scipy = R.from_matrix(place_left2cube_mat[:3, :3]).as_quat()
        place_left_quat = np.array([place_left_quat_scipy[3], place_left_quat_scipy[0], place_left_quat_scipy[1], place_left_quat_scipy[2]])  # Convert to [w, x, y, z]

        print(f"拾取参考位置: ({pick_ref_x:.3f}, {pick_ref_y:.3f})")
        print(f"放置参考位置: ({place_ref_x:.3f}, {place_ref_y:.3f})")

        # Gripper values
        gripper_close = 0.005
        gripper_open = 0.035

        self.left_trajectory = [
            {"t": 0, "joint": [0, 0, 0, 0, 0, 0], "gripper": gripper_open},
            {"t": 200, "joint": [0, 0, 0, 0, 0, 0], "gripper": gripper_open},
            {"t": 400, "xyz": pick_left_xyz + np.array([-0.11, 0, 0.1]), "quat": pick_left_quat, "gripper": gripper_open},  # Approach cube (in arm base frame)
            {"t": 500, "xyz": pick_left_xyz + np.array([-0.11, 0, 0]), "quat": pick_left_quat, "gripper": gripper_open},  # Reach cube (in arm base frame)
            # {"t": 550, "xyz": pick_left_xyz + np.array([-0.125, 0, 0]), "quat": pick_left_quat, "gripper": gripper_open},  # Grasp (in arm base frame)
            {"t": 600, "xyz": pick_left_xyz + np.array([-0.11, 0, 0]), "quat": pick_left_quat, "gripper": gripper_close},  # Grasp (in arm base frame)
            {"t": 1000, "joint": [0, 0, 0, 0, 0, 0], "gripper": gripper_close},  # Return to home
            {"t": 1300, "xyz": place_left_xyz + np.array([-0.11, 0, 0]), "quat": place_left_quat, "gripper": gripper_close},  # Reach cube (in arm base frame)
            {"t": 1400, "xyz": place_left_xyz + np.array([-0.11, 0, 0]), "quat": place_left_quat, "gripper": gripper_open},  # Reach cube (in arm base frame)
            {"t": 1500, "joint": [0, 0, 0, 0, 0, 0], "gripper": gripper_open},  # Return to home
        ]

        # Right arm stays at home position
        self.right_trajectory = [
            {"t": 0, "joint": [0, 0, 0, 0, 0, 0], "gripper": gripper_open},
            {"t": 1500, "joint": [0, 0, 0, 0, 0, 0], "gripper": gripper_open}
        ]

        # 使用A*路径点生成base_trajectory
        self.base_trajectory = self._generate_base_trajectory_from_astar(path_to_pick, path_to_place, base_init, base_pick, base_place)
        
        # Precompute joint trajectory
        self.precompute_trajectory(ts_first)

    def _generate_base_trajectory_from_astar(self, path_to_pick, path_to_place, base_init, base_pick, base_place):
        """
        基于A*规划的路径生成base trajectory waypoints

        Args:
            path_to_pick: 从初始位置到拾取位置的A*路径点列表
            path_to_place: 从拾取位置到放置位置的A*路径点列表
            base_init: 初始基座位置 [x, y, yaw]
            base_pick: 拾取位置 [x, y, yaw]
            base_place: 放置位置 [x, y, yaw]

        Returns:
            base_trajectory: 基座轨迹waypoints列表
        """
        base_trajectory = []

        # 时间分配
        # 0-400: 移动到拾取位置
        # 400-600: 在拾取位置等待
        # 600-1000: 移动到放置位置
        # 1000-1400: 在放置位置等待机械臂操作
        # 1400-1500: 机械臂释放物体
        # 1500-2000: 在放置位置等待

        # 阶段1: 从初始位置移动到拾取位置 (0-400 时间步)
        move_to_pick_duration = 200  # 时间步
        if len(path_to_pick) > 1:
            # 在路径点之间均匀分配时间
            time_per_segment = move_to_pick_duration / (len(path_to_pick) - 1)
            for i, (x, y) in enumerate(path_to_pick):
                t = int(i * time_per_segment)
                base_trajectory.append({
                    "t": t,
                    "base": np.array([x, y, 0.0])  # 假设yaw为0
                })
        else:
            # 如果路径规划失败，使用直接路径
            base_trajectory.append({"t": 0, "base": base_init})
            base_trajectory.append({"t": 200, "base": base_pick})

        # 阶段2: 在拾取位置等待 (200-600)
        base_trajectory.append({"t": 600, "base": base_pick})

        # 阶段3: 从拾取位置移动到放置位置 (600-1000)
        move_to_place_start = 600
        move_to_place_duration = 600  # 1000 - 600
        if len(path_to_place) > 1:
            time_per_segment = move_to_place_duration / (len(path_to_place) - 1)
            for i, (x, y) in enumerate(path_to_place):
                t = int(move_to_place_start + i * time_per_segment)
                base_trajectory.append({
                    "t": t,
                    "base": np.array([x, y, 0.0])
                })
        else:
            # 如果路径规划失败，使用直接路径
            base_trajectory.append({"t": 600, "base": base_pick})
            base_trajectory.append({"t": 1200, "base": base_place})

        # 阶段4-6: 在放置位置等待和操作 (1200-2000)
        # base_trajectory.append({"t": 1400, "base": base_place})
        base_trajectory.append({"t": 1500, "base": base_place})
        base_trajectory.append({"t": 2000, "base": base_place})

        # 按时间排序并去重
        base_trajectory.sort(key=lambda x: x['t'])

        # 移除重复的时间戳，保留最后一个
        unique_trajectory = []
        seen_times = set()
        for wp in reversed(base_trajectory):
            if wp['t'] not in seen_times:
                unique_trajectory.append(wp)
                seen_times.add(wp['t'])
        unique_trajectory.reverse()

        print(f"生成的基座轨迹包含 {len(unique_trajectory)} 个waypoints")
        return unique_trajectory

    def precompute_trajectory(self, ts_first):
        self.precompute_joint_trajectory(ts_first)
        max_timestep = max(
            max(wp['t'] for wp in self.left_trajectory),
            max(wp['t'] for wp in self.right_trajectory)
        )
        for t in range(max_timestep + 1):
            base_vel = self._get_interpolated_basevelocity(t, self.base_trajectory)
            self.precomputed_trajectory[t]['base_vel'] = base_vel
        print(f"Precomputed trajectory with {len(self.precomputed_trajectory)} timesteps")

    def _get_interpolated_basevelocity(self, t, base_trajectory):
        curr_waypoint = None
        next_waypoint = None
        for i, wp in enumerate(base_trajectory):
            if wp['t'] <= t:
                curr_waypoint = wp
                if i + 1 < len(base_trajectory):
                    next_waypoint = base_trajectory[i + 1]
                else:
                    next_waypoint = wp
            else:
                next_waypoint = wp
                break
        
        t_range = next_waypoint['t'] - curr_waypoint['t']
        if t_range == 0:
            return np.array([0.0, 0.0, 0.0])
        base_vel = (next_waypoint['base'] - curr_waypoint['base']) / (t_range * DT)
        return base_vel

    def __call__(self, ts):
        """Generate action for mobile dual piper: [vx, vy, omega, left_arm_joints(6), left_gripper(1), right_arm_joints(6), right_gripper(1)]"""
        # Generate trajectory at first timestep
        if self.step_count == 0:
            self.generate_trajectory(ts)
        
        # Get base velocity command
        # vx, vy, omega = self._get_base_velocity(self.step_count)
        
        # Get arm joint positions and gripper from precomputed trajectory
        if self.precomputed_trajectory is not None and self.step_count < len(self.precomputed_trajectory):
            step_data = self.get_precomputed_step(self.step_count)
            base_vel = step_data['base_vel']
            left_joints = step_data['left_joints']
            right_joints = step_data['right_joints']
            left_gripper = step_data['left_gripper']
            right_gripper = step_data['right_gripper']
        else:
            base_vel = np.array([0.0, 0.0, 0.0])
            # Fallback to home position
            left_joints = [0, 0, 0, 0, 0, 0]
            right_joints = [0, 0, 0, 0, 0, 0]
            left_gripper = 0.035  # Open
            right_gripper = 0.035  # Open
        
        # Normalize gripper positions (0-1 range)
        from constants import PIPER_GRIPPER_POSITION_NORMALIZE_FN
        left_gripper_normalized = PIPER_GRIPPER_POSITION_NORMALIZE_FN(left_gripper) if left_gripper is not None else 1.0
        right_gripper_normalized = PIPER_GRIPPER_POSITION_NORMALIZE_FN(right_gripper) if right_gripper is not None else 1.0
        
        # Build action: [vx, vy, omega, left_arm_joints(6), left_gripper(1), right_arm_joints(6), right_gripper(1)]
        action = np.concatenate([
            base_vel,
            left_joints,
            [left_gripper_normalized],
            right_joints,
            [right_gripper_normalized]
        ])
        
        self.step_count += 1
        if self.step_count >= len(self.precomputed_trajectory):
            self.step_count = 0

        return action