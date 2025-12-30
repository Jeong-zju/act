import numpy as np
import collections

import sys
import os
sys.path.append(os.path.dirname(__file__))
from constants import PIPER_GRIPPER_POSITION_NORMALIZE_FN
from constants import PIPER_GRIPPER_VELOCITY_NORMALIZE_FN
from constants import PIPER_GRIPPER_POSITION_OPEN

import mujoco
import mujoco.viewer

class TransferCubeTaskPiper():

    def __init__(self, random=None, enable_qtor=False):
        self.enable_qtor = enable_qtor
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
    def get_qtor(data):
        qtor_raw = data.actuator_force.copy()
        left_qtor_raw = qtor_raw[:8]
        right_qtor_raw = qtor_raw[8:16]
        left_arm_qtor = left_qtor_raw[:6]
        right_arm_qtor = right_qtor_raw[:6]
        left_gripper_qtor = left_qtor_raw[6]
        right_gripper_qtor = right_qtor_raw[6]
        return np.concatenate([left_arm_qtor, [left_gripper_qtor], right_arm_qtor, [right_gripper_qtor]])

    @staticmethod
    def get_env_state(data):
        env_state = data.qpos.copy()[16:]
        return env_state

    def get_observation(self, data, renderer=None):
        # note: it is important to do .copy()
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(data)
        obs['qvel'] = self.get_qvel(data)
        if self.enable_qtor:
            obs['qtor'] = self.get_qtor(data)
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


class InsertionTaskPiper(TransferCubeTaskPiper):

    def __init__(self, random=None, enable_qtor=False):
        self.enable_qtor = enable_qtor
        super().__init__(random=random)

    @staticmethod
    def get_env_state(data):
        env_state = data.qpos.copy()[16:]
        peg_pose = env_state[:7]
        socket_pose = env_state[7:]
        return {"peg": peg_pose, "socket": socket_pose}

    def get_observation(self, data, renderer=None):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(data)
        obs['qvel'] = self.get_qvel(data)
        if self.enable_qtor:
            obs['qtor'] = self.get_qtor(data)
        env_state = self.get_env_state(data)
        obs['env_state'] = dict()
        obs['env_state']["peg"] = env_state["peg"]
        obs['env_state']["socket"] = env_state["socket"]
        obs['images'] = dict()
        # obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        if renderer is not None:
            renderer.update_scene(data, camera="top")
            obs['images']['top'] = renderer.render()
        else:
            obs['images']['top'] = np.random.randint(0, 255, (480, 640, 3))

        return obs

    def get_reward(self, model, data):
        all_contact_pairs = []
        for i_contact in range(data.ncon):
            id_geom_1 = data.contact[i_contact].geom1
            id_geom_2 = data.contact[i_contact].geom2
            name_geom_1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, id_geom_1)
            name_geom_2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, id_geom_2)
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_right_gripper = ("red_peg", "piper_right/7_link7") in all_contact_pairs or \
                             ("red_peg", "piper_right/8_link8") in all_contact_pairs
        touch_left_gripper = ("socket-1", "piper_left/7_link7") in all_contact_pairs or \
                             ("socket-1", "piper_left/8_link8") in all_contact_pairs or \
                             ("socket-2", "piper_left/7_link7") in all_contact_pairs or \
                             ("socket-2", "piper_left/8_link8") in all_contact_pairs or \
                             ("socket-3", "piper_left/7_link7") in all_contact_pairs or \
                             ("socket-3", "piper_left/8_link8") in all_contact_pairs or \
                             ("socket-4", "piper_left/7_link7") in all_contact_pairs or \
                             ("socket-4", "piper_left/8_link8") in all_contact_pairs

        peg_touch_table = ("red_peg", "table") in all_contact_pairs
        socket_touch_table = ("socket-1", "table") in all_contact_pairs or \
                             ("socket-2", "table") in all_contact_pairs or \
                             ("socket-3", "table") in all_contact_pairs or \
                             ("socket-4", "table") in all_contact_pairs
        peg_touch_socket = ("red_peg", "socket-1") in all_contact_pairs or \
                           ("red_peg", "socket-2") in all_contact_pairs or \
                           ("red_peg", "socket-3") in all_contact_pairs or \
                           ("red_peg", "socket-4") in all_contact_pairs
        pin_touched = ("red_peg", "pin") in all_contact_pairs

        reward = 0
        if touch_left_gripper and touch_right_gripper: # touch both
            reward = 1
        if touch_left_gripper and touch_right_gripper and (not peg_touch_table) and (not socket_touch_table): # grasp both
            reward = 2
        if peg_touch_socket and (not peg_touch_table) and (not socket_touch_table): # peg and socket touching
            reward = 3
        if pin_touched: # successful insertion
            reward = 4
        return reward


class MobileDualPiperTaskPiper():
    
    def __init__(self, enable_qtor=False):
        self.enable_qtor = enable_qtor

    @staticmethod
    def get_qpos(data):
        qpos_raw = data.qpos.copy()
        base_qpos = qpos_raw[:3]
        base_qpos[2] = (base_qpos[2] + np.pi) % (2 * np.pi) - np.pi
        left_arm_qpos = qpos_raw[3:9]
        right_arm_qpos = qpos_raw[10:16]
        left_gripper_qpos = [PIPER_GRIPPER_POSITION_NORMALIZE_FN(qpos_raw[6])]
        right_gripper_qpos = [PIPER_GRIPPER_POSITION_NORMALIZE_FN(qpos_raw[13])]
        return np.concatenate([base_qpos, left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(data):
        qvel_raw = data.qvel.copy()
        left_arm_qvel = qvel_raw[3:9]
        right_arm_qvel = qvel_raw[10:16]
        left_gripper_qvel = [PIPER_GRIPPER_VELOCITY_NORMALIZE_FN(qvel_raw[6])]
        right_gripper_qvel = [PIPER_GRIPPER_VELOCITY_NORMALIZE_FN(qvel_raw[13])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    @staticmethod
    def get_qtor(data):
        qtor_raw = data.actuator_force.copy()
        left_arm_qtor = qtor_raw[3:9]
        right_arm_qtor = qtor_raw[10:16]
        left_gripper_qtor = qtor_raw[6]
        right_gripper_qtor = qtor_raw[13]
        return np.concatenate([left_arm_qtor, left_gripper_qtor, right_arm_qtor, right_gripper_qtor])

    @staticmethod
    def get_env_state(data):
        env_state = data.qpos.copy()[19:]
        return env_state

    def get_observation(self, data, renderer=None):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(data)
        obs['qvel'] = self.get_qvel(data)
        if self.enable_qtor:
            obs['qtor'] = self.get_qtor(data)
        obs['env_state'] = self.get_env_state(data)
        obs['images'] = dict()
        if renderer is not None:
            renderer.update_scene(data, camera="top")
            obs['images']['top'] = renderer.render()
            renderer.update_scene(data, camera="left_cam")
            obs['images']['left'] = renderer.render()
            renderer.update_scene(data, camera="right_cam")
            obs['images']['right'] = renderer.render()
        else:
            obs['images']['top'] = np.random.randint(0, 255, (480, 640, 3))
            obs['images']['left'] = np.random.randint(0, 255, (480, 640, 3))
            obs['images']['right'] = np.random.randint(0, 255, (480, 640, 3))
        return obs