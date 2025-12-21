from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence, List
from scipy.spatial.transform import Rotation as R

import numpy as np
import pybullet as p

# PyBullet getClosestPoints signature supports linkIndexA/linkIndexB, defaulting to -2 in many wrappers/docs. :contentReference[oaicite:2]{index=2}
BASE_LINK = -1
ALL_LINKS = -2

CIRCULAR_LIMITS = (-np.pi, np.pi)


@dataclass(frozen=True)
class DisabledCollision:
    link1: str
    link2: str
    reason: str = ""


def connect(gui: bool = False) -> int:
    """Create a PyBullet client connection and return physicsClientId."""
    return p.connect(p.GUI if gui else p.DIRECT)


def disconnect(physicsClientId: int) -> None:
    p.disconnect(physicsClientId=physicsClientId)


def load_model(
    urdf_path: str,
    *,
    use_fixed_base: bool = True,
    use_self_collision: bool = True,
    base_position=(0.0, 0.0, 0.0),
    base_orientation=(0.0, 0.0, 0.0, 1.0),
    physicsClientId: int = 0,
) -> int:
    # Self-collision can be enabled using URDF_USE_SELF_COLLISION. :contentReference[oaicite:3]{index=3}
    flags = 0
    if use_self_collision:
        flags |= p.URDF_USE_SELF_COLLISION
    return p.loadURDF(
        urdf_path,
        basePosition=base_position,
        baseOrientation=base_orientation,
        useFixedBase=use_fixed_base,
        flags=flags,
        physicsClientId=physicsClientId,
    )


def joint_from_name(body_id: int, joint_name: str, *, physicsClientId: int = 0) -> int:
    n = p.getNumJoints(body_id, physicsClientId=physicsClientId)
    for j in range(n):
        info = p.getJointInfo(body_id, j, physicsClientId=physicsClientId)
        name = info[1].decode("utf-8")
        if name == joint_name:
            return j
    raise KeyError(f"Joint not found: {joint_name}")


def link_from_name(body_id: int, link_name: str, *, physicsClientId: int = 0) -> int:
    # Base link handling: users often refer to "base_link" in SRDF.
    # PyBullet base link index is -1 (BASE_LINK).
    base_info = p.getBodyInfo(body_id, physicsClientId=physicsClientId)
    # base_info fields are bytes; decode defensively
    base_name = base_info[0].decode("utf-8") if base_info and base_info[0] else ""
    body_name = (
        base_info[1].decode("utf-8")
        if base_info and len(base_info) > 1 and base_info[1]
        else ""
    )
    if link_name in {base_name, body_name, "base", "base_link"}:
        return BASE_LINK

    n = p.getNumJoints(body_id, physicsClientId=physicsClientId)
    for j in range(n):
        info = p.getJointInfo(body_id, j, physicsClientId=physicsClientId)
        child_link_name = info[12].decode("utf-8")
        if child_link_name == link_name:
            return j  # In PyBullet, child link index equals joint index.
    raise KeyError(f"Link not found: {link_name}")


def set_joint_positions(
    body_id: int,
    joint_ids: Sequence[int],
    joint_positions: Sequence[float],
    *,
    physicsClientId: int = 0,
) -> None:
    # resetJointState is the standard way to “teleport” joints for queries/initialization. :contentReference[oaicite:4]{index=4}
    for j, q in zip(joint_ids, joint_positions):
        p.resetJointState(
            body_id, j, q, targetVelocity=0.0, physicsClientId=physicsClientId
        )


def get_joint_positions(
    body_id: int, joint_ids: Sequence[int], *, physicsClientId: int = 0
) -> list[float]:
    out: list[float] = []
    for j in joint_ids:
        state = p.getJointState(body_id, j, physicsClientId=physicsClientId)
        out.append(float(state[0]))
    return out


def _joint_limits_from_pybullet(
    body_id: int, joint_id: int, *, physicsClientId: int = 0
) -> tuple[float, float]:
    info = p.getJointInfo(body_id, joint_id, physicsClientId=physicsClientId)
    # getJointInfo includes jointLowerLimit and jointUpperLimit for revolute/slider joints. :contentReference[oaicite:5]{index=5}
    lower = float(info[8])
    upper = float(info[9])
    return lower, upper


def get_custom_limits(
    body_id: int,
    joint_ids: Sequence[int],
    *,
    circular_limits: tuple[float, float] = CIRCULAR_LIMITS,
    physicsClientId: int = 0,
) -> tuple[list[float], list[float]]:
    lower_all: list[float] = []
    upper_all: list[float] = []
    for j in joint_ids:
        lower, upper = _joint_limits_from_pybullet(
            body_id, j, physicsClientId=physicsClientId
        )

        # Heuristic for continuous joints: many URDFs encode them with upper < lower or a degenerate range.
        if (upper < lower) or (abs(upper - lower) < 1e-9):
            lower, upper = circular_limits

        lower_all.append(lower)
        upper_all.append(upper)
    return lower_all, upper_all


def get_sample_fn(
    body_id: int,
    joint_ids: Sequence[int],
    *,
    seed: Optional[int] = None,
    physicsClientId: int = 0,
) -> Callable[[], list[float]]:
    lower, upper = get_custom_limits(
        body_id, joint_ids, physicsClientId=physicsClientId
    )
    rng = np.random.default_rng(seed)

    def _sample() -> list[float]:
        return [float(rng.uniform(l, u)) for l, u in zip(lower, upper)]

    return _sample


def get_disabled_collisions(
    robot_id: int,
    disabled: Sequence[DisabledCollision],
    *,
    also_disable_in_pybullet: bool = True,
    physicsClientId: int = 0,
) -> set[tuple[int, int]]:
    """
    Returns a set of unordered link-index pairs (a, b) with a < b.
    Optionally also applies PyBullet collision filters to disable detection between those links.
    PyBullet provides setCollisionFilterPair to enable/disable collision detection between link pairs. :contentReference[oaicite:6]{index=6}
    """
    pairs: set[tuple[int, int]] = set()
    for item in disabled:
        try:
            a = link_from_name(robot_id, item.link1, physicsClientId=physicsClientId)
            b = link_from_name(robot_id, item.link2, physicsClientId=physicsClientId)
        except KeyError:
            continue
        x, y = (a, b) if a < b else (b, a)
        pairs.add((x, y))
        if also_disable_in_pybullet:
            p.setCollisionFilterPair(
                robot_id,
                robot_id,
                a,
                b,
                enableCollision=0,
                physicsClientId=physicsClientId,
            )
    return pairs


def get_collision_fn(
    robot_id: int,
    joint_ids: Sequence[int],
    *,
    obstacles: Sequence[int] = (),
    disabled_collisions: set[tuple[int, int]] = frozenset(),
    self_collisions: bool = True,
    physicsClientId: int = 0,
) -> Callable[[Sequence[float]], bool]:
    """
    Returns collision_fn(q) -> bool.

    This mirrors the core behavior of pybullet_planning.get_collision_fn, which relies on
    getClosestPoints-based checks under the hood. :contentReference[oaicite:7]{index=7}
    """
    # Robot link indices we will test. Include base and each child link of the controlled joints.
    robot_links = [BASE_LINK] + list(joint_ids)

    def _is_disabled(a: int, b: int) -> bool:
        x, y = (a, b) if a < b else (b, a)
        return (x, y) in disabled_collisions

    def collision_fn(q: Sequence[float]) -> bool:
        set_joint_positions(robot_id, joint_ids, q, physicsClientId=physicsClientId)

        # 1) Self-collision
        if self_collisions:
            for i in range(len(robot_links)):
                for j in range(i + 1, len(robot_links)):
                    li, lj = robot_links[i], robot_links[j]
                    if _is_disabled(li, lj):
                        continue
                    pts = p.getClosestPoints(
                        bodyA=robot_id,
                        bodyB=robot_id,
                        distance=0.0,
                        linkIndexA=li,
                        linkIndexB=lj,
                        physicsClientId=physicsClientId,
                    )
                    if len(pts) > 0:
                        return True

        # 2) Robot vs obstacles
        for obs_id in obstacles:
            for li in robot_links:
                pts = p.getClosestPoints(
                    bodyA=robot_id,
                    bodyB=obs_id,
                    distance=0.0,
                    linkIndexA=li,
                    linkIndexB=ALL_LINKS,  # check against all links of the obstacle body :contentReference[oaicite:8]{index=8}
                    physicsClientId=physicsClientId,
                )
                if len(pts) > 0:
                    return True

        return False

    return collision_fn


def set_base_pose(
    body_id: int,
    pose: tuple[Sequence[float], Sequence[float]],
    *,
    physicsClientId: int = 0,
) -> None:
    """Set the base pose (position and orientation) of a body."""
    position, orientation = pose
    p.resetBasePositionAndOrientation(
        body_id, position, orientation, physicsClientId=physicsClientId
    )


def set_base_position(
    body_id: int,
    position: Sequence[float],
    *,
    physicsClientId: int = 0,
) -> None:
    """Set the base position of a body, keeping the current orientation."""
    _, orientation = p.getBasePositionAndOrientation(
        body_id, physicsClientId=physicsClientId
    )
    p.resetBasePositionAndOrientation(
        body_id, position, orientation, physicsClientId=physicsClientId
    )


def set_base_orientation(
    body_id: int,
    orientation: Sequence[float],
    *,
    physicsClientId: int = 0,
) -> None:
    """Set the base orientation (quaternion) of a body, keeping the current position."""
    position, _ = p.getBasePositionAndOrientation(
        body_id, physicsClientId=physicsClientId
    )
    p.resetBasePositionAndOrientation(
        body_id, position, orientation, physicsClientId=physicsClientId
    )


def get_base_pose(
    body_id: int,
    *,
    physicsClientId: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Get the base pose (position and orientation) of a body."""
    position, orientation = p.getBasePositionAndOrientation(
        body_id, physicsClientId=physicsClientId
    )
    return np.array(position), np.array(orientation)


def get_link_pose(
    body_id: int,
    link_id: int,
    *,
    physicsClientId: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Get the world pose (position and orientation) of a specific link."""
    link_state = p.getLinkState(
        body_id, link_id, physicsClientId=physicsClientId
    )
    position = np.array(link_state[0])
    orientation = np.array(link_state[1])
    return position, orientation


def compute_view_matrix_from_yaw_pitch_roll(
    cameraTargetPosition: Sequence[float],
    distance: float,
    yaw: float,
    pitch: float,
    roll: float,
    upAxisIndex: int,
    *,
    physicsClientId: int = 0,
) -> list[float]:
    """Compute view matrix from yaw, pitch, roll angles."""
    return p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=cameraTargetPosition,
        distance=distance,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        upAxisIndex=upAxisIndex,
        physicsClientId=physicsClientId,
    )


def compute_projection_matrix(
    fov: float,
    aspect: float,
    nearVal: float,
    farVal: float,
    *,
    physicsClientId: int = 0,
) -> list[float]:
    """Compute projection matrix from camera parameters."""
    return p.computeProjectionMatrixFOV(
        fov=fov,
        aspect=aspect,
        nearVal=nearVal,
        farVal=farVal,
        physicsClientId=physicsClientId,
    )


def compute_view_matrix(
    cameraEyePosition: Sequence[float],
    cameraTargetPosition: Sequence[float],
    cameraUpVector: Sequence[float],
    *,
    physicsClientId: int = 0,
) -> list[float]:
    """Compute view matrix from camera eye, target, and up vector."""
    return p.computeViewMatrix(
        cameraEyePosition=cameraEyePosition,
        cameraTargetPosition=cameraTargetPosition,
        cameraUpVector=cameraUpVector,
        physicsClientId=physicsClientId,
    )


def compute_projection_matrix_fov(
    fov: float,
    aspect: float,
    nearVal: float,
    farVal: float,
    *,
    physicsClientId: int = 0,
) -> list[float]:
    """Compute projection matrix FOV from camera parameters."""
    return p.computeProjectionMatrixFOV(
        fov=fov,
        aspect=aspect,
        nearVal=nearVal,
        farVal=farVal,
        physicsClientId=physicsClientId,
    )


def get_camera_image(
    width: int,
    height: int,
    viewMatrix: Sequence[float],
    projectionMatrix: Sequence[float],
    *,
    physicsClientId: int = 0,
    **kwargs,
) -> tuple:
    """Get camera image from PyBullet."""
    return p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=viewMatrix,
        projectionMatrix=projectionMatrix,
        physicsClientId=physicsClientId,
        **kwargs,
    )


def display_rgb_image_cv2(
    rgb_image: np.ndarray,
    *,
    window_name: str = "Camera Image",
) -> None:
    """Display RGB image using OpenCV in a separate window.

    Args:
        rgb_image: RGB image as numpy array (height, width, 3)
        window_name: Name for the OpenCV window
    """
    import cv2

    # Ensure image is in the right format (uint8, RGB)
    if rgb_image.dtype != np.uint8:
        rgb_image = (rgb_image * 255).astype(np.uint8)

    # Convert RGB to BGR for OpenCV display
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    # Display image in OpenCV window
    cv2.imshow(window_name, bgr_image)
    cv2.waitKey(1)  # Allow OpenCV to process events


def draw_frame(
    pose: tuple[Sequence[float], Sequence[float]],
    length: float,
    *,
    physicsClientId: int = 0,
) -> list[int]:
    """Draw a coordinate frame with RGB lines (X=red, Y=green, Z=blue).

    Args:
        pose: Tuple of (position, orientation) where orientation is a quaternion [x,y,z,w]
        length: Length of each axis line
        physicsClientId: PyBullet client ID

    Returns:
        List of 3 debug line IDs [x_line_id, y_line_id, z_line_id]
    """
    position, orientation = pose
    pos = np.array(position)

    # Convert quaternion to rotation matrix
    from scipy.spatial.transform import Rotation as R
    rot = R.from_quat([orientation[0], orientation[1], orientation[2], orientation[3]])

    # Define axis directions in local frame
    x_axis = np.array([length, 0, 0])  # X axis (red)
    y_axis = np.array([0, length, 0])  # Y axis (green)
    z_axis = np.array([0, 0, length])  # Z axis (blue)

    # Transform to world frame
    x_world = rot.apply(x_axis)
    y_world = rot.apply(y_axis)
    z_world = rot.apply(z_axis)

    # Draw the lines
    x_line_id = p.addUserDebugLine(
        pos,
        pos + x_world,
        lineColorRGB=[1, 0, 0],  # Red for X axis
        lineWidth=3,
        physicsClientId=physicsClientId
    )

    y_line_id = p.addUserDebugLine(
        pos,
        pos + y_world,
        lineColorRGB=[0, 1, 0],  # Green for Y axis
        lineWidth=3,
        physicsClientId=physicsClientId
    )

    z_line_id = p.addUserDebugLine(
        pos,
        pos + z_world,
        lineColorRGB=[0, 0, 1],  # Blue for Z axis
        lineWidth=3,
        physicsClientId=physicsClientId
    )

    return [x_line_id, y_line_id, z_line_id]


def load_ground_plane(
    *,
    plane_path: Optional[str] = None,
    base_position: Sequence[float] = (0.0, 0.0, 0.0),
    base_orientation: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
    physicsClientId: int = 0,
) -> int:
    """Load a ground plane into the simulation.

    Args:
        plane_path: Path to plane URDF file. If None, creates a plane programmatically.
        base_position: Position of the ground plane
        base_orientation: Orientation of the ground plane (quaternion)
        physicsClientId: PyBullet client ID

    Returns:
        Body ID of the loaded ground plane
    """
    if plane_path is not None:
        # Load custom plane URDF
        return p.loadURDF(
            plane_path,
            basePosition=base_position,
            baseOrientation=base_orientation,
            physicsClientId=physicsClientId,
        )
    else:
        # Create a plane programmatically
        # Create collision shape (infinite plane)
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_PLANE,
            planeNormal=[0, 0, 1],  # Normal pointing up (Z axis)
            physicsClientId=physicsClientId
        )

        # Create visual shape (optional, for visualization)
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_PLANE,
            planeNormal=[0, 0, 1],
            rgbaColor=[0.8, 0.8, 0.8, 1],  # Light gray color
            physicsClientId=physicsClientId
        )

        # Create multi-body with the collision and visual shapes
        return p.createMultiBody(
            baseMass=0,  # Static body (infinite mass)
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=base_position,
            baseOrientation=base_orientation,
            physicsClientId=physicsClientId,
        )

def step_simulation(physicsClientId: int = 0) -> None:
    p.stepSimulation(physicsClientId=physicsClientId)


def configure_visualizer(
    *,
    enable_gui: bool = True,
    enable_depth_preview: bool = False,
    enable_segmentation_preview: bool = False,
    enable_shadows: bool = True,
    physicsClientId: int = 0,
) -> None:
    """Configure PyBullet visualizer settings.

    Args:
        enable_gui: Enable/disable GUI control panels (left panel)
        enable_depth_preview: Show/hide depth buffer preview
        enable_segmentation_preview: Show/hide segmentation preview
        enable_shadows: Enable/disable shadows
        physicsClientId: PyBullet client ID
    """
    # Configure GUI panels
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, int(enable_gui), physicsClientId=physicsClientId)

    # Configure preview panels (typically on the right)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, int(enable_depth_preview), physicsClientId=physicsClientId)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, int(enable_segmentation_preview), physicsClientId=physicsClientId)

    # Configure shadows
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, int(enable_shadows), physicsClientId=physicsClientId)


def disable_side_panels(physicsClientId: int = 0) -> None:
    """Disable the left and right side panels in PyBullet window for cleaner view."""
    configure_visualizer(
        enable_gui=False,  # Disable left control panel
        enable_depth_preview=False,  # Disable depth preview (right panel)
        enable_segmentation_preview=False,  # Disable segmentation preview (right panel)
        enable_shadows=True,  # Keep shadows enabled
        physicsClientId=physicsClientId
    )

def set_gravity(gravity: List[float] = [0, 0, -9.81], physicsClientId: int = 0) -> None:
    p.setGravity(*gravity, physicsClientId=physicsClientId)


def create_shape(
    shape: str,
    size: Sequence[float],
    pose: tuple[Sequence[float], Sequence[float]],
    color: Sequence[float],
    *,
    mass: float = 1.0,
    friction: float = 0.5,
    physicsClientId: int = 0,
) -> int:
    """Create a 3D shape (box, sphere, cylinder, capsule) with collision and visual properties.

    Args:
        shape: Shape type ("box", "sphere", "cylinder", "capsule")
        size: Size parameters depending on shape:
            - "box": [half_width, half_height, half_depth]
            - "sphere": [radius]
            - "cylinder": [radius, half_height]
            - "capsule": [radius, half_height]
        pose: Tuple of (position, orientation) where orientation is quaternion [x,y,z,w]
        color: RGBA color [r, g, b, a] with values 0-1
        mass: Mass of the shape (0 for static)
        friction: Lateral friction coefficient (default 0.5)
        physicsClientId: PyBullet client ID

    Returns:
        Body ID of the created shape

    Examples:
        # Create a red box
        box_id = create_shape(
            "box", [0.1, 0.1, 0.1],
            ([0, 0, 0.5], [0, 0, 0, 1]),
            [1, 0, 0, 1]
        )

        # Create a blue sphere
        sphere_id = create_shape(
            "sphere", [0.05],
            ([0.5, 0, 0.5], [0, 0, 0, 1]),
            [0, 0, 1, 1]
        )
    """
    position, orientation = pose

    # Create collision shape based on type
    if shape.lower() == "box":
        if len(size) != 3:
            raise ValueError("Box shape requires 3 size parameters [half_width, half_height, half_depth]")
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=size,
            physicsClientId=physicsClientId
        )
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=size,
            rgbaColor=color,
            physicsClientId=physicsClientId
        )

    elif shape.lower() == "sphere":
        if len(size) != 1:
            raise ValueError("Sphere shape requires 1 size parameter [radius]")
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=size[0],
            physicsClientId=physicsClientId
        )
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=size[0],
            rgbaColor=color,
            physicsClientId=physicsClientId
        )

    elif shape.lower() == "cylinder":
        if len(size) != 2:
            raise ValueError("Cylinder shape requires 2 size parameters [radius, half_height]")
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER,
            radius=size[0],
            height=size[1] * 2,  # PyBullet expects full height
            physicsClientId=physicsClientId
        )
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=size[0],
            length=size[1] * 2,  # PyBullet expects full length
            rgbaColor=color,
            physicsClientId=physicsClientId
        )

    elif shape.lower() == "capsule":
        if len(size) != 2:
            raise ValueError("Capsule shape requires 2 size parameters [radius, half_height]")
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_CAPSULE,
            radius=size[0],
            height=size[1] * 2,  # PyBullet expects full height
            physicsClientId=physicsClientId
        )
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_CAPSULE,
            radius=size[0],
            length=size[1] * 2,  # PyBullet expects full length
            rgbaColor=color,
            physicsClientId=physicsClientId
        )

    else:
        raise ValueError(f"Unsupported shape type: {shape}. Supported: 'box', 'sphere', 'cylinder', 'capsule'")

    # Create multi-body
    body_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_shape_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=position,
        baseOrientation=orientation,
        physicsClientId=physicsClientId,
    )

    # Set friction properties for better grasping
    p.changeDynamics(
        body_id,
        -1,  # -1 means base link
        lateralFriction=friction,
        spinningFriction=0.0,  # Reduce spinning friction
        rollingFriction=0.0,   # Reduce rolling friction
        physicsClientId=physicsClientId
    )

    return body_id


def set_friction(
    body_id: int,
    link_id: int,
    friction: float,
    *,
    physicsClientId: int = 0,
) -> None:
    """Set friction coefficient for a specific link of a body.

    Args:
        body_id: The body ID
        link_id: The link ID (-1 for base link)
        friction: Lateral friction coefficient
        physicsClientId: PyBullet client ID
    """
    p.changeDynamics(
        body_id,
        link_id,
        lateralFriction=friction,
        spinningFriction=0.0,  # Reduce spinning friction
        rollingFriction=0.0,   # Reduce rolling friction
        physicsClientId=physicsClientId
    )


def calculate_inverse_kinematics(
    body_id: int,
    end_effector_link_id: int,
    target_position: Sequence[float],
    target_orientation: Optional[Sequence[float]] = None,
    *,
    joint_ids: Optional[Sequence[int]] = None,
    current_joint_positions: Optional[Sequence[float]] = None,
    physicsClientId: int = 0,
) -> List[float]:
    """Calculate inverse kinematics for a robot arm.

    Args:
        body_id: The body ID of the robot
        end_effector_link_id: The link ID of the end effector
        target_position: Target position [x, y, z]
        target_orientation: Target orientation as quaternion [x, y, z, w] (optional)
        joint_ids: Joint IDs to solve for (optional, uses all joints if None)
        current_joint_positions: Current joint positions for IK solver (optional)
        physicsClientId: PyBullet client ID

    Returns:
        List of joint positions that achieve the target pose
    """
    return p.calculateInverseKinematics(
        body_id,
        end_effector_link_id,
        target_position,
        targetOrientation=target_orientation,
        jointDamping=None if joint_ids is None else [0.1] * len(joint_ids),
        solver=0,
        maxNumIterations=100,
        residualThreshold=1e-5,
        physicsClientId=physicsClientId,
    )


def calculate_accurate_inverse_kinematics(
    body_id: int,
    end_effector_link_id: int,
    target_position: Sequence[float],
    target_orientation: Optional[Sequence[float]] = None,
    *,
    threshold: float = 1e-4,
    max_iterations: int = 100,
    joint_ids: Optional[Sequence[int]] = None,
    physicsClientId: int = 0,
) -> List[float]:
    """Calculate accurate inverse kinematics by iterative refinement.

    This function iteratively refines the IK solution by calculating IK, setting joint states,
    checking the actual end effector pose, and repeating until the position error is below threshold.
    Based on the reference implementation that ensures precise positioning.

    Args:
        body_id: The body ID of the robot
        end_effector_link_id: The link ID of the end effector
        target_position: Target position [x, y, z]
        target_orientation: Target orientation as quaternion [x, y, z, w] (optional)
        threshold: Position error threshold for convergence (distance squared)
        max_iterations: Maximum number of iterations
        joint_ids: Joint IDs to solve for (optional, uses all joints if None)
        physicsClientId: PyBullet client ID

    Returns:
        List of joint positions that achieve the target pose within threshold
    """
    close_enough = False
    iteration = 0
    position_error_squared = 1e30

    # Get joint IDs if not provided
    if joint_ids is None:
        num_joints = p.getNumJoints(body_id, physicsClientId=physicsClientId)
        joint_ids = list(range(num_joints))

    joint_positions = None

    while not close_enough and iteration < max_iterations:
        # Calculate IK (following the original reference implementation)
        ik_result = p.calculateInverseKinematics(
            body_id,
            end_effector_link_id,
            target_position,
            targetOrientation=target_orientation,
            physicsClientId=physicsClientId,
        )

        # Extract joint positions (PyBullet returns a tuple or list)
        if isinstance(ik_result, (tuple, list)):
            joint_positions = list(ik_result)
        else:
            # If it's a single value, wrap it in a list
            joint_positions = [ik_result]

        # Set joint positions using PyBullet's resetJointState (like the original)
        for i, joint_id in enumerate(joint_ids):
            if i < len(joint_positions):
                p.resetJointState(
                    body_id, joint_id, joint_positions[i],
                    targetVelocity=0.0, physicsClientId=physicsClientId
                )

        # Get actual end effector position
        link_state = p.getLinkState(body_id, end_effector_link_id, physicsClientId=physicsClientId)
        actual_position = link_state[4]

        # Calculate position error (squared distance, like the original)
        diff = [target_position[0] - actual_position[0],
                target_position[1] - actual_position[1],
                target_position[2] - actual_position[2]]
        position_error_squared = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])

        close_enough = position_error_squared < threshold
        iteration += 1

    # print(f"Accurate IK: {iteration} iterations, final position error squared: {position_error_squared}")
    return joint_positions

def set_time_step(time_step: float, physicsClientId: int = 0) -> None:
    p.setTimeStep(time_step, physicsClientId=physicsClientId)

def multiply(pose1: tuple[Sequence[float], Sequence[float]], pose2: tuple[Sequence[float], Sequence[float]]) -> tuple[Sequence[float], Sequence[float]]:
    position1, orientation1 = pose1
    position2, orientation2 = pose2
    
    rot1 = R.from_quat(orientation1)
    rot2 = R.from_quat(orientation2)
    position = position1 + rot1.apply(position2)
    orientation = (rot1 * rot2).as_quat()
    return (position, orientation)

def pose(position: Sequence[float], euler: Sequence[float]) -> tuple[Sequence[float], Sequence[float]]:
    rotation = R.from_euler('XYZ', euler)
    orientation = rotation.as_quat()
    return (position, orientation)

def matrix_from_pose(pose: tuple[Sequence[float], Sequence[float]]) -> np.ndarray:
    position, orientation = pose
    rotation = R.from_quat(orientation)
    matrix = np.eye(4)
    matrix[:3, :3] = rotation.as_matrix()
    matrix[:3, 3] = position
    return matrix

def inverse(pose: tuple[Sequence[float], Sequence[float]]) -> tuple[Sequence[float], Sequence[float]]:
    position, orientation = pose
    inverse_rotation = R.from_quat(orientation).inv()
    inverse_position = -inverse_rotation.apply(position)
    return (inverse_position, inverse_rotation.as_quat())

def remove_body(body_id: int, physicsClientId: int = 0) -> None:
    p.removeBody(body_id, physicsClientId=physicsClientId)