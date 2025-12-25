import time
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from mujoco_lidar import MjLidarWrapper
from mujoco_lidar import scan_gen

# Define a simple MuJoCo scene
simple_demo_scene = """
<mujoco model="simple_demo">
    <worldbody>
        <!-- Ground + Four Walls -->
        <geom name="ground" type="plane" size="5 5 0.1" pos="0 0 0" rgba="0.2 0.9 0.9 1"/>
        <geom name="wall1" type="box" size="1e-3 3 1" pos=" 3 0 1" rgba="0.9 0.9 0.9 1"/>
        <geom name="wall2" type="box" size="1e-3 3 1" pos="-3 0 1" rgba="0.9 0.9 0.9 1"/>
        <geom name="wall3" type="box" size="3 1e-3 1" pos="0  3 1" rgba="0.9 0.9 0.9 1"/>
        <geom name="wall4" type="box" size="3 1e-3 1" pos="0 -3 1" rgba="0.9 0.9 0.9 1"/>

        <!-- Various Geometries -->
        <geom name="box1" type="box" size="0.5 0.5 0.5" pos="2 0 0.5" euler="45 -45 0" rgba="1 0 0 1"/>
        <geom name="sphere1" type="sphere" size="0.5" pos="0 2 0.5" rgba="0 1 0 1"/>
        <geom name="cylinder1" type="cylinder" size="0.4 0.6" pos="0 -2 0.4" euler="0 90 0" rgba="0 0 1 1"/>
        
        <!-- LiDAR Site -->
        <body name="lidar_base" pos="0 0 1" quat="1 0 0 0" mocap="true">
            <inertial pos="0 0 0" mass="1e-4" diaginertia="1e-9 1e-9 1e-9"/>
            <site name="lidar_site" size="0.001" type='sphere'/>
            <geom type="box" size="0.1 0.1 0.1" density="0" contype="0" conaffinity="0" rgba="0.3 0.6 0.3 0.2"/>
        </body>
    </worldbody>
</mujoco>
"""

# Create MuJoCo model and data
mj_model = mujoco.MjModel.from_xml_string(simple_demo_scene)
mj_data = mujoco.MjData(mj_model)

# Generate scan pattern
rays_theta, rays_phi = scan_gen.generate_grid_scan_pattern(num_ray_cols=64, num_ray_rows=16)

# Get body ID to exclude (avoid LiDAR detecting itself)
exclude_body_id = mj_model.body("lidar_base").id

# Create CPU backend LiDAR sensor
lidar = MjLidarWrapper(
    mj_model, 
    site_name="lidar_site",
    backend="cpu",  # Use CPU backend
    cutoff_dist=50.0,  # Maximum detection distance of 50 meters
    args={'bodyexclude': exclude_body_id}  # CPU backend specific parameter: exclude body
)

idx = 0

# Set up matplotlib for interactive plotting
plt.ion()
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('LiDAR Point Cloud')

# Use in simulation loop
with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(mj_model, mj_data)
        viewer.sync()
        
        # Perform ray tracing (Wrapper automatically handles pose updates)
        lidar.trace_rays(mj_data, rays_theta, rays_phi)
        
        # Get point cloud data (in local coordinate system)
        points = lidar.get_hit_points()  # shape: (N, 3)
        distances = lidar.get_distances()  # shape: (N,)

        print(f"Frame {idx}: {len(points)} points detected")
        
        # Plot points
        if len(points) > 0:
            ax.clear()
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f'LiDAR Point Cloud - Frame {idx}')
            
            # Color points by distance
            scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                               c=distances, cmap='viridis', s=10, alpha=0.6)
            # plt.colorbar(scatter, ax=ax, label='Distance (m)')
            
            # Set equal aspect ratio
            max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                                 points[:, 1].max() - points[:, 1].min(),
                                 points[:, 2].max() - points[:, 2].min()]).max() / 2.0
            mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
            mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
            mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            plt.draw()
            plt.pause(0.001)
        
        idx += 1
        time.sleep(1./60.)