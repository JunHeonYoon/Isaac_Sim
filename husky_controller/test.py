from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.stage import add_reference_to_stage
import numpy as np

np.set_printoptions(precision=3, suppress=True)

import pandas as pd
import math
import keyboard
import wheelcontroller.differential as diffcontroller


file_path = "/home/dyros/Isaac_Sim/husky_controller/trajectory"
traj_data = pd.read_table(file_path+"/eight.txt", sep=" ", header=None)
traj_data = traj_data.to_numpy()
traj_linvel = np.sqrt(traj_data[:,3]**2 + traj_data[:,4]**2)
traj_angvel = (traj_data[:,3]*traj_data[:,6] - traj_data[:,5]*traj_data[:,4]) / (traj_data[:,3]**2 + traj_data[:,4]**2)
for i in range(0,traj_linvel.shape[0]):
    if traj_linvel[i] < 0.01:
        traj_angvel[i] = 0


hz = 50
model_path = "/home/dyros/Isaac_Sim/husky_controller/model"
world = World(stage_units_in_meters=1.0)
world.set_simulation_dt(1/hz, 1/hz) 
asset_path = model_path + "/husky.usd"
add_reference_to_stage(usd_path=asset_path, prim_path="/World/husky")

# dof_names : ['front_left_wheel', 'front_right_wheel', 'rear_left_wheel', 'rear_right_wheel']
husky = world.scene.add(Articulation(prim_path="/World/husky", name="husky",))

world.scene.add_default_ground_plane()
world.reset()


tick_init = 0
tick = 0
is_mode_changed = False
control_mode = "none"
position_init = np.zeros(3)
orientation_init = np.zeros(4)
position = np.zeros(3)
orientation = np.zeros(4)
linvel = np.zeros(3)
angvel = np.zeros(3)
desired_joint_vel = np.zeros(4)
desired_linvel = 0.
desired_angvel = 0.
joint_vel = np.zeros(4)


while simulation_app.is_running():
    world.step(render=True)
    if world.is_playing():
        if world.current_time_step_index == 0:
            world.reset()
            
            
        if keyboard.is_pressed("i"):
            is_mode_changed = True
            control_mode = "init"
        elif keyboard.is_pressed("1"):
            is_mode_changed = True
            control_mode = "Tracking"
        elif keyboard.is_pressed("2"):
            is_mode_changed = True
            control_mode = "Rotating"
            
        
        if is_mode_changed:
            print("Change the mode : " + control_mode)
            is_mode_changed = False
            tick_init = world.current_time_step_index
            position_init, orientation_init = husky.get_world_pose()
            linvel = husky.get_linear_velocity()
            angvel = husky.get_angular_velocity()
            tick = tick_init
            position = position_init
            orientation = orientation_init
            joint_vel = husky.get_joint_velocities()
            
            
        if control_mode == "init":
            husky.set_world_pose(position=np.array([0.,0.,0.0780]),
                                 orientation=np.array([1.,0.,0.,0.]))
            desired_joint_vel = np.zeros(4)
            
        elif control_mode == "Tracking":
            index = tick - tick_init
            if index < 0 or index >= traj_data.shape[0]:
                desired_linvel = 0
                desired_angvel = 0
            else:
                desired_linvel = traj_linvel[index]
                desired_angvel = traj_angvel[index]
            desired_joint_vel = diffcontroller.DifferentialController(wheel_radius=0.1651,
                                                                      wheel_distance=0.5708,
                                                                      linear_velocity=desired_linvel,
                                                                      angular_velocity=desired_angvel,
                                                                      max_linear_speed=1.0,
                                                                      max_angular_speed=2.0,
                                                                      is_skid=True,
                                                                      wheel_distance_multiplier=1.875)
        elif control_mode == "Rotating":
            desired_joint_vel = diffcontroller.DifferentialController(wheel_radius=0.1651,
                                                                      wheel_distance=0.5708,
                                                                      linear_velocity=0.,
                                                                      angular_velocity=math.pi/6,
                                                                      max_linear_speed=1.0,
                                                                      max_angular_speed=2.0,
                                                                      is_skid=True,
                                                                      wheel_distance_multiplier=1.875)
            
        tick = world.current_time_step_index
        position, orientation = husky.get_world_pose()
        linvel = husky.get_linear_velocity()
        angvel = husky.get_angular_velocity()
        husky.set_joint_velocities(desired_joint_vel)
        joint_vel = husky.get_joint_velocities()
        
        
        if( world.current_time_step_index % 50 == 0 ):
            yaw = math.atan2(2*(orientation[0]*orientation[3] + orientation[1]*orientation[2]), 1 - 2*(orientation[2]**2 + orientation[3]**2))
            print("----------------------------")
            print("Time[sec]")
            print(round(world.current_time, 2))
            print("position[m] : ")
            print(position[:2])
            print("orientation[deg] : ")
            print(round(yaw*180/math.pi, 2))
            print("Linear velocity[m/s] : ")
            print(round(math.sqrt(linvel[0]**2 + linvel[1]**2), 2))
            print("Angular velocity[deg/s] : ")
            print(round(angvel[2]*180/math.pi, 2))
            print("----------------------------")
            print("\n\n\n")
        
            
simulation_app.close()