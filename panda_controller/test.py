from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.franka.franka import Franka
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.types import ArticulationAction
import numpy as np
import math
import math_type_define.dyros_math as DyrosMath
import lula
import keyboard
import pandas as pd
from omni.isaac.core.articulations.articulation_view import ArticulationView

np.set_printoptions(precision=3, suppress=True)


# ---------------------------------------- Setting of Simulation -----------------------------------------------------------
hz = 100
model_path = "/home/dyros/Isaac_Sim/panda_controller/model"
world = World(stage_units_in_meters=1.0)
world.set_simulation_dt(1/hz, 1/hz) 
asset_path = model_path + "/panda_arm_hand.usd"
add_reference_to_stage(usd_path=asset_path, prim_path="/World/panda")

# define the manipulator
franka = world.scene.add(Franka(prim_path="/World/panda", name="franka_robot",
                                end_effector_prim_name="panda_hand",
                                gripper_dof_names = ["panda_finger_joint1", "panda_finger_joint2"],
                                gripper_open_position = np.array([0.04, 0.04]),
                                gripper_closed_position = np.array([0, 0]),))
kinematic = lula.RobotDescription.kinematics(lula.load_robot(robot_description_file=model_path + "/panda_robot_description.yaml",
                                                             robot_urdf=model_path + "/panda_arm_hand.urdf"))

# define force sensor of each finger
# right_finger_ft = ArticulationView(prim_paths_expr="/World/panda/panda_rightfinger", enable_dof_force_sensors=True)
ft = ArticulationView(prim_paths_expr="/World/panda", name="ft_viewer")
world.scene.add(ft)
# ft.initialize()
# if ft.initialized:
#     print("true")

# set the default positions of the other gripper joints to be opened so
# that its out of the way of the joints we want to control when gripping an object for instance.
joints_default_positions = np.zeros(9)
joints_default_positions[3] = -math.pi/2
joints_default_positions[5] = math.pi/2
joints_default_positions[6] = math.pi/4
franka.set_joints_default_state(positions=joints_default_positions)
world.scene.add_default_ground_plane()
world.reset()
# ----------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------ Initializing of Data -----------------------------------------------------------
q_init = joints_default_positions[:7]
q = q_init
q_desired = q_init
j = np.zeros((6,7))
j_qd = np.zeros((6,7))
pose_init = np.zeros((4,4))
pose = np.zeros((4,4))
pose_qd = np.zeros((4,4))
play_time = 0.0
control_start_time = 0.0
tick_init = 0
tick = 0
is_mode_changed = False
control_mode = "default"
# ----------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------- Getting Trajecroty -----------------------------------------------------------
file_path = "/home/dyros/Isaac_Sim/panda_controller/trajectory"
traj_data = pd.read_table(file_path+"/circle.txt", sep=" ", header=None)
traj_data = traj_data.to_numpy()

x_traj = np.zeros(traj_data.shape[0])
y_traj = traj_data[:, 1]
z_traj = traj_data[:, 2]

vx_traj = np.zeros(traj_data.shape[0])
vy_traj = traj_data[:, 3]
vz_traj = traj_data[:, 4]
# ---------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------- Simulation -----------------------------------------------------------
while simulation_app.is_running():
    world.step(render=True)
    if world.is_playing():
        if world.current_time_step_index == 0:
            world.reset()
            
            
    if keyboard.is_pressed("i"):
        is_mode_changed = True
        control_mode = "joint_ctrl_home"        
    elif keyboard.is_pressed("1"):
        is_mode_changed = True
        control_mode = "CLIK"
    
    if is_mode_changed:    
        print("Change the Mode : " + control_mode)
        is_mode_changed = False
        q_init = franka.get_joint_positions()[:7].T
        q = q_init
        q_desired = q_init
        j = kinematic.jacobian(cspace_position=q.astype(np.float64), frame="panda_hand")
        j_qd = kinematic.jacobian(cspace_position=q_desired.astype(np.float64), frame="panda_hand")
        pose_init = kinematic.pose(cspace_position=q.astype(np.float64), frame="panda_hand").matrix()
        pose = pose_init
        pose_qd = pose_init
        control_start_time = world.current_time
        play_time = control_start_time
        tick_init = world.current_time_step_index
        tick = tick_init
        
        # x_traj += np.ones(x_traj.shape[0]) * pose_init[0, 3]
        # y_traj += np.ones(y_traj.shape[0]) * pose_init[1, 3]
        # z_traj += np.ones(z_traj.shape[0]) * pose_init[2, 3]
        
    # --------------------------------- Update States -----------------------------------------------
    q = franka.get_joint_positions()[:7].T
    j = kinematic.jacobian(cspace_position=q.astype(np.float64), frame="panda_hand")
    j_qd = kinematic.jacobian(cspace_position=q_desired.astype(np.float64), frame="panda_hand")
    pose = kinematic.pose(cspace_position=q.astype(np.float64), frame="panda_hand").matrix()
    pose_qd = kinematic.pose(cspace_position=q_desired.astype(np.float64), frame="panda_hand").matrix()
    # ------------------------------------------------------------------------------------------------
        

    #---------------------------------- CLIK -----------------------------------------
    if control_mode=="joint_ctrl_home":
        q_desired = DyrosMath.cubicVector(play_time, control_start_time, control_start_time+1, 
                                          q_init, joints_default_positions[:7], np.zeros(7), np.zeros(7))
    elif control_mode=="CLIK":
        x_desired = np.zeros(3)  # position
        rotation_desired = np.zeros((3,3)) # rotation
        xd_desired = np.zeros(6) # v, w
        
        # ---------------------------------- Cubic Spline -------------------------------------------
        # duration = 2.0
        # target_pose = np.array([[1,  0,  0, 0.25],
        #                         [0, -1,  0, 0.28],
        #                         [0,  0, -1, 0.65],
        #                         [0,  0,  0, 1]])
        
        
        # for i in range(0,3):
        #     x_desired[i] = DyrosMath.cubic(play_time, control_start_time, control_start_time+duration,
        #                                    pose_init[i,3], target_pose[i,3], 0, 0)
        #     xd_desired[i] = DyrosMath.cubicDot(play_time, control_start_time, control_start_time+duration,
        #                                        pose_init[i,3], target_pose[i,3], 0, 0)
        # rotation_desired = DyrosMath.rotationCubic(play_time, control_start_time, control_start_time+duration,
        #                                            pose_init[:3,:3], target_pose[:3,:3])
        # xd_desired[3:] = DyrosMath.rotationCubicDot(play_time, control_start_time, control_start_time+duration,
        #                                             np.zeros(3), np.zeros(3), pose_init[:3,:3], target_pose[:3,:3])
        # -----------------------------------------------------------------------------------------------
        
        # --------------------------------- Traj from TXT -----------------------------------------------
        index = tick - tick_init
        
        if(index < 0):
            x_desired = pose_init[:3,3]
            rotation_desired = pose_init[:3,:3]
            xd_desired = np.zeros(6)
        elif(index >= traj_data.shape[0]):
            x_desired = np.array([x_traj[-1]+pose_init[0, 3], y_traj[-1]+pose_init[1, 3], z_traj[-1]+pose_init[2, 3]])
            rotation_desired = pose_init[:3,:3]
            xd_desired = np.zeros(6)
        else:
            x_desired = np.array([x_traj[index]+pose_init[0, 3], y_traj[index]+pose_init[1, 3], z_traj[index]+pose_init[2, 3]])
            rotation_desired = pose_init[:3,:3]
            xd_desired = np.array([vx_traj[index], vy_traj[index], vz_traj[index],
                                0, 0, 0])
        # -----------------------------------------------------------------------------------------------
        
        x_error = np.zeros(6)
        x_error[:3] = x_desired - pose_qd[:3,3]
        x_error[3:] = DyrosMath.getPhi(pose_qd[:3,:3], rotation_desired)
        
        kp = np.zeros((6,6))
        np.fill_diagonal(kp, val=[50, 50, 50, 10, 10, 10])
        
        pseudo_j_inv = np.dot( j_qd.T, np.linalg.inv( np.dot(j_qd, j_qd.T) ) )
        qd_desired = np.dot(pseudo_j_inv, xd_desired + np.dot(kp, x_error) )
        q_desired = q_desired + qd_desired/hz
    #-------------------------------------------------------------------------------------
        
    play_time = world.current_time
    tick = world.current_time_step_index
    franka.set_joint_positions(positions=np.concatenate((q_desired, np.zeros(2)), axis=0))
        
    #------------------------------- Print States ----------------------------------
    if( world.current_time_step_index % 50 == 0 ):
        print("position : ")
        print(pose[:3, 3].T)
        print("orientation : ")
        print(pose[:3,:3])
        print("\n\n")
        
        # print("right_ft")
        # print(right_finger_ft._physics_view.get_force_sensor_forces())
        print("left_ft")
        print(ft._physics_view.get_force_sensor_forces())
        # print("ft_sensor : ")
        # print(ft_sensor._physics_view)
        
    #-------------------------------------------------------------------------------

simulation_app.close()
# ----------------------------------------------------------------------------------------------------------------------------
