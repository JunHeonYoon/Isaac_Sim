DOF = 3
import numpy as np
import pandas as pd
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
import math_type_define.dyros_math as DyrosMath
import math
import wheelcontroller.differential as diffcontroller

np.set_printoptions(precision=3, suppress=True)

class WheelController:
    def __init__(self, hz: float, world:World, pkg_path:str) -> None:
        self.hz = hz
        self.world = world
        self.tick = 0
        self.play_time = 0
        self.control_mode = "none"
        self.is_mode_changed = False
        self.pkg_path = pkg_path
        self.file_names = ["Circle", "Square", "Eight"]
        self.initDimension()
        self.initModel()        
        self.initFile()
        
    def initDimension(self)->None:
        self.pose_init = np.zeros(DOF) # x, y, theta
        self.pose = np.zeros(DOF)
        self.pose_desired = np.zeros(DOF) 
        # self.pose_dot_init = np.zeros(DOF) # Vx, Vy, W (for world frame)
        self.pose_dot = np.zeros(DOF)
        self.pose_dot_desired = np.zeros(DOF)
        # self.v_init = np.zeros(DOF) # Vx, Vy, W (for base frame)
        self.v = np.zeros(DOF)
        self.v_desired = np.zeros(DOF) 
        
        # self.joint_vel_init = np.zeros(4) # Wheel velocity
        self.joint_vel = np.zeros(4)
        self.joint_vel_desired = np.zeros(4)
        
        
    
    def initModel(self)->None:
        model_path = self.pkg_path + "/model"
        self.husky = self.world.scene.add(Articulation(prim_path="/World/husky", name="husky",))
        
    def initFile(self)->None:
        for i in range(len(self.file_names)):
            file_path = self.pkg_path + "/data/" + self.file_names[i] + ".txt"
            globals()['self.file_'+str(i)] = open(file_path, "w")
        
        
        
    def printState(self)->None:
        if( self.world.current_time_step_index % 50 == 0 ):
            print("----------------------------")
            print("Time[sec]")
            print(round(self.world.current_time, 2))
            print("\npose[m, m, rad] : ")
            print(self.pose)
            print(self.pose_desired)
            print("\nLinear velocity[m/s, m/s, rad/s] : ")
            print(self.v)
            print(self.v_desired)
            print("----------------------------")
            print("\n\n\n")
            
    def Quater2Yaw(self, quaternion:np.array)->float:
        yaw = math.atan2(2*(quaternion[0]*quaternion[3] + quaternion[1]*quaternion[2]), 1 - 2*(quaternion[2]**2 + quaternion[3]**2))
        return yaw
            
    
    def getTrajData(self, file_name:str)->np.array:
        traj_data = pd.read_table(self.pkg_path+"/trajectory"+file_name, sep=" ", header=None)
        traj_data = traj_data.to_numpy()

        pose_x_traj = traj_data[:, 1]
        pose_y_traj = traj_data[:, 2]
        pose_dot_x_traj = traj_data[:, 3]
        pose_dot_y_traj = traj_data[:, 4]
        a_x_traj = traj_data[:, 5]
        a_y_traj = traj_data[:, 6]
        
        th_traj = np.zeros(traj_data.shape[0])
        for i in range(th_traj.shape[0]): # For differential model
            if i == 0:
                th_traj[i] = self.pose_init[2]
            else:
                if math.atan2(pose_dot_y_traj[i], pose_dot_x_traj[i]) < 0.001:
                    th_traj[i] = th_traj[i-1]
                else:
                    th_traj[i] = math.atan2(pose_dot_y_traj[i], pose_dot_x_traj[i])
                    
        w_traj = (pose_dot_x_traj*a_y_traj - a_x_traj*pose_dot_y_traj) / (pose_dot_x_traj**2 + pose_dot_y_traj**2) # For differential model
        for i in range(w_traj.shape[0]):
            if (pose_dot_x_traj[i]**2 + pose_dot_y_traj[i]**2) < 0.001:
                w_traj[i] = 0

        
        v_x_traj =  pose_dot_x_traj * np.cos(th_traj) + pose_dot_y_traj * np.sin(th_traj)
        v_y_traj = -pose_dot_x_traj * np.sin(th_traj) + pose_dot_y_traj * np.cos(th_traj)

        traj = np.column_stack((pose_x_traj, pose_y_traj))
        traj = np.column_stack((traj, th_traj))
        traj = np.column_stack((traj, v_x_traj))
        traj = np.column_stack((traj, v_y_traj))
        traj = np.column_stack((traj, w_traj))
        return traj # x, y, theta, vx, vy, w(for base frame)
    
    def Tracking(self, traj_data: np.array):
        index = self.tick - self.tick_init
        if index < 0 or index >= traj_data.shape[0]:
            self.v_desired = np.zeros(DOF)
            self.pose_desired = self.pose
        else:
            self.v_desired = np.array([traj_data[index, 3], traj_data[index, 4], traj_data[index, 5]])
            self.pose_desired = np.array([traj_data[index, 0], traj_data[index, 1], traj_data[index, 2]])
        self.joint_vel_desired = diffcontroller.DifferentialController(wheel_radius=0.1651,
                                                                       wheel_distance=0.5708,
                                                                       linear_velocity=self.v_desired.item(0),
                                                                       angular_velocity=self.v_desired.item(2),
                                                                       max_linear_speed=1.0,
                                                                       max_angular_speed=2.0,
                                                                       is_skid=True,
                                                                       wheel_distance_multiplier=1.875)
        self.record(0, traj_data.shape[0]/self.hz)
            

    def UpdateData(self)->None:
        position, orientation = self.husky.get_world_pose()
        self.pose[:2] = position[:2]
        self.pose[2] = self.Quater2Yaw(orientation)
        
        linvel = self.husky.get_linear_velocity()
        angvel = self.husky.get_angular_velocity()
        self.pose_dot[:2] = linvel[:2]
        self.pose_dot[2] = angvel[2]
        
        self.v[0] =  self.pose_dot[0] * math.cos(self.pose[2]) + self.pose_dot[1] * math.sin(self.pose[2])
        self.v[1] = -self.pose_dot[0] * math.sin(self.pose[2]) + self.pose_dot[1] * math.cos(self.pose[2])
        
        self.joint_vel = self.husky.get_joint_velocities()
        
        
    def initPosition(self)->None:
        self.pose_init = self.pose
        self.pose_desired = self.pose_init
        self.v_desired = self.v
        self.joint_vel_desired = self.joint_vel
        
        
    def setMode(self, mode:str)->None:
        self.is_mode_changed = True
        self.control_mode = mode
        print("Current mode (changed) : "+self.control_mode)
        
    
    def compute(self)->None:
        if(self.is_mode_changed):
            self.is_mode_changed = False
            self.control_start_time = self.play_time
            self.pose_init = self.pose
            self.tick_init = self.tick
        
        if self.control_mode == "init":
            self.husky.set_world_pose(position=np.array([0.,0.,0.0780]),
                                      orientation=np.array([1.,0.,0.,0.]))
            self.joint_vel_desired = np.zeros(4)
            
        elif self.control_mode == "Tracking":
            self.Tracking(self.getTrajData("/circle.txt"))
            
        self.printState()
        self.play_time = self.world.current_time
        self.tick = self.world.current_time_step_index
        
    def getPose(self)->np.array:
        return self.pose
    
    def getVel(self)->np.array:
        return self.v      
    
    def write(self)->None:
        self.husky.set_joint_velocities(self.joint_vel_desired)
    
    def record(self, file_name_index:int, duration:float):
        if self.play_time < self.control_start_time + duration + 1.0:
            data = str(self.pose)[1:-1] + " " + str(self.v)[1:-1]
            globals()["self.file_"+str(file_name_index)].write(data + "\n" )
            
    def closeFile(self)->None:
        for i in range(len(self.file_names)):
            globals()['self.file_'+str(i)].close()