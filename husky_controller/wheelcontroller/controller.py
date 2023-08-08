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
    def __init__(self, hz: float, world:World, pkg_path:str, is_two_wheel:bool) -> None:
        self.hz = hz
        self.world = world
        self.tick = 0
        self.play_time = 0
        self.control_mode = "none"
        self.is_mode_changed = False
        self.pkg_path = pkg_path
        self.file_names = ["Circle", "Square", "Eight", "VelCommand"]
        self.is_two_wheel = is_two_wheel
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
        if(self.is_two_wheel):  
            self.joint_vel = np.zeros(2)
            self.joint_vel_desired = np.zeros(2)
            self.dummy_joint_vel = np.zeros(4) # For visual 
            self.dummy_joint_vel_desired = np.zeros(4)
        else:
            self.joint_vel = np.zeros(4)
            self.joint_vel_desired = np.zeros(4)
         
    def initModel(self)->None:
        self.husky = self.world.scene.add(Articulation(prim_path="/World/husky", name="husky",))
        
    def initFile(self)->None:
        for i in range(len(self.file_names)):
            file_path = self.pkg_path + "/data/" + self.file_names[i] + ".txt"
            globals()['self.file_'+str(i)] = open(file_path, 'w')
          
    def printState(self)->None:
        if( self.world.current_time_step_index % 50 == 0 ):
            print("----------------------------")
            print("Time[sec]")
            print(round(self.world.current_time, 2))
            print("\npose[m, m, rad] : ")
            print(self.pose_desired)
            print(self.pose)
            print("\nLinear velocity[m/s, m/s, rad/s] : ")
            print(self.v_desired)
            print(self.v)
            print("\nJoint velocity[rad/s] : ")
            print(self.joint_vel_desired)
            print(self.joint_vel)
            print("----------------------------")
            print("\n\n\n")
            
    def Quater2Yaw(self, quaternion:np.array)->float:
        yaw = math.atan2(2*(quaternion[0]*quaternion[3] + quaternion[1]*quaternion[2]), 1 - 2*(quaternion[2]**2 + quaternion[3]**2))
        return yaw
             
    def getTrajData(self, file_name:str)->np.array:
        traj_data = pd.read_table(self.pkg_path+"/trajectory"+file_name, sep=" ", header=None)
        traj = traj_data.to_numpy()
        return traj # x, y, theta (from global frame), vx, vy, w(from base frame)
    
    def Kanayama(self, traj_data: np.array, traj_type: str)->None:
        index = self.tick - self.tick_init
        if index < 0 or index >= traj_data.shape[0]:
            self.v_desired = np.zeros(DOF)
            self.pose_desired = self.pose
        else:
            Kx = 9
            Ky = 9
            Kth = 4.5

            ref_pose = traj_data[index, 0:3] # x, y, theta wrt global frame
            ref_vel  = traj_data[index, 3:6] # vx, vy, w wrt base frame

            Rz = np.array([[ math.cos(self.pose[2]), math.sin(self.pose[2]), 0],
                           [-math.sin(self.pose[2]), math.cos(self.pose[2]), 0],
                           [0,                       0,                      1]])
            
            error_pose = Rz @ (ref_pose - self.pose) # Error pose wrt base frame
            error_pose[2] = math.atan2(math.sin(error_pose[2]), math.cos(error_pose[2])) # Angle error correction
            
            # Feedback velocity command
            self.v_desired[0] = ( ref_vel[0] * math.cos(error_pose[2]) ) + Kx *  error_pose[0]
            self.v_desired[2] = ref_vel[2] + ref_vel[0] * ( Ky * error_pose[1] + Kth * math.sin(error_pose[2]) )

            # Desired pose
            self.pose_desired = np.array([traj_data[index, 0], traj_data[index, 1], traj_data[index, 2]])

        if self.is_two_wheel:
            self.joint_vel_desired = diffcontroller.DifferentialController(wheel_radius=0.1651,
                                                                           wheel_distance=0.5708,
                                                                           linear_velocity=self.v_desired.item(0),
                                                                           angular_velocity=self.v_desired.item(2),
                                                                           max_linear_speed=1.0,
                                                                           max_angular_speed=2.0,
                                                                           is_skid=False,
                                                                           wheel_distance_multiplier=1.0)
            self.dummy_joint_vel_desired = np.concatenate([self.joint_vel_desired ,self.joint_vel_desired], axis=0)
        else:
            self.joint_vel_desired = diffcontroller.DifferentialController(wheel_radius=0.1651,
                                                                           wheel_distance=0.5708,
                                                                           linear_velocity=self.v_desired.item(0),
                                                                           angular_velocity=self.v_desired.item(2),
                                                                           max_linear_speed=1.0,
                                                                           max_angular_speed=2.0,
                                                                           is_skid=True,
                                                                           wheel_distance_multiplier=1.875)
        if traj_type == "Kanayama Circle":
            self.record(0, traj_data.shape[0]/self.hz)
        if traj_type == "Kanayama Square":
            self.record(1, traj_data.shape[0]/self.hz)
        if traj_type == "Kanayama Eight":
            self.record(2, traj_data.shape[0]/self.hz)
    
    def VelControl(self, traj_data: np.array, traj_type: str)->None:
        index = self.tick - self.tick_init
        if index < 0 or index >= traj_data.shape[0]:
            self.v_desired = np.zeros(DOF)
            self.pose_desired = self.pose
        else:
            # Desired velocity
            self.v_desired = traj_data[index, 3:6]

            # Desired pose
            self.pose_desired = np.array([traj_data[index, 0], traj_data[index, 1], traj_data[index, 2]])

        if self.is_two_wheel:
            self.joint_vel_desired = diffcontroller.DifferentialController(wheel_radius=0.1651,
                                                                           wheel_distance=0.5708,
                                                                           linear_velocity=self.v_desired.item(0),
                                                                           angular_velocity=self.v_desired.item(2),
                                                                           max_linear_speed=1.0,
                                                                           max_angular_speed=2.0,
                                                                           is_skid=False,
                                                                           wheel_distance_multiplier=1.0)
            self.dummy_joint_vel_desired = np.concatenate([self.joint_vel_desired ,self.joint_vel_desired], axis=0)
        else:
            self.joint_vel_desired = diffcontroller.DifferentialController(wheel_radius=0.1651,
                                                                           wheel_distance=0.5708,
                                                                           linear_velocity=self.v_desired.item(0),
                                                                           angular_velocity=self.v_desired.item(2),
                                                                           max_linear_speed=1.0,
                                                                           max_angular_speed=2.0,
                                                                           is_skid=True,
                                                                           wheel_distance_multiplier=1.875)
        if traj_type == "Velocity Circle":
            self.record(0, traj_data.shape[0]/self.hz)
        if traj_type == "Velocity Square":
            self.record(1, traj_data.shape[0]/self.hz)
        if traj_type == "Velocity Eight":
            self.record(2, traj_data.shape[0]/self.hz)

    def VelCommand(self, linvel_desired:float, angvel_desired:float, duration:float)->None:
        if self.play_time <= self.control_start_time + duration:
            self.v_desired = np.array([linvel_desired, 0, angvel_desired])
        else:
            self.v_desired = np.zeros(3)
        if self.is_two_wheel:
            self.joint_vel_desired = diffcontroller.DifferentialController(wheel_radius=0.1651,
                                                                           wheel_distance=0.5708,
                                                                           linear_velocity=self.v_desired.item(0),
                                                                           angular_velocity=self.v_desired.item(2),
                                                                           max_linear_speed=1.0,
                                                                           max_angular_speed=2.0,
                                                                           is_skid=False,
                                                                           wheel_distance_multiplier=1.0)
            self.dummy_joint_vel_desired = np.concatenate([self.joint_vel_desired , self.joint_vel_desired], axis=0)
        else:
            self.joint_vel_desired = diffcontroller.DifferentialController(wheel_radius=0.1651,
                                                                           wheel_distance=0.5708,
                                                                           linear_velocity=self.v_desired.item(0),
                                                                           angular_velocity=self.v_desired.item(2),
                                                                           max_linear_speed=1.0,
                                                                           max_angular_speed=2.0,
                                                                           is_skid=True,
                                                                           wheel_distance_multiplier=1.875)
        self.record(3, duration)
        
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
        self.v[2] = angvel[2]
        
        if self.is_two_wheel:
            self.joint_vel = self.husky.get_joint_velocities()[4:]
            self.dummy_joint_vel = self.husky.get_joint_velocities()[:4]
        else:
            self.joint_vel = self.husky.get_joint_velocities()
        
    def initPosition(self)->None:
        self.pose_init = self.pose
        self.pose_desired = self.pose_init
        self.v_desired = self.v
        self.joint_vel_desired = self.joint_vel
        if self.is_two_wheel:
            self.dummy_joint_vel_desired = self.dummy_joint_vel
        
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
            if self.is_two_wheel:
                self.joint_vel_desired = np.zeros(2)
                self.dummy_joint_vel_desired = np.zeros(4)
            else:
                self.joint_vel_desired = np.zeros(4)
            
        elif self.control_mode == "Kanayama Circle":
            self.Kanayama(self.getTrajData("/circle.txt"), self.control_mode)
        elif self.control_mode == "Kanayama Square":
            self.Kanayama(self.getTrajData("/square.txt"), self.control_mode)
        elif self.control_mode == "Kanayama Eight":
            self.Kanayama(self.getTrajData("/eight.txt"), self.control_mode)
        elif self.control_mode == "Velocity Circle":
            self.VelControl(self.getTrajData("/circle.txt"), self.control_mode)
        elif self.control_mode == "Velocity Square":
            self.VelControl(self.getTrajData("/square.txt"), self.control_mode)
        elif self.control_mode == "Velocity Eight":
            self.VelControl(self.getTrajData("/eight.txt"), self.control_mode)
        elif self.control_mode == "Velocity Command":
            linear_vel = 0.25
            angular_vel = 0.125
            duration = 2*2*math.pi / linear_vel
            # duration = 5
            self.VelCommand(linear_vel, angular_vel, duration)
            
        self.printState()
        self.play_time = self.world.current_time
        self.tick = self.world.current_time_step_index
        
    def getPose(self)->np.array:
        return self.pose
    
    def getVel(self)->np.array:
        return self.v      
    
    def write(self)->None:
        if self.is_two_wheel:
            self.husky.set_joint_velocities(velocities=np.concatenate([self.dummy_joint_vel_desired, self.joint_vel_desired], axis=0))
        else:
            self.husky.set_joint_velocities(self.joint_vel_desired)
    
    def record(self, file_name_index:int, duration:float):
        if self.play_time < self.control_start_time + duration + 1.0:
            data = str(self.play_time - self.control_start_time) + " " + str(self.pose*1000)[1:-1] + " " + str(self.v*1000)[1:-1]
            globals()["self.file_"+str(file_name_index)].write(data + "\n" )
            
    def closeFile(self)->None:
        for i in range(len(self.file_names)):
            globals()['self.file_'+str(i)].close()