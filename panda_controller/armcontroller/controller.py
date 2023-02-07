DOF = 7
import numpy as np
import pandas as pd
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core import World
from omni.isaac.franka.franka import Franka
import lula
import math_type_define.dyros_math as DyrosMath
import math

np.set_printoptions(precision=3, suppress=True)

class ArmController:
    def __init__(self, hz: float, world: World, pkg_path: str)->None: 
        self.hz = hz
        self.world = world
        self.tick = 0
        self.play_time = 0.0
        self.control_mode = "none"
        self.is_mode_changed = False
        self.pkg_path = pkg_path
        self.initDimension()
        self.initModel()
        
        
        
        
    def initDimension(self)->None:
        self.q_init = np.zeros(DOF)
        self.q = np.zeros(DOF)
        self.q_desired = np.zeros(DOF)
        self.qd_init = np.zeros(DOF)
        self.qd = np.zeros(DOF)
        self.torque = np.zeros(DOF)
        self.j = np.zeros((6, DOF))
        self.j_qd = np.zeros((6,DOF))
        self.pose_init = np.zeros((4,4))
        self.pose = np.zeros((4,4))
        self.pose_qd = np.zeros((4,4))
        self.gripper_init = np.zeros(2)
        self.gripper_desired = np.zeros(2)
        self.gripper = np.zeros(2)
        
        
    
    def initModel(self)->None:
        model_path = self.pkg_path + "/model"
        self.franka = self.world.scene.add(Franka(prim_path="/World/panda", name="franka_robot",
                                                  end_effector_prim_name="panda_hand",
                                                  gripper_dof_names = ["panda_finger_joint1", "panda_finger_joint2"],
                                                  gripper_open_position = np.array([0.04, 0.04]),
                                                  gripper_closed_position = np.array([-0.005, -0.005]),))
        self.kinematic = lula.RobotDescription.kinematics(lula.load_robot(robot_description_file=model_path + "/panda_robot_description.yaml",
                                                             robot_urdf=model_path + "/panda_arm_hand.urdf"))
        joints_default_positions = np.zeros(9)
        joints_default_positions[3] = -math.pi/2
        joints_default_positions[5] = math.pi/2
        joints_default_positions[6] = math.pi/4
        joints_default_positions[7] = 0.
        joints_default_positions[8] = 0.
        self.franka.set_joints_default_state(positions=joints_default_positions)
        
        
        
    def printState(self)->None:
        if( self.world.current_time_step_index % 50 == 0 ):
            print("---------------------------------")
            print("Time : ")
            print(round(self.play_time, 2))
            print("position : ")
            print(self.pose[:3, 3].T)
            print("orientation : ")
            print(self.pose[:3,:3])
            print("jacobian")
            print(self.j)
            print("---------------------------------")
            print("\n\n")
            
            
            
    def moveJointPosition(self, target_q: np.array, duration: float)->None:
        self.q_desired = DyrosMath.cubicVector(self.play_time, self.control_start_time, self.control_start_time+duration, 
                                               self.q_init, target_q, np.zeros(7), np.zeros(7))
                
        
        
    def getTrajData(self, file_name:str)->np.array:
        traj_data = pd.read_table(self.pkg_path+"/trajectory"+file_name, sep=" ", header=None)
        traj_data = traj_data.to_numpy()

        x_traj = np.zeros(traj_data.shape[0])
        y_traj = traj_data[:, 1]
        z_traj = traj_data[:, 2]

        vx_traj = np.zeros(traj_data.shape[0])
        vy_traj = traj_data[:, 3]
        vz_traj = traj_data[:, 4]

        traj = np.column_stack((x_traj, y_traj))
        traj = np.column_stack((traj, z_traj))
        traj = np.column_stack((traj, vx_traj))
        traj = np.column_stack((traj, vy_traj))
        traj = np.column_stack((traj, vz_traj))
        return traj
        
        
        
    def CLIK(self, target_pose: np.array = None, duration: float = None, traj_data: np.array = None)->None:
        x_desired = np.zeros(3)  # position
        rotation_desired = np.zeros((3,3)) # rotation
        xd_desired = np.zeros(6) # v, w
        
        # ---------------------------------- Traj from .txt -------------------------------------------
        if traj_data.any():
            index = self.tick - self.tick_init
            
            if(index < 0):
                x_desired = self.pose_init[:3,3]
                rotation_desired = self.pose_init[:3,:3]
                xd_desired = np.zeros(6)
            elif(index >= traj_data.shape[0]):
                x_desired = np.array([traj_data[-1,0]+self.pose_init[0, 3],
                                      traj_data[-1,1]+self.pose_init[1, 3], 
                                      traj_data[-1,2]+self.pose_init[2, 3]])
                rotation_desired = self.pose_init[:3,:3]
                xd_desired = np.zeros(6)
            else:
                x_desired = np.array([traj_data[index,0]+self.pose_init[0, 3],
                                      traj_data[index,1]+self.pose_init[1, 3], 
                                      traj_data[index,2]+self.pose_init[2, 3]])
                rotation_desired = self.pose_init[:3,:3]
                xd_desired = np.array([traj_data[index,3], traj_data[index,4], traj_data[index,5], 0, 0, 0])
        # -----------------------------------------------------------------------------------------------
        
        # --------------------------------- Cubic Spline -----------------------------------------------
        else:
            for i in range(0,3):
                x_desired[i] = DyrosMath.cubic(self.play_time, self.control_start_time, self.control_start_time+duration,
                                               self.pose_init[i,3], target_pose[i,3], 0, 0)
                xd_desired[i] = DyrosMath.cubicDot(self.play_time, self.control_start_time, self.control_start_time+duration,
                                                   self.pose_init[i,3], target_pose[i,3], 0, 0)
            rotation_desired = DyrosMath.rotationCubic(self.play_time, self.control_start_time, self.control_start_time+duration,
                                                       self.pose_init[:3,:3], target_pose[:3,:3])
            xd_desired[3:] = DyrosMath.rotationCubicDot(self.play_time, self.control_start_time, self.control_start_time+duration,
                                                        np.zeros(3), np.zeros(3), self.pose_init[:3,:3], target_pose[:3,:3])
        # -----------------------------------------------------------------------------------------------
        
        x_error = np.zeros(6)
        x_error[:3] = x_desired - self.pose_qd[:3,3]
        x_error[3:] = DyrosMath.getPhi(self.pose_qd[:3,:3], rotation_desired)
        
        kp = np.zeros((6,6))
        np.fill_diagonal(kp, val=[50, 50, 50, 10, 10, 10])
        
        pseudo_j_inv = np.dot( self.j_qd.T, np.linalg.inv( np.dot(self.j_qd, self.j_qd.T) ) )
        qd_desired = np.dot(pseudo_j_inv, xd_desired + np.dot(kp, x_error) )
        self.q_desired = self.q_desired + qd_desired/self.hz
        
    def setGripperPosition(self, target_position:np.array, duration:float)->None:
        self.gripper_desired = DyrosMath.cubicVector(self.play_time, self.control_start_time, self.control_start_time+duration, 
                                                     self.gripper_init, target_position, np.zeros(2), np.zeros(2))
        
        
        
        
    def UpdateData(self)->None:
        self.q = self.franka.get_joint_positions()[:7]
        self.qd = self.franka.get_joint_velocities()[:7]
        self.torque = self.franka.get_applied_joint_efforts()[:7]
        self.gripper = self.franka.gripper.get_joint_positions()
            
            
            
    def initPosition(self)->None:
        self.q_init = self.q
        self.q_desired = self.q_init
        self.gripper_init = self.gripper
        self.gripper_desired = self.gripper_init
        
        
        
    def setMode(self, mode:str)->None:
        self.is_mode_changed = True
        self.control_mode = mode
        print("Current mode (changed) : "+self.control_mode)
        
        
    def compute(self)->None:
        self.q = self.franka.get_joint_positions()[:7]
        self.gripper = self.franka.gripper.get_joint_positions()
        self.j = self.kinematic.jacobian(cspace_position=self.q.astype(np.float64), frame="panda_hand")
        self.j_qd = self.kinematic.jacobian(cspace_position=self.q_desired.astype(np.float64), frame="panda_hand")
        self.pose = self.kinematic.pose(cspace_position=self.q.astype(np.float64), frame="panda_hand").matrix()
        self.pose_qd = self.kinematic.pose(cspace_position=self.q_desired.astype(np.float64), frame="panda_hand").matrix()
        self.vel = np.dot(self.j, self.qd) 
        
        if(self.is_mode_changed):
            self.is_mode_changed = False
            self.control_start_time = self.play_time
            self.q_init = self.q
            self.qd_init = self.qd
            self.pose_init = self.pose
            self.gripper_init = self.gripper
            self.tick_init = self.tick
        
        if(self.control_mode == "joint_ctrl_init"):
            target_q = np.array([0.0, 0.0, 0.0, -math.pi/2, 0.0, math.pi/2, math.pi/4])
            target_gripper = np.zeros(2)
            self.moveJointPosition(target_q, 1)
            self.setGripperPosition(target_gripper, 1)
        elif(self.control_mode == "CLIK"):
            # target_pose = np.array([[1,  0,  0, 0.25],
            #                         [0, -1,  0, 0.28],
            #                         [0,  0, -1, 0.65],
            #                         [0,  0,  0, 1]])
            # self.CLIK(target_pose=target_pose, duration=2.0)
            self.CLIK(traj_data=self.getTrajData("/eight.txt"))
        elif(self.control_mode == "gripper_open"):
            target_gripper = np.array([0.04, 0.04])
            self.setGripperPosition(target_gripper, 1)
        elif(self.control_mode == "gripper_close"):
            target_gripper = np.array([-0.005, -0.005])
            self.setGripperPosition(target_gripper, 1)
            
        self.printState()
        self.play_time = self.world.current_time
        self.tick = self.world.current_time_step_index
        
    def getPosition(self)->np.array:
        return self.q
    
    def getGripperPosition(self)->np.array:
        return self.gripper
    
    def write(self)->None:
        self.franka.set_joint_positions(positions=np.concatenate((self.q_desired, self.gripper_desired), axis=0))
            