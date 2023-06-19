DOF = 7
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation 
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core import World
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.franka.franka import Franka
from omni.isaac.manipulators import SingleManipulator
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.objects import VisualCuboid, VisualSphere
import lula
import math_type_define.dyros_math as DyrosMath
import math
import ik.IK_solver as IK_solver

np.set_printoptions(precision=3, suppress=True)

def rot2quat(rotation):
    r=Rotation.from_matrix(rotation)
    return r.as_quat()
def quat2rot(quat):
    r=Rotation.from_quat(quat)
    return r.as_matrix()

class ArmController:
    def __init__(self, hz: float, world: World, pkg_path: str)->None: 
        self.hz = hz
        self.world = world
        self.tick = 0
        self.play_time = 0.0
        self.control_mode = "none"
        self.obs_mode = "Stop"
        self.tar_mode = "Stop"
        self.is_mode_changed = False
        self.is_obsmode_changed = False
        self.is_tarmode_changed = False
        self.pkg_path = pkg_path
        self.file_names = ["log", "linear", "RBF"]
        self.initDimension()
        self.initModel()
        self.initObs(0.05)
        self.initFile()
        self.ik = IK_solver.IKsolver(obs_radius=0.05, hz=self.hz, barrier_func="log")
             
    def initDimension(self)->None:
        self.q_init = np.zeros(DOF)
        self.q = np.zeros(DOF)
        self.q_desired = np.zeros(DOF)
        self.qd_init = np.zeros(DOF)
        self.qd = np.zeros(DOF)
        self.torque = np.zeros(DOF)
        self.torque_desired = np.zeros(DOF)
        self.j = np.zeros((6, DOF))
        self.j_qd = np.zeros((6,DOF))
        self.pose_init = np.zeros((4,4))
        self.pose = np.zeros((4,4))
        self.pose_desired = np.zeros((4,4))
        self.pose_qd = np.zeros((4,4))
        # self.gripper_pose_init = np.zeros(2)
        # self.gripper_pose_desired = np.zeros(2)
        # self.gripper_pose = np.zeros(2)
        self.action = ArticulationAction()
        self.event_index = 0
        # self.gripper_mode = False
        
    def initModel(self)->None:
        model_path = self.pkg_path + "/model"       
        self.franka = self.world.scene.add(SingleManipulator(prim_path="/World/panda", 
                                                             name="franka_robot",
                                                             end_effector_prim_name="panda_hand",
                                                            #  gripper=self.gripper
                                                             ))
        
        self.franka_controller = self.franka.get_articulation_controller()
        self.kinematic = lula.RobotDescription.kinematics(lula.load_robot(robot_description_file=model_path + "/panda_robot_description.yaml",
                                                             robot_urdf=model_path + "/panda_arm_hand_wo_gripper.urdf"))
        
        # Target pose cube
        self.target_cube = VisualCuboid(prim_path="/Target_cube",
                                        position=[0.5, 0.5, 0.5],
                                        # orientation=[1, 0, 0, 0],
                                        scale=np.array([.05,.1,.05]),
                                        color=np.array([0,1,0]))
      
    def initObs(self, radius:float)->None:
        self.obstacle = VisualSphere(prim_path="/Obstacle",
                                     position=[0.5, 0, 0.45],
                                     orientation=[0, 0,0, 1],
                                     radius=radius,
                                     color=np.array([1,0,0]))
        self.obstacle.set_collision_enabled(False)
        
    def initFile(self)->None:
        for i in range(len(self.file_names)):
            file_path = self.pkg_path + "/data/" + self.file_names[i] + ".txt"
            globals()['self.file_'+str(i)] = open(file_path, "w")
    
    def printState(self)->None:
        if( self.world.current_time_step_index % 50 == 0 ):
            print("---------------------------------")
            print("Time : ")
            print(round(self.play_time, 2))
            print("Joint(q) :")
            print(self.q)   
            print("\nposition : ")
            print(self.pose[:3, 3].T)
            print("\norientation : ")
            print(self.pose[:3,:3])
            print("\njacobian : ")
            print(self.j)
            print("\ntorque : ")
            print(self.torque)        
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
        
    def CLIK(self, target_pose: np.array = None, duration: float = None, traj_data: np.array = np.zeros(1), traj_type: str = None)->None:
        x_desired = np.zeros(3)  # position
        rotation_desired = np.zeros((3,3)) # rotation
        xd_desired = np.zeros(6) # v, w
        
        # --------------------------------- Cubic Spline -----------------------------------------------
        if np.all(traj_data == 0):
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
        
        # ---------------------------------- Traj from .txt -------------------------------------------
        else:
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
        
        x_error = np.zeros(6)
        x_error[:3] = x_desired - self.pose_qd[:3,3]
        x_error[3:] = DyrosMath.getPhi(self.pose_qd[:3,:3], rotation_desired)
        
        kp = np.zeros((6,6))
        np.fill_diagonal(kp, val=[50, 50, 50, 10, 10, 10])
        
        pseudo_j_inv = np.dot( self.j_qd.T, np.linalg.inv( np.dot(self.j_qd, self.j_qd.T) ) )
        qd_desired = np.dot(pseudo_j_inv, xd_desired + np.dot(kp, x_error) )
        self.q_desired = self.q_desired + qd_desired/self.hz
        
        # target_gripper = np.array([-0.005, -0.005])
        # self.setGripperPosition(target_gripper, 1)
        
        if traj_type == "CLIK Circle":
            self.record(0, traj_data.shape[0]/self.hz)
        elif traj_type == "CLIK Square":
            self.record(1, traj_data.shape[0]/self.hz)
        elif traj_type == "CLIK Eight":
            self.record(2, traj_data.shape[0]/self.hz)
  
    def NSDF(self):
        # self.ik.setCurrentStates(self.pose, self.q, self.j)
        self.ik.setCurrentStates(self.pose, self.q_desired, self.j)

        
        cube_position, cube_quat = self.target_cube.get_world_pose()
        cube_rotation = quat2rot(cube_quat)
        desired_pose = np.diag(np.ones(4))
        desired_pose[:3, :3] = cube_rotation
        desired_pose[:3, 3] = cube_position
        self.ik.setDesiredPose(desired_pose)
        
        obs_posi,_ = self.obstacle.get_world_pose()
        self.ik.setObsPosition(obs_posi)
        
        self.ik.solveIK()
        qdel_desired = self.ik.getJointDisplacement()
        self.q_desired = self.q_desired + qdel_desired
        
    def ObstacleMove(self, velocity:float = 1.0, length:float = 0.2, dir:np.array = np.array([0, 0, 1]))->None:
        if abs(np.linalg.norm(self.obs_posi_init - self.obs_posi) - length) < velocity/self.hz:
            self.obs_dir_sign = self.obs_dir_sign * -1
        del_posi = dir*self.obs_dir_sign*velocity/self.hz
        self.obstacle.set_local_pose(translation=self.obs_posi + del_posi)

    def TargetMove(self, velocity:float = 1.0, dir:np.array = np.array([1,0,0]))->None:
        del_posi = dir*velocity/self.hz
        self.target_cube.set_local_pose(translation=self.tar_posi + del_posi)
        self.record(0,10)
  
    def UpdateData(self)->None:
        self.q = self.franka.get_joint_positions()[:7]
        self.qd = self.franka.get_joint_velocities()[:7]
        self.torque = self.franka.get_applied_joint_efforts()[:7]
        # self.gripper_pose = self.franka.gripper.get_joint_positions()
                 
    def initPosition(self)->None:
        self.q_init = self.q
        self.q_desired = self.q_init
        # self.gripper_pose_init = self.gripper_pose
        # self.gripper_pose_desired = self.gripper_pose_init
          
    def setMode(self, mode:str)->None:
        self.is_mode_changed = True
        self.control_mode = mode
        print("Current mode (changed) : "+self.control_mode)

    def setObsMode(self, mode:str)->None:
        self.is_obsmode_changed = True
        self.obs_mode = mode
        print("Obstacle condition (changed) : "+self.obs_mode)

    def setTarMode(self, mode:str)->None:
        self.is_tarmode_changed = True
        self.tar_mode = mode
        print("Target condition (changed) : "+self.tar_mode)
        
    def compute(self)->None:
        self.q = self.franka.get_joint_positions()[:7]
        self.j = self.kinematic.jacobian(cspace_position=self.q.astype(np.float64), frame="panda_hand")
        self.j_qd = self.kinematic.jacobian(cspace_position=self.q_desired.astype(np.float64), frame="panda_hand")
        self.pose = self.kinematic.pose(cspace_position=self.q.astype(np.float64), frame="panda_hand").matrix()
        self.pose_qd = self.kinematic.pose(cspace_position=self.q_desired.astype(np.float64), frame="panda_hand").matrix()
        self.vel = np.dot(self.j, self.qd) 
        self.obs_posi,_ = self.obstacle.get_local_pose()
        self.tar_posi,_ = self.target_cube.get_local_pose()
        
        if(self.is_mode_changed):
            self.is_mode_changed = False
            self.control_start_time = self.play_time
            self.q_init = self.q
            self.qd_init = self.qd
            self.pose_init = self.pose
            self.tick_init = self.tick
        if(self.is_obsmode_changed):
            self.is_obsmode_changed = False
            self.obs_posi_init = self.obs_posi
            self.obs_dir_sign = 1
        if(self.is_tarmode_changed):
            self.control_start_time = self.play_time
            self.is_tarmode_changed = False
            
        
        if(self.control_mode == "joint_ctrl_init"):
            target_q = np.array([0.0, 0.0, 0.0, -math.pi/2, 0.0, math.pi/2, math.pi/4])
            self.moveJointPosition(target_q, 1)
            
        elif(self.control_mode == "CLIK Circle"):
            self.CLIK(traj_data=self.getTrajData("/circle.txt"), traj_type=self.control_mode)
            
        elif(self.control_mode == "CLIK Square"):
            self.CLIK(traj_data=self.getTrajData("/square.txt"), traj_type=self.control_mode)
            
        elif(self.control_mode == "CLIK Eight"):
            self.CLIK(traj_data=self.getTrajData("/eight.txt"), traj_type=self.control_mode)
            
            
        elif(self.control_mode == "collision_avoidance"):
            self.NSDF()
        
        if(self.obs_mode == "Move"):
            self.ObstacleMove(dir=np.array([0,0,1]), velocity=0.05, length=0.05)
        
        if(self.tar_mode == "Move Up"):
            self.TargetMove(dir = np.array([0,0,1]), velocity=0.1)
        elif(self.tar_mode == "Move Down"):
            self.TargetMove(dir = np.array([0,0,-1]), velocity=0.1)
        elif(self.tar_mode == "Move Left"):
            self.TargetMove(dir = np.array([0,-1,0]), velocity=0.1)
        elif(self.tar_mode == "Move Right"):
            self.TargetMove(dir = np.array([0,1,0]), velocity=0.1)
        elif(self.tar_mode == "Move Forward"):
            self.TargetMove(dir = np.array([1,0,0]), velocity=0.1)
        elif(self.tar_mode == "Move Back"):
            self.TargetMove(dir = np.array([-1,0,0]), velocity=0.1)
        
        self.printState()
        self.play_time = self.world.current_time
        self.tick = self.world.current_time_step_index
        
    def getPosition(self)->np.array:
        return self.q
    
    def write(self)->None:
        self.action = ArticulationAction(joint_positions=self.q_desired)
        self.franka_controller.apply_action(self.action)

    def getObsMode(self)->str:
        return self.obs_mode
    
    def getTarMode(self)->str:
        return self.tar_mode

    def record(self, file_name_index:int, duration:float):
        if self.play_time < self.control_start_time + duration:
            data = str(round(self.play_time-self.control_start_time,2))+ " " + str(self.obs_posi)[1:-1] + " " + str(self.tar_posi)[1:-1] + " " + str(self.pose[:3,3])[1:-1]
            data = data + " "+ str(self.pose[:3,0])[1:-1] + str(self.pose[:3,1])[1:-1] + str(self.pose[:3,2])[1:-1]
            data  = data + " " + str(self.vel[:3])[1:-1]
            globals()["self.file_"+str(file_name_index)].write(data + "\n" )
    
    def closeFile(self)->None:
        for i in range(len(self.file_names)):
            globals()['self.file_'+str(i)].close()