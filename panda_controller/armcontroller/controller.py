DOF = 7
import numpy as np
from scipy.spatial.transform import Rotation 
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core import World
from omni.isaac.core.utils.types import ArticulationAction
# from omni.isaac.franka.franka import Franka
from omni.isaac.manipulators import SingleManipulator
import lula
import math_type_define.dyros_math as DyrosMath
import math
import mpc.MPC_solver as MPC_solver
import trajectory.TrajectoryPlanner as TrajectoryPlanner

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
        self.is_mode_changed = False
        self.pkg_path = pkg_path
        self.file_names = ["debug"]
        self.initDimension()
        self.initModel()
        self.initFile()
        self.TrajectoyPlanner = TrajectoryPlanner.TrajectoryPlanner(traj_type="circle", Hz=self.hz)
        self.MPCcontroller = MPC_solver.MPCsolver(self.hz, receding_horizon=10)
             
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
        self.action = ArticulationAction()
        
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
        
    def CLIK(self, target_pose: np.array = None, duration: float = None)->None:
        x_desired = np.zeros(3)  # position
        rotation_desired = np.zeros((3,3)) # rotation
        xd_desired = np.zeros(6) # v, w
        
        # --------------------------------- Cubic Spline -----------------------------------------------
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
  
    def UpdateData(self)->None:
        self.q = self.franka.get_joint_positions()[:7]
        self.qd = self.franka.get_joint_velocities()[:7]
        self.torque = self.franka.get_applied_joint_efforts()[:7]
                 
    def initPosition(self)->None:
        self.q_init = self.q
        self.q_desired = self.q_init
          
    def setMode(self, mode:str)->None:
        self.is_mode_changed = True
        self.control_mode = mode
        print("Current mode (changed) : "+self.control_mode)

    def compute(self)->None:
        self.q = self.franka.get_joint_positions()[:7]
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
            self.tick_init = self.tick
            
        
        if(self.control_mode == "joint_ctrl_init"):
            target_q = np.array([0.0, 0.0, 0.0, -math.pi/2, 0.0, math.pi/2, math.pi/4])
            self.moveJointPosition(target_q, 1)
        elif(self.control_mode == "MPC_tracking"):
            if self.tick - self.tick_init == 0:
                self.TrajectoyPlanner.setInitialJoint(self.q_init)
                self.TrajectoyPlanner.calculateDesiredJoint()
                self.MPCcontroller.setDesiredTraectory(self.TrajectoyPlanner.q_desired, self.TrajectoyPlanner.j_desired)
                self.q_before = np.zeros(DOF)
                self.q_bbefore = np.zeros(DOF)
            
            self.MPCcontroller.setCurrentStates(self.q)
            self.MPCcontroller.formulateOCP(self.tick - self.tick_init, self.q_before, self.q_bbefore)
            self.MPCcontroller.solveOCP()
            self.q_desired = self.MPCcontroller.getOptimalJoint()
            self.q_before = self.q
            self.q_bbefore = self.q_before
        
        self.printState()
        self.play_time = self.world.current_time
        self.tick = self.world.current_time_step_index
        
    def getPosition(self)->np.array:
        return self.q
    
    def write(self)->None:
        self.action = ArticulationAction(joint_positions=self.q_desired)
        self.franka_controller.apply_action(self.action)
    
    def closeFile(self)->None:
        for i in range(len(self.file_names)):
            globals()['self.file_'+str(i)].close()