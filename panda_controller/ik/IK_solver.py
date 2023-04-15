import numpy as np
from math_type_define.dyros_math import getPhi
from scipy.linalg import block_diag
from casadi import * 
import torch
import torch.nn.functional as F
import numpy as np
import time
import yaml

import os
import matplotlib.pyplot as plt
from ik.sdf.robot_sdf import RobotSdfCollisionNet

os.environ['QPOASES_VERBOSE'] = 'NO'



DOF = 7
NUM_LINKS = 9

class IKsolver:
    def __init__(self, obs_radius:float) -> None:
        self.r = obs_radius
        self.setNNModel()
        self.setJointLimit(joint_lower_limit=np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973]),
                           joint_upper_limit=np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973]))
        self.setWeightMatrix(slack_weight=np.diag(1*np.ones(6)),
                             damping_weight=np.diag(10*np.ones(7)))
    
    def setNNModel(self)->None:
        device = torch.device('cpu', 0)
        tensor_args = {'device': device, 'dtype': torch.float32}
        
        dof = 10 # 7 for robot joints, 3 for object points
        s = 256
        n_layers = 5
        skips = []
        if skips == []:
            n_layers-=1
        self.nn_model = RobotSdfCollisionNet(in_channels=dof, out_channels=9, layers=[s] * n_layers, skips=skips)
        self.nn_model.load_weights('/home/yoonjunheon/Isaac_Sim/panda_controller/ik/NNmodel/sdf_256x5_mesh_50000.pt', tensor_args)
        self.nn_model.model.to(**tensor_args)

    def setJointLimit(self, joint_lower_limit:np.array, joint_upper_limit:np.array):
        self.jlb = joint_lower_limit
        self.jub = joint_upper_limit
        
    def setWeightMatrix(self, slack_weight:np.array, damping_weight:np.array):
        self.Q = slack_weight
        self.R = damping_weight
        
    def setCurrentStates(self, ee_pose:np.array, joints:np.array, jacobian:np.array):
        self.x = ee_pose
        self.j = jacobian
        self.q = joints
        
    def setDesiredPose(self, desired_pose:np.array):
        self.x_desired = desired_pose
    
    def setObsPosition(self, obs_position:np.array):
        self.obs_posi = obs_position
        
    def getSDF(self, joint:np.array, obs_position:np.array):
        x = torch.from_numpy(np.array([np.concatenate([joint, obs_position], axis=0, dtype=np.float32)]))
        y_pred, j_pred, _ = self.nn_model.compute_signed_distance_wgrad(x)
        y_pred = y_pred.cpu().detach().numpy()[0]
        j_pred = j_pred.cpu().detach().numpy()[0, 0:7]
        return y_pred, j_pred 
    
    def solveIK(self):
        gamma, j_gamma = self.getSDF(self.q, self.obs_posi)
        x_error = np.concatenate([self.x_desired[:3,3]-self.x[:3,3], getPhi(self.x[:3,:3], self.x_desired[:3,:3])], axis=0)
        gain = np.diag([5,5,5,1,1,1])
        x_error = np.matmul(gain, x_error)
        
        H = DM(block_diag(2*self.Q, 2*self.R))
        g = DM.zeros(13)
        x_lb = DM(np.concatenate([-np.inf*np.ones(6), self.jlb-self.q],axis=0))
        x_ub = DM(np.concatenate([np.inf*np.ones(6), self.jub-self.q],axis=0))
        if np.min(gamma - self.r*100) > 1e-10:
            a_ub = DM(np.concatenate([x_error, np.log(gamma-self.r*100) - self.r*100*0.2*np.ones(gamma.shape)],axis=0))
            a_lb = DM(np.concatenate([x_error, -np.inf*np.ones(9)],axis=0))
            A = DM(np.block([[-np.identity(6), self.j], [np.zeros((9,6)), -j_gamma.T]]))
        else:
            print("\n\n\n\n!!coliision dectation!!\n\n\n")
            a_ub = DM(x_error)
            a_lb = DM(x_error)
            A = DM(np.block([-np.identity(6), self.j]))
        qp = {}
        qp['h'] = H.sparsity()
        qp['a'] = A.sparsity()
        opts = {}
        opts['printLevel'] = 'none'
        
        s = conic('solver', 'qpoases', qp, opts)
        solver = s(h=H, g=g, a=A, lbx=x_lb, ubx=x_ub, lba=a_lb, uba=a_ub)
        sol = np.array(solver['x'])[:,0]
        self.del_q = sol[6:]
        
        # print(np.matmul(-j_gamma.T, sol[6:]))
        # print(np.log(gamma-self.r*100))
        # print("\n\n")
        

        
    def getJointDisplacement(self)->np.array:
        return self.del_q
        
        
        
    