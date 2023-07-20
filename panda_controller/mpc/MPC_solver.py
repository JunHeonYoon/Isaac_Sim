import numpy as np
from math_type_define.dyros_math import getPhi
from scipy.linalg import block_diag
from casadi import * 
import numpy as np
import numpy.matlib as ml
from scipy import linalg
import math
import time

import os
import osqp
from scipy import sparse


DOF = 7

python_file_path= os.path.dirname(os.path.abspath(__file__))

# Lee, Jaemin, et al. 
# "Real-Time Model Predictive Control for Industrial Manipulators with Singularity-Tolerant Hierarchical Task Control." 
# 2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2023.
class MPCsolver:
    def __init__(self, hz:float, receding_horizon:int) -> None:
        self.hz = hz
        self.N = receding_horizon
        self.setJointLimit(joint_lower_limit=np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973]),
                           joint_upper_limit=np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973]))
        self.setJointVelocityLimit(vel_min=-np.array([2.1750,2.1750,2.1750,2.1750,2.61,2.61,2.61]),
                                   vel_max=np.array([2.1750,2.1750,2.1750,2.1750,2.61,2.61,2.61]))
        self.setWeightMatrix(x_error_weight=np.diag(10*np.ones(6)),
                             qd_weight=np.diag(0.0001*np.ones(7)),
                             qdd_weight=np.diag(0.*np.ones(7)))
        self.q_set_warm_start = np.zeros((DOF*self.N, 1))

    def setJointLimit(self, joint_lower_limit:np.array, joint_upper_limit:np.array):
        self.jlb = joint_lower_limit.reshape(DOF,1)
        self.jub = joint_upper_limit.reshape(DOF,1)

    def setJointVelocityLimit(self, vel_min:np.array, vel_max:np.array):
        self.jvel_min = vel_min.reshape(DOF,1)
        self.jvel_max = vel_max.reshape(DOF,1)
        
    def setWeightMatrix(self, x_error_weight:np.array, qd_weight:np.array, qdd_weight:np.array):
        self.Qe = x_error_weight
        self.Qd = qd_weight
        self.Qa = qdd_weight
        
    def setCurrentStates(self,joints:np.array):
        self.q = joints.reshape(DOF,1)
        self.q_set_warm_start[:DOF] = self.q

    def setDesiredTraectory(self, total_q_desired_set:np.array, total_j_desired_set:np.array):
        self.total_q_desired_set = total_q_desired_set
        self.total_j_desired_set = total_j_desired_set
  
    def formulateOCP(self, current_idx:int, q_before:np.array, q_bbefore:np.array):
        q_before = q_before.reshape(DOF,1)
        q_bbefore = q_bbefore.reshape(DOF,1)
        self.current_idx = current_idx

        if self.current_idx + self.N > len(self.total_q_desired_set) - 1:
            if self.current_idx >= len(self.total_q_desired_set) - 1:
                return 0
            else:
                self.N = len(self.total_q_desired_set) - self.current_idx
                self.q_set_warm_start = self.q_set_warm_start[:self.N*DOF]

        # q_dot = Sv * q + v
        Sv = np.kron(np.eye(self.N), np.eye(DOF)) + np.kron(np.eye(self.N, k=-1), -np.eye(DOF))
        # print("Sv: {}".format(Sv.shape))
        v = np.zeros((DOF*self.N, 1))
        v[0:DOF] = -q_before
        # print("v: {}".format(v.shape))

        # q_ddot = Sa * q + a
        Sa = np.kron(np.eye(self.N), np.eye(DOF)) + np.kron(np.eye(self.N, k=-1), -2*np.eye(DOF)) + np.kron(np.eye(self.N, k=-2), np.eye(DOF)) 
        # print("Sa: {}".format(Sa.shape))
        a = np.zeros((DOF*self.N, 1))
        a[0:DOF] = -2 * q_before + q_bbefore
        a[DOF:2*DOF] = q_before
        # print("a: {}".format(a.shape))

        if self.current_idx == 0:
            Sv[0:DOF,0:DOF] = np.zeros((DOF,DOF))
            Sa[0:DOF,0:DOF] = np.zeros((DOF,DOF))
            Sa[DOF:2*DOF,0:DOF] = -np.eye(DOF)
        elif self.current_idx == 1:
            a[0:DOF] = -q_before
        Sv = Sv * self.hz
        v = v * self.hz
        Sa = Sa * self.hz**2
        a = a * self.hz**2

        # Weight matrix term
        Qe = np.kron(np.eye(self.N), self.Qe)
        Qd = np.kron(np.eye(self.N), self.Qd)
        Qa = np.kron(np.eye(self.N), self.Qa)
        # print("Qe: {}".format(Qe.shape))
        # print("Qd: {}".format(Qd.shape))
        # print("Qa: {}".format(Qa.shape))

        # Nominal Jacobian & q term
        J = np.array(linalg.block_diag(*self.total_j_desired_set[self.current_idx:self.current_idx + self.N]))
        q = self.total_q_desired_set[self.current_idx:self.current_idx + self.N].reshape(self.N*DOF,1)
        # print("J: {}".format(J.shape))
        # print("q: {}".format(q.shape))
        # QP parameter
        self.Q = (J.T @ Qe @ J) + (Sv.T @ Qd @ Sv) + (Sa.T @ Qa @ Sa)
        # print("Q: {}".format(self.Q.shape))
        self.p = -(J.T @ Qe @ J @ q) + (Sa.T @ Qa @ a) + (Sv.T @ Qd @ v)
        # print("p: {}".format(self.p.shape))
        self.qd_min = ml.repmat(self.jvel_min, self.N, 1)
        # print("qd_min: {}".format(self.qd_min.shape))
        self.qd_max = ml.repmat(self.jvel_max, self.N, 1)
        # print("qd_max: {}".format(self.qd_max.shape))
        self.q_min = np.concatenate( [self.q, ml.repmat(self.jlb, self.N-1, 1)], axis=0)
        # print("q_min: {}".format(self.q_min.shape))
        self.q_max = np.concatenate( [self.q, ml.repmat(self.jub, self.N-1, 1)], axis=0)
        # print("q_max: {}".format(self.q_max.shape))
        self.v = v
        self.Sv = Sv

    # def solveOCP(self):  
    #     if self.current_idx >= len(self.total_q_desired_set) - 1:
    #         return 0
    #     H = DM(self.Q)
    #     g = DM(self.p)
    #     x_lb = DM(self.q_min)
    #     x_ub = DM(self.q_max)
    #     a_lb = DM(self.qd_min - self.v)
    #     a_ub = DM(self.qd_max - self.v)
    #     A = DM(self.Sv)
    #     x0 = DM(self.q_set_warm_start)
    #     qp = {}
    #     qp['h'] = H.sparsity()
    #     qp['a'] = A.sparsity()
    #     opts = {}
    #     # opts['printLevel'] = 'none'  # For qpoases
    #     opts['osqp'] = {'verbose':False} # For osqp
    #     # opts['warm_start_primal'] = True
        
    #     s = conic('solver', 'osqp', qp, opts)
    #     solver = s(h=H, g=g, a=A, lbx=x_lb, ubx=x_ub, lba=a_lb, uba=a_ub, x0=x0)
    #     sol = np.array(solver['x'])[:,0]
    #     self.opt_q_set = sol.reshape(self.N, DOF)
    #     self.opt_q = sol[DOF:2*DOF]
    #     self.q_set_warm_start = np.concatenate([sol[DOF:], sol[DOF*(self.N-1):]]).reshape(DOF*self.N,1)
    #     # print(self.q_set_warm_start.shape)
        
    def solveOCP(self):  
        if self.current_idx >= len(self.total_q_desired_set) - 1:
            return 0
        P = self.Q
        q = self.p
        x_lb = self.q_min
        x_ub = self.q_max
        a_lb = self.qd_min - self.v
        a_ub = self.qd_max - self.v
        A = self.Sv
        x0 = self.q_set_warm_start

        A = np.concatenate([np.eye(DOF*self.N), A], axis=0)
        l = np.concatenate([x_lb, a_lb], axis=0)
        u = np.concatenate([x_ub, a_ub], axis=0)

        P = sparse.csc_matrix(P)
        A = sparse.csc_matrix(A)

        # print("P: {}".format(P.shape))
        # print("q: {}".format(q.shape))
        # print("A: {}".format(A.shape))
        # print("l: {}".format(l.shape))
        # print("u: {}".format(u.shape))

        # if self.current_idx == 0:
        self.prob = osqp.OSQP()
        self.prob.setup(P, q, A, l, u, verbose = False)
            
        # else:
        #     print("iter:{}".format(self.current_idx))
        #     self.prob.update(q=q, l=l, u=u, Px=P, Ax=A)
        self.prob.warm_start(x=x0)

        res = self.prob.solve()
        if res.info.status != 'solved':
            raise ValueError('OSQP did not solve the problem!')
        sol = np.array(res.x)
        self.opt_q_set = sol.reshape(self.N, DOF)
        self.opt_q = sol[DOF:2*DOF]
        self.q_set_warm_start = np.concatenate([sol[DOF:], sol[DOF*(self.N-1):]]).reshape(DOF*self.N,1)
        # print(self.q_set_warm_start.shape)


    def getOptimalJoint(self)->np.array:
        # print("opt: {}".format(self.opt_q))
        return self.opt_q
        
        
        