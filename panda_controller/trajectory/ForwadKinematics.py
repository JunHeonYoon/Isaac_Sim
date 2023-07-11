import math
import numpy as np

def dh_params(joint_var):
    M_PI = math.pi
    # Create DH parameters (data given by maker franka-emika)
    dh = np.array([[ 0,      0,        0.333,   joint_var[0]],
        [-M_PI/2,   0,        0,       joint_var[1]],
        [ M_PI/2,   0,        0.316,   joint_var[2]],
        [ M_PI/2,   0.0825,   0,       joint_var[3]],
        [-M_PI/2,  -0.0825,   0.384,   joint_var[4]],
        [ M_PI/2,   0,        0,       joint_var[5]],
        [ M_PI/2,   0.088,    0.107,   joint_var[6]-M_PI/4]])
    
    return dh

def TF_matrix(i,dh):
    # Define Transformation matrix based on DH params
    alpha = dh[i,0]
    a = dh[i,1]
    d = dh[i,2]
    q = dh[i,3]
    
    TF = np.array([[math.cos(q),-math.sin(q), 0, a],
                [math.sin(q)*math.cos(alpha), math.cos(q)*math.cos(alpha), -math.sin(alpha), -math.sin(alpha)*d],
                [math.sin(q)*math.sin(alpha), math.cos(q)*math.sin(alpha),  math.cos(alpha),  math.cos(alpha)*d],
                [   0,  0,  0,  1]])
    return TF

def calculateFK(q:np.array) -> np.array:
    dh_parameters = dh_params(q)

    T = np.eye(4)
    for i in range(7):
        T = T @ TF_matrix(i, dh_parameters)
    return T

def calculateJac(q:np.array) -> np.array:
    dh_parameters = dh_params(q)
    T_EE = calculateFK(q)
    J = np.zeros((6, 7))
    T = np.identity(4)
    for i in range(7):
        T = T @ TF_matrix(i, dh_parameters)

        p = T_EE[:3, 3] - T[:3, 3]
        z = T[:3, 2]

        J[:3, i] = np.cross(z, p)
        J[3:, i] = z

    return J