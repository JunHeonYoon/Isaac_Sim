import numpy as np
import pandas as pd 
import os

import ForwadKinematics as FK

class TrajectoryPlanner:
    def __init__(self, traj_type:str, Hz:float) -> None:
        self.ReadTrajList()
        if traj_type not in self.traj_list:
            print("There is no trajectory type name: {}".format(traj_type))
            self.__del__()
        self.traj_type = traj_type
        self.dt = 1/Hz
        self.ReadTrajData()

    def __del__(self) -> None:
        print("Trajectory-Planner terminated.")

    def ReadTrajList(self) -> None:
        folder_path = os.path.dirname(os.path.abspath(__file__)) + "/data"
        self.traj_list = []

        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path) and file_name.endswith('.txt'):
                traj_type = file_name.replace('.txt', '')
                self.traj_list.append(traj_type)

    def ReadTrajData(self) -> None:
        folder_path = os.path.dirname(os.path.abspath(__file__)) + "/data/"
        traj_data = pd.read_table(folder_path+self.traj_type+".txt", sep=" ", header=None)
        traj_data = traj_data.to_numpy()
        self.total_time = len(traj_data)

        x_traj = np.zeros(traj_data.shape[0])
        y_traj = traj_data[:, 1]
        z_traj = traj_data[:, 2]

        vx_traj = np.zeros(traj_data.shape[0])
        vy_traj = traj_data[:, 3]
        vz_traj = traj_data[:, 4]

        # x_desired = [x_traj, y_traj, z_traj]
        # xd_desired = [vx_traj, vy_traj, vz_traj]
        self.x_desired = np.column_stack((x_traj, y_traj))
        self.x_desired = np.column_stack((self.x_desired, z_traj))
        self.xd_desired = np.column_stack((vx_traj, vy_traj))
        self.xd_desired = np.column_stack((self.xd_desired, vz_traj))

    def setInitialJoint(self, init_joint:np.array) -> None:
        self.init_joint = init_joint
    
    def calculateDesiredJoint(self) -> None:
        self.q_desired = self.init_joint
        self.j_desired = np.array([])
        kp = np.zeros((6,6))
        np.fill_diagonal(kp, val=[50, 50, 50, 10, 10, 10])
        
        # CLIK
        for x_desired, xd_desired in zip(self.x_desired, self.xd_desired):
            x_desired = np.append(x_desired.reshape(3,1), np.zeros([3,1]), axis=0) # NO rotation
            xd_desired = np.append(xd_desired.reshape(3,1), np.zeros([3,1]), axis=0) # NO rotation
            x = FK.calculateFK(self.q_desired[-1,:]) 
            x = np.append(x[:3,3].reshape(3,1), np.zeros([3,1]), axis=0) # NO rotation
            J = FK.calculateJac(self.q_desired[-1,:])
            if self.j_desired.size == 0:
                self.j_desired = J.reshape(1,6,7)
            else:
                self.j_desired = np.append(self.j_desired, J.reshape(1,6,7), axis=0)
            J_pinv = J.T @ np.linalg.inv(J @ J.T)
            qd_desired = J_pinv @ (xd_desired + kp @ (x_desired - x))
            q_desired = qd_desired[-1,:] + qd_desired * self.dt
            self.q_desired = np.concatenate([self.q_desired, q_desired.T], axis=0)
        self.q_desired = self.q_desired[:-1,:]
