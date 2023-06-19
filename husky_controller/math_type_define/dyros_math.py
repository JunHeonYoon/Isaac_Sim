import numpy as np
from scipy.linalg import logm, expm

def cubic(time, time_0, time_f, x_0, x_f, xd_0, xd_f):
    x_t = 0
    if (time < time_0):
        x_t = x_0
    elif(time > time_f):
        x_t = x_f
    else:
        elapsed_time = time - time_0
        total_time = time_f - time_0
        total_time2 = total_time * total_time
        total_time3 = total_time * total_time * total_time
        total_x = x_f - x_0
        
        x_t = x_0 + xd_0 * elapsed_time + (3*total_x/total_time2 - 2*xd_0/total_time - xd_f/total_time)*elapsed_time*elapsed_time + (-2*total_x/total_time3 + (xd_0+xd_f)/total_time2)*elapsed_time*elapsed_time*elapsed_time
        
    return x_t
        
def cubicDot(time, time_0, time_f, x_0, x_f, xd_0, xd_f):
    x_t = 0
    if (time < time_0):
        x_t = xd_0
    elif(time > time_f):
        x_t = xd_f
    else:
        elapsed_time = time - time_0
        total_time = time_f - time_0
        total_time2 = total_time * total_time
        total_time3 = total_time * total_time * total_time
        total_x = x_f - x_0
        
        x_t = xd_0 + 2*(3*total_x/total_time2 - 2*xd_0/total_time - xd_f/total_time)*elapsed_time + 3*(-2*total_x/total_time3 + (xd_0+xd_f)/total_time2)*elapsed_time*elapsed_time
        
    return x_t

def cubicVector(time, time_0, time_f, x_0, x_f, xd_0, xd_f):
    res = np.zeros(x_0.shape[0])
    for i in range(0, res.shape[0]):
        res[i] = cubic(time, time_0, time_f, x_0[i], x_f[i], xd_0[i], xd_f[i])
    return res

def rotationCubic(time, time_0, time_f, rotation_0, rotation_f):
    if(time < time_0):
        return rotation_0
    elif(time >= time_f):
        return rotation_f
    else:
        tau = cubic(time, time_0, time_f, 0, 1, 0, 0)
        rot_scaler_skew = logm(np.dot(rotation_0.T, rotation_f))
        result = np.dot(rotation_0, expm(rot_scaler_skew * tau))
    return result

def rotationCubicDot(time, time_0, time_f, w_0, a_0, rotation_0, rotation_f):
    r_skew = logm(np.dot(rotation_0.T, rotation_f))
    tau = (time-time_0) / (time_f - time_0)
    r = np.array([r_skew[2,1], r_skew[0,2], r_skew[1,0]])
    c = w_0
    b = a_0/2
    a = r-b-c
    rd = np.zeros(3)
    for i in range(0,3):
        rd[i] = cubicDot(time, time_0, time_f, 0, r[i], 0, 0)
    rd = np.dot(rotation_0, rd)
    if(tau<0):
        return w_0
    if(tau>1):
        return np.zeros(3)
    return rd

def getPhi(current_rotation, desired_rotation):
    phi = np.zeros(3)
    s, v, w = np.zeros((3,3,3))
    for i in range(0,3):
        v[:,i] = current_rotation[:,i]
        w[:,i] = desired_rotation[:,i]
        s[:,i] = np.cross(w[:,i], v[:,i])
    phi = s[:,0] + s[:,1] + s[:,2]
    phi = -0.5*phi
    return phi