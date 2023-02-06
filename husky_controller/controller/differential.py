import numpy as np

def DifferentialController(wheel_radius:float, 
                           wheel_distance:float,
                           linear_velocity:float,
                           angular_velocity:float,
                           max_linear_speed:float = None,
                           max_angular_speed:float = None,
                           max_wheel_speed:float = None,
                           is_skid:bool = False,
                           wheel_distance_multiplier:float = 1.0,
                           wheel_radius_multiplier:float = 1.0,
                           )->np.array:
    
    linear_velocity_ = linear_velocity
    angular_velocity_ = angular_velocity
        
    if max_linear_speed != None:
        if linear_velocity > max_linear_speed:
            linear_velocity_ = max_linear_speed
    if max_angular_speed != None:
        if angular_velocity > max_angular_speed:
            angular_velocity_ = max_angular_speed
        
    w_L = (linear_velocity_ - angular_velocity_ * (wheel_distance * wheel_distance_multiplier / 2) ) / (wheel_radius * wheel_radius_multiplier)
    w_R = (linear_velocity_ + angular_velocity_ * (wheel_distance * wheel_distance_multiplier / 2) ) / (wheel_radius * wheel_radius_multiplier)
    
    if max_wheel_speed != None:
        if w_L > max_wheel_speed:
            w_L = max_wheel_speed
        if w_R > max_wheel_speed:
            w_R = max_wheel_speed
            
    if is_skid:
        result = np.array([w_L, w_R, w_L, w_R])
    else:
        result = np.array([w_L, w_R])
        
    return result
            