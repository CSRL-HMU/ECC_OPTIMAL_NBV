import numpy as np
import math

global L, g, b, m, dt, pivot_x,pivot_y,pivot_z
g = 9.81  # 
L = 1    
b = 0.4
m = 0.2
dt = 0.033



pivot_x = 0.965
pivot_y = 0.288
pivot_z = 0.86

def F_x(x_pos0, y_pos0, z_pos0, vx0, vy0, vz0):


    # Preallocate arrays for storing results
    p_dot = np.zeros(3)
    p = np.zeros(3)


    # Set initial conditions
    p[0] = x_pos0 - pivot_x
    p[1] = y_pos0 - pivot_y
    p[2] = z_pos0 - pivot_z
    p_dot[0] = vx0
    p_dot[1] = vy0
    p_dot[2] = vz0


    L_est = math.sqrt(p[0]**2+p[1]**2+p[2]**2)



    # Compute Np matrix
    Np = np.eye(3) - np.outer(p, p) / (np.linalg.norm(p) ** 2)

    # Compute accelerations from state-space equations
    v_dot = (Np @ np.array([0, 0, -g])) - (b / m) * p_dot

    # Update velocities
    p_dot = p_dot + v_dot * dt

    # Update positions
    p = p + (Np @ p_dot) * dt

    # Return the updated state
    state = [p[0] + pivot_x, p[1] + pivot_y, p[2] + pivot_z, p_dot[0], p_dot[1], p_dot[2]] 
    return state