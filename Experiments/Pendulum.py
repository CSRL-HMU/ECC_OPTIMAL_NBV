import numpy as np
global pivot_x,pivot_y,pivot_z

g = 9.81  # gravitational acceleration (m/s^2)
L = 1    # length of the pendulum (m)
b = 0.04
m = 0.2
dt = 0.033

pivot_x = 0.885 + 0.099
pivot_y = 0.288
pivot_z = 0.86

def Pendulum(x_pos0, y_pos0, z_pos0, vx0, vy0, vz0):

    
    # Time parameters
    t_end = 10      # End time for simulation
    t = np.arange(0, t_end, dt)  # Time vector

    # Preallocate arrays for storing results
    v_dot = np.zeros((len(t), 3))
    p_dot = np.zeros((len(t), 3))
    p = np.zeros((len(t), 3))

    # Set initial conditions
    p[0, 0] = x_pos0 - pivot_x
    p[0, 1] = y_pos0 - pivot_y
    p[0, 2] = z_pos0 - pivot_z
    p_dot[0, 0] = vx0
    p_dot[0, 1] = vy0
    p_dot[0, 2] = vz0
    
    # Time-stepping loop
    for i in range(len(t) - 1):
        
        #print("p=", p)
        Np = np.eye(3) - np.outer(p, p) / (np.linalg.norm(p) ** 2)
        
        # Compute accelerations from state-space equations
        v_dot = (Np @ np.array([0, 0, -g])) - (b / m) * p_dot
        
        # Update velocities
        p_dot = p_dot + v_dot * dt

        # Update positions
        p = p + (Np @ p_dot) * dt

        p[0]=p[0] + pivot_x
        p[1]=p[1] + pivot_y
        p[2]=p[2] + pivot_z
        


    
    state = np.array(np.hstack((p, p_dot)))
    return state
