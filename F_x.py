import numpy as np

global L, g, b, m, dt
g = 9.81  # gravitational acceleration (m/s^2)
L = 1     # length of the pendulum (m)
b = 0.04
m = 1
dt = 0.033

def F_x(x_pos0, y_pos0, z_pos0, vx0, vy0, vz0, QdiagIn):


    # Preallocate arrays for storing results
    p_dot = np.zeros(3)
    p = np.zeros(3)

    # Set initial conditions
    p[0] = x_pos0
    p[1] = y_pos0
    p[2] = z_pos0
    p_dot[0] = vx0
    p_dot[1] = vy0
    p_dot[2] = vz0

    # Compute Np matrix
    Np = np.eye(3) - np.outer(p, p) / (np.linalg.norm(p) ** 2)

    # Compute accelerations from state-space equations
    v_dot = (Np @ np.array([0, 0, -g])) - (b / m) * p_dot

    # Update velocities
    p_dot = p_dot + v_dot * dt

    # Update positions
    p = p + (Np @ p_dot) * dt

    # Return the updated state
    state = [p[0], p[1], p[2], p_dot[0], p_dot[1], p_dot[2]] + QdiagIn * np.random.randn(6)
    return state