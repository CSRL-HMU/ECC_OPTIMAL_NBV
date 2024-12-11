import numpy as np

g = 9.81  # gravitational acceleration (m/s^2)
L = 1    # length of the pendulum (m)
b = 0.04
m = 1
dt = 0.033



def Pendulum(x_pos0, y_pos0, z_pos0, vx0, vy0, vz0):
    # Time parameters
    t_end = 3      # End time for simulation
    t = np.arange(0, t_end, dt)  # Time vector

    # Preallocate arrays for storing results
    v_dot = np.zeros((len(t), 3))
    p_dot = np.zeros((len(t), 3))
    p = np.zeros((len(t), 3))

    # Set initial conditions
    p[0, 0] = x_pos0
    p[0, 1] = y_pos0
    p[0, 2] = z_pos0
    p_dot[0, 0] = vx0
    p_dot[0, 1] = vy0
    p_dot[0, 2] = vz0

    # Time-stepping loop
    for i in range(len(t) - 1):
        Np = np.eye(3) - np.outer(p[i, :], p[i, :]) / (np.linalg.norm(p[i, :]) ** 2)
        # Compute accelerations from state-space equations
        v_dot[i, :] = (Np @ np.array([0, 0, -g])) - (b / m) * p_dot[i, :]

        # Update velocities
        p_dot[i + 1, :] = p_dot[i, :] + v_dot[i, :] * dt

        # Update positions
        p[i + 1, :] = p[i, :] + (Np @ p_dot[i + 1, :]) * dt

    state = np.array(np.hstack((p, p_dot)))
    return state
