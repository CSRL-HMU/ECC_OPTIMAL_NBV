from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg
import numpy as np
import matplotlib.pyplot as plt
from Pendulum import Pendulum
from camera_extrinsic import camera_extrinsic
from F_x import F_x
import cma
from current_extrinsic import current_extrinsic
global L, g, b, m, dt, noise_level, fx, fy, cx, cy, image_width, image_height, SIGMA_normal, SIGMA_STO_8EO, Q, x_hat, N, state_dim, K, measurements, estimations



# Constants

g = 9.81  # gravitational acceleration (m/s^2)
L = 1     # length of the pendulum (m)
b = 0.04
m = 1
dt = 0.033
# Generate Gaussian (normal) noise for each component
target_point= Pendulum(0, L, 0, 0, 0, 0).T

noise_level = 0.1  # Noise level
X_noise = noise_level * np.random.randn(np.size(target_point[0, :]))
Y_noise = noise_level * np.random.randn(np.size(target_point[1, :]))
Z_noise = noise_level * np.random.randn(np.size(target_point[2, :]))

# Add the noise to the trajectory
X_noisy = target_point[0, :] + X_noise
Y_noisy = target_point[1, :] + Y_noise
Z_noisy = target_point[2, :] + Z_noise
measurements = (np.vstack((X_noisy, Y_noisy, Z_noisy))).T

# Generate ground truth trajectory
trajectory = Pendulum(0, L, 0, 0, 0, 0)

# Measurement Noise (R matrix in Bibliography)
SIGMA_normal = np.array([[0.5, 0, 0],
                         [0, 0.1, 0],
                         [0, 0, 0.1]])
SIGMA_STO_8EO = np.array([[5000, 0, 0],
                          [0, 5000, 0],
                          [0, 0, 5000]])

# Process Noise (Q matrix)
Q = np.array([[0.01, 0, 0, 0, 0, 0],
              [0, 0.01, 0, 0, 0, 0],
              [0, 0, 0.1, 0, 0, 0],
              [0, 0, 0, 50, 0, 0],
              [0, 0, 0, 0, 50, 0],
              [0, 0, 0, 0, 0, 500]])

# State vector
L = 1  # Pendulum length
x_hat = np.array([0, L, 0, 0, 0, 0])

N = 13
state_dim = len(x_hat)
X_i_iprev = np.zeros((6, N))

# Camera intrinsic parameters of ZED2
fx = 700.819  # Focal length in x
fy = 700.819  # Focal length in y
cx = 665.465  # Principal point x (center of the image)
cy = 371.953  # Principal point y (center of the image)

# Camera intrinsic matrix K
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])

image_width = 680
image_height = 720

# Define the radius of the sphere
r = 1

offset = 0



def sigmaPoints(n, x, P):
    # Lambda and kappa values
    Lambda = -5.94
    kappa = 0

    # Cholesky decomposition of (Lambda + n + kappa) * P
    sqrMtrx = (Lambda + n + kappa) * P
    L = np.linalg.cholesky(sqrMtrx)  # Lower triangular Cholesky decomposition

    # Initialize sigma point matrix S (n x 2n + 1)
    S = np.zeros((n, 2 * n + 1))

    # Reshape x to a column vector (equivalent to MATLAB's x(:))
    x = x.reshape(-1)

    # Set the first column of S to x
    S[:, 0] = x

    # Loop through to set the remaining sigma points
    for i in range(n):
        S[:, i + 1] = x + L[:, i]
        S[:, i + n + 1] = x - L[:, i]

    return S


def get_camera_position(r, phi, theta, offset=0):
    X = r * np.cos(phi) + offset
    Y = r * np.sin(phi) * np.cos(theta)
    Z = r * np.sin(phi) * np.sin(theta)

    return X, Y, Z


def objective_function(params):
    alpha, beta, phi, theta = params
    global estimations
    # Parameter constraints
    if alpha <= 0 or beta <= 0 or not (0 < phi <= np.pi / 2) or not (0 <= theta <= 2 * np.pi):
        return np.inf  # Penalize invalid parameters
    
    try:
        # Set UKF weights
        n = 6
        kappa = 0
        lambda_ = alpha**2 * (n + kappa) - n
        wm = np.full(2 * n + 1, 0.5 / (n + lambda_))
        wm[0] = lambda_ / (n + lambda_)
        wc = wm.copy()
        wc[0] += (1 - alpha**2 + beta)
        
        # Set camera position
        r = 1
        X_cam, Y_cam, Z_cam = get_camera_position(r, phi, theta)
        
        # Update camera extrinsics or other parameters as needed
        c_extr = current_extrinsic(X_cam, Y_cam, Z_cam)
        
        # Initialize variables
        x_hat = np.array([0, L, 0, 0, 0, 0]).T
        # Generate the target point trajectory
        target_point = Pendulum(0, L, 0, 0, 0, 0).T


        num_points = target_point.shape[1]
        target_points_hom = np.vstack((target_point[0:3,:], np.ones(num_points)))
        
        
        estimations = np.zeros((num_points, 3))
        
        estimations[0, :] = x_hat[0:3]


        P = Q.copy()
        total_cost = 0
        
        # Run UKF over all time steps
        for k in range(1, num_points):
            # Measurement prediction
            # Project the point onto the image plane using the intrinsic matrix
            projected_point = np.linalg.inv(c_extr) @ target_points_hom[:, k]

            # Project the 3D point onto the 2D pixel plane
            pixelCoords = K @ projected_point[0:3]

            if pixelCoords[2] >= 0:
                # Normalize to get pixel coordinates
                pixelCoords = pixelCoords / pixelCoords[2]
                u = pixelCoords[0]
                v = pixelCoords[1]

                # Check if the point is within the image boundaries
                if 0 <= u <= image_width and 0 <= v <= image_height:
                    SIGMA = SIGMA_normal
                else:
                    SIGMA = SIGMA_STO_8EO

                if k == 1:
                    P = Q.copy()

                # PREDICT STAGE
                X = sigmaPoints(state_dim, x_hat, P)  # Sigma points

                # SIGMA POINTS PROPAGATION
                for o in range(N):
                    X_i_iprev[:, o] = F_x(*X[:, o])

                # MEAN AND COVARIANCE COMPUTATION
                x_hat_i_iprev = X_i_iprev @ wm
                Pi_iprev = (X_i_iprev - x_hat_i_iprev[:, None]) @ np.diag(wc) @ (X_i_iprev - x_hat_i_iprev[:, None]).T + Q

                # UPDATE STAGE
                ZHTA = X_i_iprev[0:3, :]  # Sigma in measurement space
                zhta_tilda = ZHTA @ wm  # Mean measurement

                # Compute covariance in measurement space
                Pz = (ZHTA - zhta_tilda[:, None]) @ np.diag(wc) @ (ZHTA - zhta_tilda[:, None]).T + \
                     c_extr[0:3, 0:3] @ SIGMA @ c_extr[0:3, 0:3].T

                # Compute the cross-covariance of the state and the measurement
                Pxz = (X_i_iprev - x_hat_i_iprev[:, None]) @ np.diag(wc) @ (ZHTA - zhta_tilda[:, None]).T

                KALMAN_GAIN = Pxz @ np.linalg.inv(Pz)

                # Update estimate with measurement
                x_hat = x_hat_i_iprev +KALMAN_GAIN @ (measurements[k, :] - zhta_tilda)#---------------!!!!!!!!!!!!!!!!---------------------#
                estimations[k, :] = x_hat[0:3]
                P = Pi_iprev - KALMAN_GAIN @ Pz @ KALMAN_GAIN.T
                total_cost += np.linalg.det(P)*dt
        
        return total_cost
    except Exception as e:
        print(f"Exception in objective function: {e}")
        return np.inf


# Set initial parameters and bounds
initial_params = [0.5, 2.0, np.pi / 4, np.pi]
sigma = 0.1
bounds = [[1e-3, 1e-3, 0, 0], [1.0, 5.0, np.pi / 2, 2 * np.pi]]


# Run CMA-ES
es = cma.CMAEvolutionStrategy(initial_params, sigma, {'bounds': bounds})
while not es.stop():
    solutions = es.ask()
    fitnesses = [objective_function(x) for x in solutions]
    es.tell(solutions, fitnesses)
    es.logger.add()
    es.disp()

optimized_params = es.result.xbest
print("Optimized parameters:", optimized_params)




# Setup for the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Camera Motion Animation with Moving Target on Hemisphere')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.grid(True)
ax.set_box_aspect([1,1,1])  # For axis equal

# Plotting results for a specific camera position (e.g., position [25, 18])
plt.figure()
plt.plot(trajectory[:, 0], 'k', label='Ground Truth')
plt.plot(measurements[:, 0], 'b', label='Measurements')
plt.plot(estimations[:, 0], 'r', label='UKF')
plt.grid(True)
plt.title('Pendulum - X Position')
plt.xlabel('Samples')
plt.ylabel('X Position (m)')
plt.legend()

plt.figure()
plt.plot(trajectory[:, 1], 'k', label='Ground Truth')
plt.plot(measurements[:, 1], 'b', label='Measurements')
plt.plot(estimations[:, 1], 'r', label='UKF')
plt.grid(True)
plt.title('Pendulum - Y Position')
plt.xlabel('Samples')
plt.ylabel('Y Position (m)')
plt.legend()

plt.figure()
plt.plot(trajectory[:, 2], 'k', label='Ground Truth')
plt.plot(measurements[:, 2], 'b', label='Measurements')
plt.plot(estimations[:, 2], 'r', label='UKF')
plt.grid(True)
plt.title('Pendulum - Z Position')
plt.xlabel('Samples')
plt.ylabel('Z Position (m)')
plt.legend()

plt.show()
