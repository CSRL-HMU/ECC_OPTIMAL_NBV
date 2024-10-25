from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg
import numpy as np
import matplotlib.pyplot as plt
from Pendulum import Pendulum
from camera_extrinsic import camera_extrinsic
from camera_initial_pose import camera_initial_pose
from F_x import F_x
import cma
from current_extrinsic import current_extrinsic
global L, g, b, m, dt, noise_level, fx, fy, cx, cy, image_width, image_height, SIGMA_normal, SIGMA_STO_8EO, Q, x_hat, N, state_dim, K
import scienceplots
import random

plt.style.use(["default","no-latex"])
# Constants

g = 9.81  # gravitational acceleration (m/s^2)
L = 1    # length of the pendulum (m)
b = 0.04
m = 1
dt = 0.033 # time step

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
SIGMA_normal = np.array([[0.01, 0, 0],
                         [0, 0.01, 0],
                         [0, 0, 0.2]])
SIGMA_STO_8EO = np.array([[5000, 0, 0],
                          [0, 5000, 0],
                          [0, 0, 5000]])

# Process Noise (Q matrix)
Q = np.array([[0.02, 0, 0, 0, 0, 0],
              [0, 0.001, 0, 0, 0, 0],
              [0, 0, 0.001, 0, 0, 0],
              [0, 0, 0, 0.1, 0, 0],
              [0, 0, 0, 0, 0.01, 0],
              [0, 0, 0, 0, 0, 0.01]])

Qdiag = np.sqrt(Q @ np.ones(6))

# State vector
x_hat = np.array([0, L, 0, 0, 0, 0])

N = 13
state_dim = len(x_hat)
X_i_iprev = np.zeros((6, N))

# Camera intrinsic parameters of ZED2 for 1280 X 720 resolution
fx = 720  # Focal length in x
fy = 720  # Focal length in y
cx = 640  # Principal point x (center of the image)
cy = 360  # Principal point y (center of the image)

# # Camera intrinsic parameters of D435 Realsense
# fx = 870.00
# fy = 900.00
# cx = 640.886
# cy = 363.087

# Camera intrinsic matrix - K
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])

# ZED 2
image_width = 1280
image_height = 720

# Define the radius of the sphere
r = 5

offset = -1 # In order to define the x-coordinate center value of dome  

def sigmaPoints(n, x, P, alpha_):
    # Lambda and kappa values
    kappa = 0

    Lambda = alpha_**2 * (n + kappa) - n # λ = a ^2 *(N+k) - N (general equation for λ) 

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


def get_camera_position(r, phi, theta, offset=-1):
    X = r * np.cos(phi) + offset
    Y = r * np.sin(phi) * np.cos(theta)
    Z = r * np.sin(phi) * np.sin(theta)
    return X, Y, Z


def objective_function(params):
    alpha, beta, phi, theta = params
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
        r = 5
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
                    # SIGMA = SIGMA_normal

                if k == 1:
                    P = Q.copy()

                # PREDICT STAGE
                X = sigmaPoints(state_dim, x_hat, P, alpha)  # Sigma points

                # SIGMA POINTS PROPAGATION
                for o in range(N):
                    X_i_iprev[:, o] = F_x(*X[:, o], QdiagIn = Qdiag)

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
                x_hat = x_hat_i_iprev +0* KALMAN_GAIN @ (measurements[k, :] - zhta_tilda)#---------------!!!!!!!!!!!!!!!!---------------------#
                estimations[k, :] = x_hat[0:3]
                P = Pi_iprev - KALMAN_GAIN @ Pz @ KALMAN_GAIN.T
                total_cost += np.linalg.det(P)*dt

        return total_cost
    except Exception as e:
        print(f"Exception in objective function: {e}")
        return np.inf

experiments = np.zeros([10,4])
initial_params = np.zeros([10,4])

# Define ranges for phi and theta (you can adjust these ranges based on your needs)
phi_range = np.linspace(np.pi / 5, np.pi / 3, 2)   # 10 values between 0 and pi for phi
theta_range = np.linspace(0, 2 * np.pi - (1/5)* 2 * np.pi, 5)  # 10 values between 0 and 2*pi for theta

# Create the meshgrid
phi_grid, theta_grid = np.meshgrid(phi_range, theta_range)

# Reshape the meshgrid into a list of combinations (10x10 combinations)
phi_combinations = phi_grid.flatten()
theta_combinations = theta_grid.flatten()


for exper in range(1, 10):
# Set initial parameters and bounds
    
    initial_params[exper,:] = [0.1, 1.0, phi_combinations[exper], theta_combinations[exper]]
    initial_parameters = [0.1, 1.0, phi_combinations[exper], theta_combinations[exper]]
    sigma = 0.1 # step for exploring the parameter space around the current mean solution.
    bounds =[[1e-3, 1e-3, 0, 0], [1.0, 3.0, np.pi / 2, 2 * np.pi]]

    # Run CMA-ES
    es = cma.CMAEvolutionStrategy(initial_parameters, sigma, {'bounds': bounds, 'maxiter': 250})

    # Lists to store covariance metrics

    cma_cov_dets = [] # Confidence Level of CMA output
    # Initialize a list to store the best objective function value at each iteration
    best_objective_function_value_each_iteration = []
    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective_function(x) for x in solutions]
        es.tell(solutions, fitnesses)
        es.logger.add()
        es.disp()

        # Record the best (minimum) objective function value for this iteration
        min_fitness = min(fitnesses)
        best_objective_function_value_each_iteration.append(min_fitness)
        # Record CMA-ES covariance matrix metrics
        covariance_matrix = es.C  # Internal CMA-ES covariance matrix
        cma_cov_dets.append(np.linalg.det(covariance_matrix))

    optimized_params = es.result.xbest
    experiments[exper,:] = np.array(optimized_params)
    print("Optimized parameters:", optimized_params)



# Create a figure and two subplots (2 rows, 1 column)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7))

# First subplot: Uncertainty plot
ax1.plot(cma_cov_dets, label=r'$\det(\mathbf{C})$', color='b')
ax1.set_xlabel(r'Iterations', fontsize=10)  
ax1.set_ylabel(r'$\det(\mathbf{C})$', fontsize=10)  
#ax1.set_title(r'Uncertainty Level of CMA-ES Output', fontsize=12)  
ax1.legend()
ax1.grid(True)

# Second subplot: Objective function value over iterations
ax2.plot(best_objective_function_value_each_iteration, 
         label=r'$\mathbf{\xi}^* = \underset{\mathbf{\xi}}{\arg\min} \sum_{k=1}^{\bar{k}} \det(\mathbf{P}_{k,k}) \, dt$', 
         color='r')
ax2.set_xlabel(r'Iterations', fontsize=10)  
ax2.set_ylabel(r'$f(\mathbf{\xi})$', fontsize=10)  
#ax2.set_title(r'Progress of Objective Function During CMA-ES', fontsize=12)  
ax2.legend()
ax2.grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the combined plot with subplots
plt.show()



n = 6
kappa = 0


##################-------------------AFTER OPTIMIZATION--------------------#########################


lambda_ = optimized_params[0]**2 * (n + kappa) - n
wm = np.full(2 * n + 1, 0.5 / (n + lambda_))
wm[0] = lambda_ / (n + lambda_)
wc = wm.copy()
wc[0] += (1 - optimized_params[0]**2 + optimized_params[1])

# Set camera position
r = 5
X_cam, Y_cam, Z_cam = get_camera_position(r, optimized_params[2], optimized_params[3])

# Update camera extrinsics or other parameters as needed
c_extr = current_extrinsic(X_cam, Y_cam, Z_cam)


X_noise =  np.sqrt(SIGMA_normal[0,0]) * np.random.randn(np.size(target_point[0, :]))
Y_noise =  np.sqrt(SIGMA_normal[1,1]) * np.random.randn(np.size(target_point[1, :]))
Z_noise =  np.sqrt(SIGMA_normal[2,2]) * np.random.randn(np.size(target_point[2, :]))

num_points = target_point.shape[1]

for i in range(num_points):
    noise_cam = np.array([X_noise[i],Y_noise[i],Z_noise[i]])
    noise_world = c_extr[0:3,0:3] @ noise_cam
    X_noise[i] = noise_world[0]
    Y_noise[i] = noise_world[1]
    Z_noise[i] = noise_world[2]


# Add the noise to the trajectory
X_noisy = target_point[0, :] + X_noise
Y_noisy = target_point[1, :] + Y_noise
Z_noisy = target_point[2, :] + Z_noise

measurements = (np.vstack((X_noisy, Y_noisy, Z_noisy))).T


# Initialize variables
x_hat = np.array([0, L, 0, 0, 0, 0]).T
# Generate the target point trajectory
target_point = Pendulum(0, L, 0, 0, 0, 0).T


num_points = target_point.shape[1]
target_points_hom = np.vstack((target_point[0:3,:], np.ones(num_points)))


final_estimations = np.zeros((num_points, 3))
final_estimations[0, :] = x_hat[0:3]


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
        X = sigmaPoints(state_dim, x_hat, P,optimized_params[0])  # Sigma points

        # SIGMA POINTS PROPAGATION
        for o in range(N):
            X_i_iprev[:, o] = F_x(*X[:, o], QdiagIn = Qdiag)

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
        final_estimations[k, :] = x_hat[0:3]
        P = Pi_iprev - KALMAN_GAIN @ Pz @ KALMAN_GAIN.T
    
measurements_after = measurements.copy()

##################-------------------BEFORE OPTIMIZATION--------------------#########################

lambda_ = initial_parameters[0]**2 * (n + kappa) - n
wm = np.full(2 * n + 1, 0.5 / (n + lambda_))
wm[0] = lambda_ / (n + lambda_)
wc = wm.copy()
wc[0] += (1 - initial_parameters[0]**2 + initial_parameters[1])

# Set camera position
r = 5
X_cam, Y_cam, Z_cam = get_camera_position(r, initial_parameters[2], initial_parameters[3])

# Update camera extrinsics or other parameters as needed
c_extr = current_extrinsic(X_cam, Y_cam, Z_cam)


X_noise =  np.sqrt(SIGMA_normal[0,0]) * np.random.randn(np.size(target_point[0, :]))
Y_noise =  np.sqrt(SIGMA_normal[1,1]) * np.random.randn(np.size(target_point[1, :]))
Z_noise =  np.sqrt(SIGMA_normal[2,2]) * np.random.randn(np.size(target_point[2, :]))


num_points = target_point.shape[1]

for i in range(num_points):
    noise_cam = np.array([X_noise[i],Y_noise[i],Z_noise[i]])
    noise_world = c_extr[0:3,0:3] @ noise_cam
    X_noise[i] = noise_world[0]
    Y_noise[i] = noise_world[1]
    Z_noise[i] = noise_world[2]


# Add the noise to the trajectory
X_noisy = target_point[0, :] + X_noise
Y_noisy = target_point[1, :] + Y_noise
Z_noisy = target_point[2, :] + Z_noise

measurements = (np.vstack((X_noisy, Y_noisy, Z_noisy))).T



# Initialize variables
x_hat = np.array([0, L, 0, 0, 0, 0]).T
# Generate the target point trajectory
target_point = Pendulum(0, L, 0, 0, 0, 0).T


num_points = target_point.shape[1]
target_points_hom = np.vstack((target_point[0:3,:], np.ones(num_points)))


initial_estimations = np.zeros((num_points, 3))
initial_estimations[0, :] = x_hat[0:3]

outliers_before = np.full(num_points, -1)
P = Q.copy()
total_cost = 0
counter = 0
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
            outliers_before [k] = k
            counter+=1
            print("u=",u,"v=",v)
            print("projected_point[0:3] = ", projected_point[0:3])
            print("cextr = ", c_extr)
            print("target_points_hom[:, k] = ", target_points_hom[:, k])
            print("X_cam=", X_cam)

        if k == 1:
            P = Q.copy()

        # PREDICT STAGE
        X = sigmaPoints(state_dim, x_hat, P,initial_parameters[0])  # Sigma points

        # SIGMA POINTS PROPAGATION
        for o in range(N):
            X_i_iprev[:, o] = F_x(*X[:, o], QdiagIn = Qdiag)

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
        x_hat = x_hat_i_iprev + KALMAN_GAIN @ (measurements[k, :] - zhta_tilda)#---------------!!!!!!!!!!!!!!!!---------------------#
        initial_estimations[k, :] = x_hat[0:3]
        P = Pi_iprev - KALMAN_GAIN @ Pz @ KALMAN_GAIN.T


measurements_before = measurements.copy()
        
print("Outliers",counter/num_points)
# Setup for the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Camera Motion Animation with Moving Target on Hemispherical Dome')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.grid(True)
ax.set_box_aspect([1,1,1])  # For axis equal



# Create subplots: 3x2 layout
fig, axs = plt.subplots(3, 2, figsize=(12, 15))

# First subplot: Pendulum X Position
axs[0, 0].plot(trajectory[:, 0], 'k', label=r'Ground Truth')
axs[0, 0].plot(measurements_before[:, 0], 'b--', label=r'Measurements Before')
axs[0, 0].plot(initial_estimations[:, 0], 'b', label=r'UKF Before')
axs[0, 0].plot(measurements_after[:, 0], 'r--', label=r'Measurements After')
axs[0, 0].plot(final_estimations[:, 0], 'r', label=r'UKF After')
axs[0, 0].grid(True)
axs[0, 0].set_title(r'Pendulum - X Position', fontsize=12)
axs[0, 0].set_xlabel(r'Samples', fontsize=10)
axs[0, 0].set_ylabel(r'X Position (m)', fontsize=10)
axs[0, 0].legend()

# Second subplot: Pendulum Y Position
axs[1, 0].plot(trajectory[:, 1], 'k', label=r'Ground Truth')
axs[1, 0].plot(measurements_before[:, 1], 'b--', label=r'Measurements Before')
axs[1, 0].plot(initial_estimations[:, 1], 'b', label=r'UKF Before')
axs[1, 0].plot(measurements_after[:, 1], 'r--', label=r'Measurements After')
axs[1, 0].plot(final_estimations[:, 1], 'r', label=r'UKF After')
axs[1, 0].grid(True)
axs[1, 0].set_title(r'Pendulum - Y Position', fontsize=12)
axs[1, 0].set_xlabel(r'Samples', fontsize=10)
axs[1, 0].set_ylabel(r'Y Position (m)', fontsize=10)
axs[1, 0].legend()

# Third subplot: Pendulum Z Position
axs[2, 0].plot(trajectory[:, 2], 'k', label=r'Ground Truth')
axs[2, 0].plot(measurements_before[:, 2], 'b--', label=r'Measurements Before')
axs[2, 0].plot(initial_estimations[:, 2], 'b', label=r'UKF Before')
axs[2, 0].plot(measurements_after[:, 2], 'r--', label=r'Measurements After')
axs[2, 0].plot(final_estimations[:, 2], 'r', label=r'UKF After')
axs[2, 0].grid(True)
axs[2, 0].set_title(r'Pendulum - Z Position', fontsize=12)
axs[2, 0].set_xlabel(r'Samples', fontsize=10)
axs[2, 0].set_ylabel(r'Z Position (m)', fontsize=10)
axs[2, 0].legend()

# Fourth subplot: Error in X
axs[0, 1].plot(trajectory[:, 0] - final_estimations[:, 0], 'r', label=r'$X_{Ground\_Tr} - X_{After\_Optim}$')
axs[0, 1].plot(trajectory[:, 0] - initial_estimations[:, 0], 'b', label=r'$X_{Ground\_Tr} - X_{Before\_Optim}$')
axs[0, 1].grid(True)
axs[0, 1].set_title(r'Error in X', fontsize=12)
axs[0, 1].legend()

# Fifth subplot: Error in Y
axs[1, 1].plot(trajectory[:, 1] - final_estimations[:, 1], 'r', label=r'$Y_{Ground\_Tr} - Y_{After\_Optim}$')
axs[1, 1].plot(trajectory[:, 1] - initial_estimations[:, 1], 'b', label=r'$Y_{Ground\_Tr} - Y_{Before\_Optim}$')
axs[1, 1].grid(True)
axs[1, 1].set_title(r'Error in Y', fontsize=12)
axs[1, 1].legend()

# Sixth subplot: Error in Z
axs[2, 1].plot(trajectory[:, 2] - final_estimations[:, 2], 'r', label=r'$Z_{Ground\_Tr} - Z_{After\_Optim}$')
axs[2, 1].plot(trajectory[:, 2] - initial_estimations[:, 2], 'b', label=r'$Z_{Ground\_Tr} - Z_{Before\_Optim}$')
axs[2, 1].grid(True)
axs[2, 1].set_title(r'Error in Z', fontsize=12)
axs[2, 1].legend()

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()




fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
r = 5
offset = -1

# Create mesh grid for spherical coordinates (theta, phi)
theta = np.linspace(0, 2 * np.pi, 50)  # Angle around the z-axis (azimuth)
phi = np.linspace(0, np.pi / 2, 25)    # Angle from the z-axis (elevation, only upper half)

Theta, Phi = np.meshgrid(theta, phi)

# Parametric equations for the sphere in Cartesian coordinates (World frame coordinates)
X_position_of_lens = r * np.cos(Phi) + offset
Y_position_of_lens = r * np.sin(Phi) * np.cos(Theta)
Z_position_of_lens = r * np.sin(Phi) * np.sin(Theta)

for exper in range(1, 10):

    x_camera_before = r * np.cos(initial_params[exper,2]) + offset
    y_camera_before  = r * np.sin(initial_params[exper,2]) * np.cos(initial_params[exper,3])
    z_camera_before  = r * np.sin(initial_params[exper,2]) * np.sin(initial_params[exper,3])

    x_camera_after = r * np.cos(experiments[exper,2]) + offset
    y_camera_after  = r * np.sin(experiments[exper,2]) * np.cos(experiments[exper,3])
    z_camera_after   = r * np.sin(experiments[exper,2]) * np.sin(experiments[exper,3])


    targ_point = np.array([0, 0, 0])

    #------------------------------------------------------------------------------------------
    # Camera forward vector (local z-axis) in spherical coordinates
    forward_vector_before = targ_point - np.array([x_camera_before,
                            y_camera_before,
                            z_camera_before])

    forward_vector_before = forward_vector_before / np.linalg.norm(forward_vector_before)
    # Camera up vector (local y-axis), assuming the global up is initially along the z-axis
    up_vector_before = np.array([0, 0, 1])  # Global up direction
    right_vector_before = np.cross(up_vector_before, forward_vector_before)  # Right vector (local x-axis)
    right_vector_before = right_vector_before / np.linalg.norm(right_vector_before)  # Normalize it
    up_vector_before = np.cross(forward_vector_before, right_vector_before)  # Adjust up vector to be orthogonal to forward and right

    # Scale vectors for visualization purposes
    scale = 1
    right_vector_before = right_vector_before * scale
    up_vector_before = up_vector_before * scale
    forward_vector_before = forward_vector_before * scale



    # Camera forward vector (local z-axis) in spherical coordinates----AFTER
    forward_vector_after = targ_point - np.array([x_camera_after,
                            y_camera_after,
                            z_camera_after])

    forward_vector_after = forward_vector_after / np.linalg.norm(forward_vector_after)
    # Camera up vector (local y-axis), assuming the global up is initially along the z-axis
    up_vector_after = np.array([0, 0, 1])  # Global up direction
    right_vector_after = np.cross(up_vector_after, forward_vector_after)  # Right vector (local x-axis)
    right_vector_after = right_vector_after / np.linalg.norm(right_vector_after)  # Normalize it
    up_vector_after = np.cross(forward_vector_after, right_vector_after)  # Adjust up vector to be orthogonal to forward and right

    # Scale vectors for visualization purposes
    scale = 1
    right_vector_after = right_vector_after * scale
    up_vector_after = up_vector_after * scale
    forward_vector_after = forward_vector_after * scale


    #------------------------------------------------------------------------------------------

    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],'k', label='Ground Truth' if exper == 1 else "")
    ax.plot(final_estimations[:, 0],final_estimations[:, 1],final_estimations[:, 2],'c', label='UKF_AFTER'if exper == 1 else "")
    ax.plot(initial_estimations[:, 0],initial_estimations[:, 1],initial_estimations[:, 2],'m', label='UKF_BEFORE'if exper == 1 else "")


    ############################------------BEFORE-------------------------------
    # Plot the camera position
    ax.scatter(x_camera_before, y_camera_before, z_camera_before, color='y', s=50,label='Initial Camera Position' if exper == 1 else "")
    ax.text(x_camera_before, y_camera_before, z_camera_before, f'{exper}', color='black', fontsize=12, fontweight='bold')
    # Plot the camera orientation (right, up, forward vectors)
    ax.quiver(x_camera_before, y_camera_before, z_camera_before, right_vector_before[0], right_vector_before[1], right_vector_before[2], color='g', length=scale)
    ax.quiver(x_camera_before, y_camera_before, z_camera_before, up_vector_before[0], up_vector_before[1], up_vector_before[2], color='b', length=scale)
    ax.quiver(x_camera_before, y_camera_before, z_camera_before, forward_vector_before[0], forward_vector_before[1], forward_vector_before[2], color='r', length=scale)

    ############################------------BEFORE-------------------------------


    ############################------------AFTER-------------------------------
    ax.scatter(x_camera_after, y_camera_after, z_camera_after, color='r', s=50, marker='*', label='Camera Position after Optimization' if exper == 1 else "")
    # Add a text label next to each "after" point showing its experiment number
    ax.text(x_camera_after, y_camera_after, z_camera_after, f'{exper}', color='black' ,fontsize=12, fontweight='bold')
    

    # Plot the camera orientation (right, up, forward vectors)
    ax.quiver(x_camera_after, y_camera_after, z_camera_after, right_vector_after[0], right_vector_after[1], right_vector_after[2], color='g', length=scale)
    ax.quiver(x_camera_after, y_camera_after, z_camera_after, up_vector_after[0], up_vector_after[1], up_vector_after[2], color='b', length=scale)
    ax.quiver(x_camera_after, y_camera_after, z_camera_after, forward_vector_after[0], forward_vector_after[1], forward_vector_after[2], color='r', length=scale)


    ############################------------AFTER-------------------------------

    for i in range(1, num_points): 
        if(outliers_before [i]>=0):
            ax.scatter(initial_estimations[outliers_before [i], 0],initial_estimations[outliers_before [i], 1],initial_estimations[outliers_before [i], 2], label='initial_outliers')
            

# Set the muted red color for the surface
red_color = [0.8, 0.5, 0.5]  # Muted red RGB values
ax.plot_surface(X_position_of_lens, Y_position_of_lens, Z_position_of_lens, color=red_color, edgecolor='none', alpha=0.5)


# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Camera Positions')

# Disable auto-scaling
ax.set_aspect('auto')
ax.set_box_aspect([1, 1, 1])  # Keep the aspect ratio consistent for all axe

# Set labels and grid
ax.set_title('Camera Position and Orientation on Hemisphere')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.grid(True)

# Set tighter x, y, z limits to zoom in and focus on the points
ax.set_xlim(-r+0.3, r+0.3)  # Adjust these limits based on your data range
ax.set_ylim(-r+0.3, r+0.3)
ax.set_zlim(-r+0.3, r+0.3)

# Show legend
ax.legend()
plt.axis('equal')
# Show the plot
plt.show()
