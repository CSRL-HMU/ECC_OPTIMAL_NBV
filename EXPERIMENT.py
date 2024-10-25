from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg
import numpy as np
import matplotlib.pyplot as plt
from Pendulum import Pendulum
from camera_extrinsic import camera_extrinsic
from F_x import F_x
import time

import pyzed.sl as sl
import cv2
import numpy as np
import dt_apriltags as apriltag  # Alternative package


# Initialize the ZED camera
zed = sl.Camera()

init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720  # Use 720p for performance

init_params.camera_fps = 30  # Set FPS to 30
init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # For depth information



status = zed.open(init_params)
if status != sl.ERROR_CODE.SUCCESS:
    print(f"Error: {status}")
    exit(1)

# Set camera to manual exposure mode
zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 10)
# Set up AprilTag detector
detector = apriltag.Detector(families='tag36h11')

# Prepare containers
image_zed = sl.Mat()
point_cloud = sl.Mat()  # For 3D coordinates (X, Y, Z)

# Main loop
while True:
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        # Retrieve the left image and the point cloud
        zed.retrieve_image(image_zed, sl.VIEW.LEFT)
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)  # XYZ data

        # Convert ZED image to a NumPy array
        frame = image_zed.get_data()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        

        # Get the frame dimensions (image size)
        frame_height, frame_width = frame_gray.shape[:2]  # Correctly get the height and width

        # Detect AprilTags
        tags = detector.detect(frame_gray)

        for tag in tags:
            if tag.tag_id == 0:  # Filter only for the tag with ID 0
                # Draw a bounding box around the detected tag
                for idx in range(len(tag.corners)):
                    cv2.line(frame,
                             tuple(tag.corners[idx - 1, :].astype(int)),
                             tuple(tag.corners[idx, :].astype(int)),
                             (0, 255, 0), 2)

                # Calculate the center of the detected tag
                center = np.mean(tag.corners, axis=0).astype(int)
                center_x, center_y = int(center[0]), int(center[1])

                # Retrieve the 3D coordinates (X, Y, Z) at the center of the detected AprilTag
                err, point_cloud_value = point_cloud.get_value(center_x, center_y)
                x, y, z = point_cloud_value[0], point_cloud_value[1], point_cloud_value[2]

                # Check if the tag is within the field of view (i.e., within image bounds)
                if 0 <= center_x < frame_width and 0 <= center_y < frame_height:
                    # It's within the FOV
                    print("AprilTag is within the field of view")
                else:
                    # It's outside the FOV
                    print("AprilTag is out of the field of view")


                if not np.isnan(z) and not np.isinf(z):
                    # Display the detected tag ID and depth information
                    cv2.putText(frame, f"ID: {tag.tag_id} Depth: {z:.2f}m",
                                (center_x - 50, center_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Print the X, Y, Z coordinates to the console
                    print(f"AprilTag ID {tag.tag_id} - X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}")
                else:
                    cv2.putText(frame, f"ID: {tag.tag_id} Depth: N/A",
                                (center_x - 50, center_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Display the image with detected AprilTag and depth information
        cv2.imshow("ZED2 Camera - AprilTag Detection with Depth and XYZ", frame)

    key = cv2.waitKey(1)
    if key == 27:  # Press 'ESC' to quit
        break

# Release resources
cv2.destroyAllWindows()
zed.close()





























# global L, g, b, m, dt, noise_level, fx, fy, cx, cy, image_width, image_height, SIGMA_normal, SIGMA_STO_8EO, Q, x_hat, N, state_dim, K

# g = 9.81  # gravitational acceleration (m/s^2)
# L = 1     # length of the pendulum (m)
# b = 0.04
# m = 1
# dt = 0.01


# def sigmaPoints(n, x, P):
#     # Lambda and kappa values
#     Lambda = -5.94
#     kappa = 0

#     # Cholesky decomposition of (Lambda + n + kappa) * P
#     sqrMtrx = (Lambda + n + kappa) * P
#     L = np.linalg.cholesky(sqrMtrx)  # Lower triangular Cholesky decomposition

#     # Initialize sigma point matrix S (n x 2n + 1)
#     S = np.zeros((n, 2 * n + 1))

#     # Reshape x to a column vector (equivalent to MATLAB's x(:))
#     x = x.reshape(-1)

#     # Set the first column of S to x
#     S[:, 0] = x

#     # Loop through to set the remaining sigma points
#     for i in range(n):
#         S[:, 
# global L, g, b, m, dt, noise_level, fx, fy, cx, cy, image_width, image_height, SIGMA_normal, SIGMA_STO_8EO, Q, x_hat, N, state_dim, K

# g = 9.81  # gravitational acceleration (m/s^2)
# L = 1     # length of the pendulum (m)
# b = 0.04
# m = 1
# dt = 0.01


# def sigmaPoints(n, x, P):
#     # Lambda and kappa values
#     Lambda = -5.94
#     kappa = 0

#     # Cholesky decomposition of (Lambda + n + kappa) * P
#     sqrMtrx = (Lambda + n + kappa) * P
#     L = np.linalg.cholesky(sqrMtrx)  # Lower triangular Cholesky decomposition

#     # Initialize sigma point matrix S (n x 2n + 1)
#     S = np.zeros((n, 2 * n + 1))

#     # Reshape
# noise_level = 0.05
# g = 9.81  # gravitational acceleration (m/s^2)
# L = 1     # length of the pendulum (m)
# b = 0.04
# m = 1
# dt = 0.01

# # Generate trajectory
# trajectory = Pendulum(0, L, 0, 0, 0, 0)

# # Plot 3D trajectory
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot3D(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], linewidth=2)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('3D Pendulum Trajectory - Revised')
# plt.grid(True)
# plt.axis('equal')
# plt.show()

# # Plot X, Y, Z components
# plt.figure()
# plt.plot(trajectory[:, 0], label='X')
# plt.plot(trajectory[:, 1], label='Y')
# plt.plot(trajectory[:, 2], label='Z')
# plt.legend()
# plt.show()

# # Plot (X^2 + Y^2 + Z^2) - L^2
# plt.figure()
# plt.plot((trajectory[:, 0]**2 + trajectory[:, 1]**2 + trajectory[:, 2]**2) - L**2)
# plt.legend(['(X^2 + Y^2 + Z^2) - L^2'])
# plt.show()





# ###################
# c_extr = camera_extrinsic()
# ###################


# # Measurement Noise (R matrix in Bibliography)
# SIGMA_normal = np.array([[0.5, 0, 0],
#                          [0, 0.1, 0],
#                          [0, 0, 0.1]])
# SIGMA_STO_8EO = np.array([[5000, 0, 0],
#                           [0, 5000, 0],
#                           [0, 0, 5000]])

# # Process Noise (Q matrix)
# Q = np.array([[0.01, 0, 0, 0, 0, 0],
#               [0, 0.01, 0, 0, 0, 0],
#               [0, 0, 0.1, 0, 0, 0],
#               [0, 0, 0, 50, 0, 0],
#               [0, 0, 0, 0, 50, 0],
#               [0, 0, 0, 0, 0, 500]])

# # State vector
# L = 1  # Pendulum length
# x_hat = np.array([0, L, 0, 0, 0, 0])

# N = 13
# state_dim = len(x_hat)
# X_i_iprev = np.zeros((6, N))

# # Camera intrinsic parameters of ZED2
# fx = 700.819  # Focal length in x
# fy = 700.819  # Focal length in y
# cx = 665.465  # Principal point x (center of the image)
# cy = 371.953  # Principal point y (center of the image)

# # Camera intrinsic matrix K
# K = np.array([[fx, 0, cx],
#               [0, fy, cy],
#               [0, 0, 1]])

# image_width = 680
# image_height = 720

# # Define the radius of the sphere
# r = 1

# offset = 0
# # Create mesh grid for spherical coordinates (theta, phi)
# theta = np.linspace(0, 2 * np.pi, 50)  # angle around the z-axis (azimuth)
# phi = np.linspace(0, np.pi / 2, 25)    # angle from the z-axis (elevation, only upper half)
# Theta, Phi = np.meshgrid(theta, phi)

# # Parametric equations for the sphere in Cartesian coordinates (World frame coords)
# X_pos_of_lens = r * np.cos(Phi) + offset
# Z_pos_of_lens = r * np.sin(Phi) * np.sin(Theta)
# Y_pos_of_lens = r * np.sin(Phi) * np.cos(Theta)

# # Setup for the plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_title('Camera Motion Animation with Moving Target on Hemisphere')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.grid(True)
# ax.set_box_aspect([1,1,1])  # For axis equal

# # Plot the hemisphere surface (fixed)
# # ax.plot_surface(X_pos_of_lens, Y_pos_of_lens, Z_pos_of_lens, alpha=0.3, edgecolors='none')

# # Initial plot objects for the camera position, forward direction, and target point
# h_camera, = ax.plot([0], [0], [0], 'ro', markersize=5, markerfacecolor='r')  # Camera position
# h_forward = ax.quiver(0, 0, 0, 0, 0, 0, length=0.2, color='b', linewidth=1.5)  # Forward direction
# h_target, = ax.plot([0], [0], [0], 'go', markersize=10, markerfacecolor='g')  # Target point



# dt = 0.01     # Time step


# # Generate the target point trajectory
# target_point = Pendulum(0, L, 0, 0, 0, 0).T


# # Generate Gaussian (normal) noise for each component
# noise_level = 0.1  # Noise level
# X_noise = noise_level * np.random.randn(np.size(target_point[0, :]))
# Y_noise = noise_level * np.random.randn(np.size(target_point[1, :]))
# Z_noise = noise_level * np.random.randn(np.size(target_point[2, :]))

# # Add the noise to the trajectory
# X_noisy = target_point[0, :] + X_noise
# Y_noisy = target_point[1, :] + Y_noise
# Z_noisy = target_point[2, :] + Z_noise


# measurements = (np.vstack((X_noisy, Y_noisy, Z_noisy))).T

# # Weights for the Unscented Kalman Filter
# wm = np.array([-99] + [8.33] * 12)
# wc = np.array([-96.01] + [8.33] * 12)

# num_points = target_point.shape[1]

# target_points_hom = np.vstack((target_point[0:3,:], np.ones(num_points)))
# print(target_points_hom.shape)
# print(target_points_hom[0:4,1])

# # Initialize in_fov and cost arrays
# in_fov = np.zeros(X_pos_of_lens.shape, dtype=int)
# cost = np.zeros(X_pos_of_lens.shape)

# # Initialize estimations
# estimations = [[np.zeros((num_points, 3)) for _ in range(X_pos_of_lens.shape[1])] for _ in range(X_pos_of_lens.shape[0])]
# for i in range(1, X_pos_of_lens.shape[0]):
#     for j in range(X_pos_of_lens.shape[1]):
#         estimations[i][j][0, :] = x_hat[0:3]

# # Main loop to animate the camera moving across the hemisphere with dynamic target points
# for i in range(1, X_pos_of_lens.shape[0]):
#     for j in range(X_pos_of_lens.shape[1]):
#         for k in range(1, num_points):
#             # Update the target point's plot
#             h_target.set_data(target_point[0, k], target_point[1, k])
#             h_target.set_3d_properties(target_point[2, k])

#             # Project the point onto the image plane using the intrinsic matrix
#             projected_point = np.linalg.inv(c_extr[i][j]) @ target_points_hom[:, k]

#             # Project the 3D point onto the 2D pixel plane
#             pixelCoords = K @ projected_point[0:3]

#             if pixelCoords[2] >= 0:
#                 # Normalize to get pixel coordinates
#                 pixelCoords = pixelCoords / pixelCoords[2]
#                 u = pixelCoords[0]
#                 v = pixelCoords[1]

#                 # Check if the point is within the image boundaries
#                 if 0 <= u <= image_width and 0 <= v <= image_height:
#                     in_fov[i, j] += 1  # Point is within the field of view
#                     SIGMA = SIGMA_normal
#                 else:
#                     SIGMA = SIGMA_STO_8EO

#                 if k == 1:
#                     P = Q.copy()

#                 # PREDICT STAGE
#                 X = sigmaPoints(state_dim, x_hat, P)  # Sigma points

#                 # SIGMA POINTS PROPAGATION
#                 for r in range(N):
#                     X_i_iprev[:, r] = F_x(*X[:, r])

#                 # MEAN AND COVARIANCE COMPUTATION
#                 x_hat_i_iprev = X_i_iprev @ wm
#                 Pi_iprev = (X_i_iprev - x_hat_i_iprev[:, None]) @ np.diag(wc) @ (X_i_iprev - x_hat_i_iprev[:, None]).T + Q

#                 # UPDATE STAGE
#                 ZHTA = X_i_iprev[0:3, :]  # Sigma in measurement space
#                 zhta_tilda = ZHTA @ wm  # Mean measurement

#                 # Compute covariance in measurement space
#                 Pz = (ZHTA - zhta_tilda[:, None]) @ np.diag(wc) @ (ZHTA - zhta_tilda[:, None]).T + \
#                      c_extr[i][j][0:3, 0:3] @ SIGMA @ c_extr[i][j][0:3, 0:3].T

#                 # Compute the cross-covariance of the state and the measurement
#                 Pxz = (X_i_iprev - x_hat_i_iprev[:, None]) @ np.diag(wc) @ (ZHTA - zhta_tilda[:, None]).T

#                 KALMAN_GAIN = Pxz @ np.linalg.inv(Pz)

#                 # Update estimate with measurement
#                 x_hat = x_hat_i_iprev + KALMAN_GAIN @ (measurements[k, :] - zhta_tilda)
#                 estimations[i][j][k, :] = x_hat[0:3]
#                 P = Pi_iprev - KALMAN_GAIN @ Pz @ KALMAN_GAIN.T
#                 cost[i, j] += np.linalg.det(P)

# # Compute the cost function
# cost_function = cost * dt

# # Plotting results for a specific camera position (e.g., position [25, 18])
# plt.figure()
# plt.plot(trajectory[:, 0], 'k', label='Ground Truth')
# plt.plot(measurements[:, 0], 'b', label='Measurements')
# plt.plot(estimations[24][17][:, 0], 'r', label='UKF')
# plt.grid(True)
# plt.title('Pendulum - X Position')
# plt.xlabel('Samples')
# plt.ylabel('X Position (m)')
# plt.legend()

# plt.figure()
# plt.plot(trajectory[:, 1], 'k', label='Ground Truth')
# plt.plot(measurements[:, 1], 'b', label='Measurements')
# plt.plot(estimations[24][17][:, 1], 'r', label='UKF')
# plt.grid(True)
# plt.title('Pendulum - Y Position')
# plt.xlabel('Samples')
# plt.ylabel('Y Position (m)')
# plt.legend()

# plt.figure()
# plt.plot(trajectory[:, 2], 'k', label='Ground Truth')
# plt.plot(measurements[:, 2], 'b', label='Measurements')
# plt.plot(estimations[24][17][:, 2], 'r', label='UKF')
# plt.grid(True)
# plt.title('Pendulum - Z Position')
# plt.xlabel('Samples')
# plt.ylabel('Z Position (m)')
# plt.legend()

# plt.show()