import numpy as np
from camera_initial_pose import camera_initial_pose

def camera_extrinsic():
    X_pos_of_len, Y_pos_of_len, Z_pos_of_len = camera_initial_pose()
    target_point = np.array([0, 0, 0])
    
    # Initialize transformation matrices array
    T_matrices = [[None for _ in range(X_pos_of_len.shape[1])] for _ in range(X_pos_of_len.shape[0])]

    for i in range(1, X_pos_of_len.shape[0]):
        for j in range(X_pos_of_len.shape[1]):
            # Position (translation vector)
            translation_vector = np.array([X_pos_of_len[i, j], Y_pos_of_len[i, j], Z_pos_of_len[i, j]])

            # Forward vector (pointing from camera to target point)
            forward = target_point - translation_vector
            forward = forward / np.linalg.norm(forward)  # Normalize forward vector

            # Up vector (towards z-axis)
            up = np.array([0, 0, 1])

            # Check for the degenerate case at the "north pole" (Phi = 0, forward = up)
            if np.linalg.norm(forward - up) < 1e-6:
                # When forward and up are the same, we set an arbitrary right vector
                right = np.array([1, 0, 0])  # You can choose any vector perpendicular to up
                up = np.cross(right, forward)  # Recalculate up as orthogonal to forward and right
            else:
                # Right vector (cross product of forward and up)
                right = np.cross(up, forward)
                right = right / np.linalg.norm(right)  # Normalize right vector

                # Recompute up vector as orthogonal to forward and right
                up = np.cross(forward, right)

            # Rotation matrix R
            R = np.column_stack((right, up, forward))

            # Homogeneous transformation matrix
            T = np.vstack((np.column_stack((R, translation_vector)), [0, 0, 0, 1]))

            # Store the matrix in the array
            T_matrices[i][j] = T

    return T_matrices
