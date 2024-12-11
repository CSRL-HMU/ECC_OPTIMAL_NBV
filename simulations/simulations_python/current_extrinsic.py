import numpy as np

def current_extrinsic(X_pos_of_len, Y_pos_of_len, Z_pos_of_len):

    target_point = np.array([0, 0, 0])
    
    # Position (translation vector)
    translation_vector = np.array([X_pos_of_len, Y_pos_of_len, Z_pos_of_len])

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

    return T
