import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def camera_initial_pose():
    # Define the radius of the sphere
    r = 5
    offset = 0

    # Create mesh grid for spherical coordinates (theta, phi)
    theta = np.linspace(0, 2 * np.pi, 50)  # Angle around the z-axis (azimuth)
    phi = np.linspace(0, np.pi / 2, 25)    # Angle from the z-axis (elevation, only upper half)

    Theta, Phi = np.meshgrid(theta, phi)

    # Parametric equations for the sphere in Cartesian coordinates (World frame coordinates)
    X_position_of_lens = r * np.cos(Phi) + offset
    Z_position_of_lens = r * np.sin(Phi) * np.sin(Theta)
    Y_position_of_lens = r * np.sin(Phi) * np.cos(Theta)

    # Adjust plot settings for better visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set the muted red color for the surface
    red_color = [0.8, 0.5, 0.5]  # Muted red RGB values
    ax.plot_surface(X_position_of_lens, Y_position_of_lens, Z_position_of_lens, color=red_color, edgecolor='none', alpha=0.5)
    
    # Add labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Positions')
    ax.set_box_aspect([1, 1, 0.6])  # Aspect ratio

    # # Camera body (cuboid)
    # z_body = np.array([-0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5])  # X coordinates of the body
    # y_body = np.array([-0.3, -0.3, 0.3, 0.3, -0.3, -0.3, 0.3, 0.3])  # Y coordinates of the body
    # x_body = np.array([-0.2 + 1.5 + offset, -0.2 + 1.5 + offset, -0.2 + 1.5 + offset, -0.2 + 1.5 + offset,
    #                    0.2 + 1.5 + offset, 0.2 + 1.5 + offset, 0.2 + 1.5 + offset, 0.2 + 1.5 + offset])  # Z coordinates of the body

    # ax.plot_trisurf(x_body, y_body, z_body, color=[0.3, 0.3, 0.3])  # Plot the camera body (gray color)

    # Camera lens (cylinder)
    # z_lens, y_lens, x_lens = [arr.flatten() for arr in np.meshgrid(np.linspace(-0.15, 0.15, 30), np.linspace(-0.3, 0.3, 30))]  # Define a cylinder for the lens
    # x_lens *= 0.3  # Adjust the height of the cylinder
    # ax.plot_surface(x_lens + 1, y_lens, z_lens + offset, color=[0.1, 0.1, 0.1], edgecolor='none')  # Dark color for lens

    # Adjust the plot
    ax.view_init(30, 60)
    plt.axis('equal')
    #plt.show()
    return X_position_of_lens, Y_position_of_lens, Z_position_of_lens



