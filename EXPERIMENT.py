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
import dt_apriltags as apriltag  # Alternative package

# Initialize the ZED camera
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.coordinate_units = sl.UNIT.METER
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.camera_fps = 30
init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE

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
            if tag.tag_id in [0, 1]:  # Filter for tags with IDs 0 and 1
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

                # Check if the tag is within the field of view (FOV)
                if 0 <= center_x < frame_width and 0 <= center_y < frame_height:
                    print(f"AprilTag ID {tag.tag_id} is within the field of view")
                else:
                    print(f"AprilTag ID {tag.tag_id} is out of the field of view")

                # Display depth and coordinates if within range
                if not np.isnan(z) and not np.isinf(z):
                    cv2.putText(frame, f"ID: {tag.tag_id} Depth: {z:.2f}m",
                                (center_x - 50, center_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    print(f"AprilTag ID {tag.tag_id} - X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}")
                else:
                    cv2.putText(frame, f"ID: {tag.tag_id} Depth: N/A",
                                (center_x - 50, center_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Display the frame with detected AprilTags and depth information
        cv2.imshow("ZED2 Camera - AprilTag Detection with Depth and XYZ", frame)

    key = cv2.waitKey(1)
    if key == 27:  # Press 'ESC' to quit
        break


# Release resources
cv2.destroyAllWindows()
zed.close()



