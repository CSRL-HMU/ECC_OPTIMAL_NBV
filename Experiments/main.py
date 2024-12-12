import numpy as np 
from current_extrinsic import current_extrinsic
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg
import matplotlib.pyplot as plt
from Pendulum import Pendulum
from camera_extrinsic import camera_extrinsic
from camera_initial_pose import camera_initial_pose
from F_x import F_x
import cma

global L, g, b, m, dt, noise_level, fx, fy, cx, cy, image_width, image_height, SIGMA_normal, increased_SIGMA, Q, x_hat, N, state_dim, K, P, total_cost, sigma_world, initial_parameters, Pi_iprev,Pz,pivot_x,pivot_y,pivot_z,INIT_POSIT_SIMUL
import scienceplots
import random
from scipy.io import savemat
import roboticstoolbox as rt
import math
from spatialmath import SE3
import rtde_receive
import rtde_control
from CSRL_math import *
from CSRL_orientation import *
import time
import pyzed.sl as sl
import cv2
import dt_apriltags as apriltag 
from pinhole import *


pi = math.pi


Rec = np.identity(3)
Rec[0,0] = -1
Rec[1,1] = -1
pec = np.array([0.06, 0.02, 0.0999])

gec = np.identity(4)
gec[0:3,0:3] = Rec.copy()
gec[0:3,3] = pec.copy()

gce = np.linalg.inv(gec)


# Measurement Noise (R matrix in Bibliography)
#SIGMA_normal = np.array([[0.002*0.002, 0, 0],
#                         [0, 0.002*0.002, 0],
#                         [0, 0, 0.1*0.1]])
increased_SIGMA = np.array([[5000, 0, 0],
                          [0, 5000, 0],
                          [0, 0, 5000]])



# UKF parameters

kappa = 0
alpha = 0.5
beta = 0.2
n=6
lambda_ = alpha**2 * (n + kappa) - n
wm = np.full(2 * n + 1, 0.5 / (n + lambda_))
wm[0] = lambda_ / (n + lambda_)
wc = wm.copy()
wc[0] += (1 - alpha**2 + beta)




# Define leader (UR3e)
rtde_c = rtde_control.RTDEControlInterface("192.168.1.64")
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.64")

# Define the kinematics of the leader robot (UR10e)
UR_robot = rt.DHRobot([
    rt.RevoluteDH(d = 0.1807, alpha = pi/2),
    rt.RevoluteDH(a = -0.6127),
    rt.RevoluteDH(a = -0.57155),
    rt.RevoluteDH(d = 0.17415, alpha = pi/2),
    rt.RevoluteDH(d = 0.11985, alpha = -pi/2),
    rt.RevoluteDH(d = 0.11655)
], name='UR10e')

g = 9.81  # gravitational acceleration (m/s^2)
L = 1    # length of the pendulum (m)
b = 0.04
m = 1
dt = 0.033 # time step


pivot_x = 0.90535872
pivot_y = 0.28
pivot_z = 0.53869575

pend_pivot = np.array([ 0.885 + 0.099, 0.288,  0.86])


SIGMA_normal = np.array([[0.002*0.002, 0, 0],
                         [0, 0.002*0.002, 0],
                         [0, 0, 0.02*0.02]])
                         
increased_SIGMA = np.array([[5000, 0, 0],
                          [0, 5000, 0],
                          [0, 0, 5000]])




N = 13
state_dim = 6
X_i_iprev = np.zeros((6, N))

# Camera intrinsic parameters of ZED2 for 1280 X 720 resolution
fx = 720  # Focal length in x+ 0.14
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
image_height = 7200.19127344


ph = PinholeCamera(fx, fy, cx, cy, image_width, image_height)

# Define the radius of the sphere
r = 0.5



def getQ(p_world):
    lvec = (p_world - pend_pivot)
    lvec_norm = math.sqrt(lvec[0]*lvec[0] + lvec[1]*lvec[1] + lvec[2]*lvec[2])
    lvec = lvec / lvec_norm 

    xpend =  np.cross(lvec, np.array([1,0,0]))
    xpend_norm = math.sqrt(xpend[0]*xpend[0] + xpend[1]*xpend[1] + xpend[2]*xpend[2])
    xpend = xpend / xpend_norm

    ypend = np.cross(lvec, xpend)

    Rpend = np.zeros((3,3))
    Rpend[0:3,0] = xpend
    Rpend[0:3,1] = ypend
    Rpend[0:3,2] = lvec

    Qpend = np.zeros((3,3))
    # Qpend[0,0] = 0.1*0.1
    Qpend[2,2] = 0.1*0.1 
    Qret = Rpend @ Qpend @ Rpend.T
    Qret = Qret + np.identity(3)*0.02*0.02
    return Qret


def moveToPose_custom(posed, duration):


    q = np.array(rtde_r.getActualQ())
    qd = q.copy()
    dt = 0.002
    t = 0 

    q = np.array(rtde_r.getActualQ())

    g = UR_robot.fkine(q)
    p = np.array(g.t)

    exit_flag = False

    if (p[0]-posed[0,3])**2 + (p[1]-posed[1,3])**2 + (p[2]-posed[2,3])**2 < 0.01**2:
        exit_flag = True
 
    

    kClik = 5/duration

    while t < duration+2.0 and not exit_flag:


        t = t + dt

        t_start = rtde_c.initPeriod()


        q = np.array(rtde_r.getActualQ())

        J = UR_robot.jacob0(q)
        Jinv = np.linalg.pinv(J)

        g = UR_robot.fkine(q)
        R = np.array(g.R)
        p = np.array(g.t)

        e = np.zeros(6)
        e[:3] =  posed[0:3,3] - p
        e[-3:] = logError(posed[0:3,0:3], R)

        qdot = kClik * Jinv @ e


        rtde_c.speedJ(qdot, 1.0, dt)


        rtde_c.waitPeriod(t_start)

    # Stop robot 
    rtde_c.speedStop()


    return


def sigmaPoints(n, x, P):
    # Lambda and kappa values
    kappa = 0

    Lambda = alpha**2 * (n + kappa) - n # λ = a ^2 *(N+k) - N (general equation for λ) 

    # Cholesky decomposition of (Lambda + n + kappa) * P
    sqrMtrx = (Lambda + n + kappa) * P
    # print(P)
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
    global pivot_x, pivot_y, pivot_z

    Y = r * np.sin(phi) * np.cos(theta) + pivot_y
    X = -r * np.cos(phi) + pivot_x
    Z = r * np.sin(phi) * np.sin(theta) + pivot_z
    return X, Y, Z




def objective_function(params):
    alpha, beta, phi, theta = params
    global estimations, INIT_POSIT_SIMUL
    # Parameter constraints
    if alpha <= 0 or beta <= 0 or not (0 < phi <= np.pi/3) or not (np.pi/4 <= theta <= np.pi/6 +np.pi):
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
        r = 0.5
        X_cam, Y_cam, Z_cam = get_camera_position(r, phi, theta)
        
        # Update camera extrinsics or other parapi/2meters as needed
        c_extr = current_extrinsic(X_cam, Y_cam, Z_cam)
        
        # Initialize variables
        x_hat = np.array([INIT_POSIT_SIMUL[0], INIT_POSIT_SIMUL[1], INIT_POSIT_SIMUL[2], 0, 0, 0]).T

        num_points = 100
        
        
        estimations = np.zeros((num_points, 3))
        
        estimations[0, :] = x_hat[0:3]

        Q = np.vstack((np.hstack((getQ(x_hat[0:3]),np.zeros((3,3)))),np.hstack((np.zeros((3,3)),0.01*np.identity(3)))))
        P = Q.copy()
        total_cost = 0
        
        # Run UKF over all time steps
        for k in range(1, num_points):

            # Q = np.vstack((np.hstack((getQ(x_hat[0:3]),np.zeros((3,3)))),np.hstack((np.zeros((3,3)),0.01*np.identity(3)))))
            Q = np.vstack((np.hstack((getQ(x_hat[0:3]),np.zeros((3,3)))),np.hstack((np.zeros((3,3)),0.00001*np.identity(3)))))
            
            # Measurement prediction
            # Project the point onto the image plane using the intrinsic matrix
            projected_point = np.linalg.inv(c_extr) @ np.hstack((x_hat[0:3],[1]))

            # Project the 3D point onto the 2D pixel plane
            pixelCoords = K @ projected_point[0:3]
            

            if pixelCoords[2] >= 0:
                # Normalize to get pixel coordinates
                pixelCoords = pixelCoords / pixelCoords[2]
                u = pixelCoords[0]
                v = pixelCoords[1]


                
                # Check if the point is within the image boundaries
                if 0 <= u <= image_width and 0 <= v <= image_height:
                    SIGMA = SIGMA_normal * (phi + 1) 
                else:
                    SIGMA = increased_SIGMA

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

                # UPDATE STAGEprint("tot
                ZHTA = X_i_iprev[0:3, :]  # Sigma in measurement space
                zhta_tilda = ZHTA @ wm  # Mean measurement
                
                # Compute covariance in measurement space
                Pz = (ZHTA - zhta_tilda[:, None]) @ np.diag(wc) @ (ZHTA - zhta_tilda[:, None]).T + \
                     c_extr[0:3, 0:3] @ SIGMA @ c_extr[0:3, 0:3].T

                # Compute the cross-covariance of the state and the measurement
                Pxz = (X_i_iprev - x_hat_i_iprev[:, None]) @ np.diag(wc) @ (ZHTA - zhta_tilda[:, None]).T

                KALMAN_GAIN = Pxz @ np.linalg.inv(Pz)
                
                # Update estimate with measurement
                x_hat = x_hat_i_iprev #---------------!!!!!!!!!!!!!!!!---------------------#
                estimations[k, :] = x_hat[0:3]
                P = Pi_iprev - KALMAN_GAIN @ Pz @ KALMAN_GAIN.T
                total_cost += np.linalg.det(P)*dt
                

                
        
        return total_cost
    except Exception as e:
        print(f"Exception in objective function: {e}")
        return np.inf



# Initialize the ZED camera
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.coordinate_units = sl.UNIT.METER
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.camera_fps = 30
init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE 



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

depth_map = sl.Mat()



####!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!######
####!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!######
####!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!######
##########################------------------------OPTIMIZATION---------------------------########################


#ARBITRARY/Initial
phi = pi/3
theta = pi/2


X,Y,Z = get_camera_position(r, phi, theta, offset=0)


G_initial = current_extrinsic(X, Y, Z)
R_initial =  G_initial[0:3,0:3]

T = SE3(G_initial)

moveToPose_custom(G_initial, 10)

q = np.array(rtde_r.getActualQ())


g = UR_robot.fkine(q)
R = np.array(g.R)
p = np.array(g.t)


g0e = np.identity(4)
g0e[0:3,0:3] = R.copy()
g0e[0:3,3] = p.copy()


g0c =  g0e @ gec

input('When UR ready :)...and the apriltag is at the initial position...please press "Enter"...')

exit_flag = False
#Initialize variables
while not exit_flag:
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
    
        zed.retrieve_image(image_zed, sl.VIEW.LEFT)
        zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
        frame = image_zed.get_data()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

        frame_height, frame_width = frame_gray.shape[:2]

        tags = detector.detect(frame_gray)

        for tag in tags:

            if tag.tag_id == 0:  # Assuming we're tracking tag ID 0
                # Draw a bounding box around the detected tag
                for idx in range(len(tag.corners)):
                    cv2.line(frame,
                             tuple(tag.corners[idx - 1, :].astype(int)),
                             tuple(tag.corners[idx, :].astype(int)),
                             (0, 255, 0), 2)

        
                center = np.mean(tag.corners, axis=0).astype(int)

         
                center_x, center_y = int(center[0]), int(center[1])

               
                depth_value = depth_map.get_value(center_x, center_y)
    

                p_camera = np.ones(4)
                p_camera[0:3] = ph.back_project([center_x, center_y], depth_value[1])


                p_temp = g0c @ p_camera
                p_world = p_temp[0:3]

                print(depth_value[1])
                if depth_value[1]<1 and depth_value[1]>0.2:
                    exit_flag = True


INIT_POSIT_SIMUL = np.array([p_world[0], p_world[1], p_world[2], 0, 0, 0]).T


input('Press "Enter"...')

initial_parameters = [0.1, 1, phi, theta]
sigma = 10 # step for exploring the parameter space around the current mean solution.
bounds =[[1e-3, 1e-3, 0, np.pi/4], [1.0, 3.0, np.pi/3, np.pi+np.pi/6]]

# Run CMA-ES
es = cma.CMAEvolutionStrategy(initial_parameters, sigma, {'bounds': bounds, 'maxiter': 200})

while not es.stop():
    solutions = es.ask()
    fitnesses = [objective_function(x) for x in solutions]
    es.tell(solutions, fitnesses)
    es.logger.add()
    es.disp()

optimized_params = es.result.xbest
print("Optimized parameters:", optimized_params)




alpha = optimized_params[0]
# alpha = 0.5
beta = optimized_params[1]
# beta = 0.2

phi = optimized_params[2]
theta = optimized_params[3]


X,Y,Z = get_camera_position(r, phi, theta, offset=0)


G_initial = current_extrinsic(X, Y, Z)
R_initial =  G_initial[0:3,0:3]

T = SE3(G_initial)

moveToPose_custom(G_initial, 10)

q = np.array(rtde_r.getActualQ())


g = UR_robot.fkine(q)
R = np.array(g.R)
p = np.array(g.t)


g0e = np.identity(4)
g0e[0:3,0:3] = R.copy()
g0e[0:3,3] = p.copy()


g0c =  g0e @ gec


##########################------------------------OPTIMIZATION---------------------------########################
####!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!######
####!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!######
####!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!######

exit_flag = False
#Initialize variables
input('Press "Enter"...')
while not exit_flag:
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
    
        zed.retrieve_image(image_zed, sl.VIEW.LEFT)
        zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
        frame = image_zed.get_data()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

        frame_height, frame_width = frame_gray.shape[:2]

        tags = detector.detect(frame_gray)

        for tag in tags:

            if tag.tag_id == 0:  # Assuming we're tracking tag ID 0
                # Draw a bounding box around the detected tag
                for idx in range(len(tag.corners)):
                    cv2.line(frame,
                             tuple(tag.corners[idx - 1, :].astype(int)),
                             tuple(tag.corners[idx, :].astype(int)),
                             (0, 255, 0), 2)

        
                center = np.mean(tag.corners, axis=0).astype(int)

         
                center_x, center_y = int(center[0]), int(center[1])

               
                depth_value = depth_map.get_value(center_x, center_y)
    

                p_camera = np.ones(4)
                p_camera[0:3] = ph.back_project([center_x, center_y], depth_value[1])


                p_temp = g0c @ p_camera
                p_world = p_temp[0:3]

                print(depth_value[1])
                if depth_value[1]<1 and depth_value[1]>0.2:
                    exit_flag = True


x_hat = np.array([p_world[0], p_world[1], p_world[2], 0, 0, 0]).T
t1 = 0

#print('x_hat=', x_hat)
estimations = []

estimations.append(x_hat[0:3])
PREV_X_HAT = np.zeros(3)

endofexp = 0
total_cost = 0
first_run =0
exp_time =0
prev_timestamp = None
#############################################3
# Find optimum

input('Finding optimal...Press "Enter"...')

##############################################


x_hat_log = x_hat.copy()
detP_log = np.array([0])
t_log = np.array([0])
y_log = x_hat[0:3]

# Main loop
while True and exp_time < 7:
    if zed.grab() == sl.ERROR_CODE.SUCCESS and endofexp==0:
        

        # Retrieve the left image and the point cloud
        zed.retrieve_image(image_zed, sl.VIEW.LEFT)
        
        zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)

        # Convert ZED image to a NumPy array
        frame = image_zed.get_data()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

        # Get the frame dimensions (image size)
        frame_height, framelpend_width = frame_gray.shape[:2]

        # Time update
        timestamp = time.time()
        if prev_timestamp is None:
            dt = 1/init_params.camera_fps  
        else:
            dt = timestamp - prev_timestamp
        prev_timestamp = timestamp
        # Detect AprilTags
        tags = detector.detect(frame_gray)

        

        if tags:
            for tag in tags:
                for idx in range(len(tag.corners)):
                    cv2.line(frame,
                                tuple(tag.corners[idx - 1, :].astype(int)),
                                tuple(tag.corners[idx, :].astype(int)),
                                (0, 255, 0), 2)          

                # Calculate the center of the detected tag
                center = np.mean(tag.corners, axis=0).astype(int)
                SIGMA = SIGMA_normal *  (phi + 1) 

        else:
            SIGMA = increased_SIGMA


        center_x, center_y = int(center[0]), int(center[1])


        depth_value = depth_map.get_value(center_x, center_y)
        # print(depth_value[1])
        d_camera = depth_value[1]

        if d_camera>4.0 or d_camera<0 or math.isnan(d_camera):
            d_camera = 0.45

        p_camera = np.ones(4)
        p_camera[0:3] = ph.back_project([center_x, center_y], d_camera)


        p_temp = g0c @ p_camera
        p_world = p_temp[0:3]

        
        #print('p_world=',p_world)
        
        # print('g0c=', g0c)
        # print('g0e=', g0e)

        Q = np.vstack((np.hstack((getQ(p_world),np.zeros((3,3)))),np.hstack((np.zeros((3,3)),0.001*np.identity(3)))))
        

        if first_run == 0:
            P= Q.copy()
            first_run = 1

        
        # PREDICT STAGE
        
        X = sigmaPoints(state_dim, x_hat, P)  # Sigma points
        

        for o in range(N):
            X_i_iprev[:, o] = F_x(*X[:, o])
        # print("end of sigma")
        # MEAN AND COVARIANCE COMPUTATION
        x_hat_i_iprev = X_i_iprev @ wm  ##########
        
        
        Pi_iprev = (X_i_iprev - x_hat_i_iprev[:, None]) @ np.diag(wc) @ (X_i_iprev - x_hat_i_iprev[:, None]).T + Q

        # UPDATE STAGE
        ZHTA = X_i_iprev[0:3, :]  # Sigma in measurement space
        zhta_tilda = ZHTA @ wm  # Mean measurement

        c_extr = g0c.copy()
        # Compute covariance in measurement space
        Pz = (ZHTA - zhta_tilda[:, None]) @ np.diag(wc) @ (ZHTA - zhta_tilda[:, None]).T + \
            c_extr[0:3, 0:3] @ SIGMA @ c_extr[0:3, 0:3].T

        # Compute the cross-covariance of the state and the measurement
        Pxz = (X_i_iprev - x_hat_i_iprev[:, None]) @ np.diag(wc) @ (ZHTA - zhta_tilda[:, None]).T

        KALMAN_GAIN = Pxz @ np.linalg.inv(Pz)

        sigma_world = c_extr[0:3, 0:3] @ SIGMA @ c_extr[0:3, 0:3].T
        # Update estimate with measurement
        measurements =  np.hstack([p_world[0], p_world[1], p_world[2]])
        
        x_hat = x_hat_i_iprev + KALMAN_GAIN @ (measurements - zhta_tilda)#---------------!!!!!!!!!!!!!!!!---------------------#
        
        

        print('x_hat=',x_hat)
        PREV_X_HAT = x_hat[0:3]
        estimations.append(x_hat[0:3])
        P = Pi_iprev - KALMAN_GAIN @ Pz @ KALMAN_GAIN.T
        total_cost += np.linalg.det(P)*dt

         
        
        # Optionally, display the estimated position on the frame
        est_x, est_y, est_z = x_hat[0], x_hat[1], x_hat[2]
        cv2.putText(frame, f"Est Pos: ({est_x:.2f}, {est_y:.2f}, {est_z:.2f})",
                    (10, frame_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            # else:
            #     cv2.putText(frame, f"ID: {tag.tag_id} Depth: N/A",
            #                 (center_x - 50, center_y - 10),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        exp_time+=dt
        # Display the frame with detected AprilTags and depth information
        cv2.imshow("ZED2 Camera - AprilTag Detection with UKF", frame)
        
        x_hat_log = np.vstack((x_hat_log,x_hat))
        y_log = np.vstack((y_log, p_world))
        detP_log = np.hstack((detP_log,np.linalg.det(P)))
        t_log = np.hstack((t_log,exp_time))

    key = cv2.waitKey(1)
    if key == 27:  # Press 'ESC' to quit
        pass
estimations = np.array(estimations)
# Release resources


# Create some example data
data_dict = {
    'x_hat_log': x_hat_log,
    'detP_log': detP_log,
    'y_log': y_log,
    't_log': t_log,
    'G_optimal':G_initial,
    'Optimized_params':optimized_params
}

# Save the dictionary to a .mat file
savemat('exp_outputs.mat', data_dict)



cv2.destroyAllWindows()
zed.close()





