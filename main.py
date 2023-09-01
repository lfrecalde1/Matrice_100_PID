import sys
import time
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped
from scipy.spatial.transform import Rotation as R
from scipy.io import savemat
from sensor_msgs.msg import Joy

## Global variables system
xd = 0.0
yd = 0.0
zd = 0.0
vxd = 0.0
vyd = 0.0
vzd = 0.0

qx = 0.0005
qy = 0.0
qz = 0.0
qw = 1.0
wxd = 0.0
wyd = 0.0
wzd = 0.0

# Global control action
u_global = 0
# System Parameters
mass = 3.75
gravity = 9.81
ts = 0.02

## Global rc
axes = [0, 0, 0, 0, 0, 0]

flag_system = 0
def rc_callback(data):
    # Extraer los datos individuales del mensaje
    global axes
    axes_aux = data.axes
    psi = -np.pi / 2

    R = np.array([[np.cos(psi), -np.sin(psi), 0, 0, 0, 0],
                [np.sin(psi), np.cos(psi), 0, 0, 0, 0],
                [0, 0, -1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]])
    
    axes = R@axes_aux
    return None

def jacobian_matrix(h):
    psi = h[9]
    J_11 = np.cos(psi)
    J_12 = -np.sin(psi)
    J_13 = 0.0
    J_14 = 0.0

    J_21 = np.sin(psi)
    J_22 = np.cos(psi)
    J_23 = 0.0
    J_24 = 0.0

    J_31 = 0.0
    J_32 = 0.0
    J_33 = 1.0
    J_34 = 0.0

    J_41 = 0.0
    J_42 = 0.0
    J_43 = 0.0
    J_44 = 1.0

    J = np.array([[J_11, J_12, J_13, J_14], [J_21, J_22, J_23, J_24], [J_31, J_32, J_33, J_34], [J_41, J_42, J_43, J_44]], dtype=np.double)
    return J

def get_system_velocity_sensor_body():
    q = np.array([qx, qy, qz, qw], dtype=np.double)
    rot = R.from_quat(q)
    rot = rot.as_matrix()
    v_world = np.array([[vxd], [vyd], [vzd]], dtype=np.double)
    rot_inv = np.linalg.inv(rot)
    v_body = rot_inv@v_world
    x = np.array([v_body[0,0], v_body[1, 0], v_body[2, 0], wxd, wyd, wzd], dtype=np.double)
    return x
def pid(sp, real, memories, kp, ki, kd, t_sample):
    error = np.tanh(sp - real)
    error_1 = memories[1, 0]
    error_2 = memories[2, 0]
    u_1 = memories[0, 0]
    p = kp * (error - error_1)
    i = ki * error * t_sample
    d = kd * (error - 2 * error_1 + error_2) / t_sample
    u = u_1 + p + i + d

    # Update memories
    memories[0, 0] = u
    memories[2, 0] = error_1
    memories[1, 0] = error
    return u, memories

def controller_attitude_pitch(qdp, qp, ts, v_c):
    # Rotational matrix
    quat = np.array([qx, qy, qz, qw], dtype=np.double)
    rot = R.from_quat(quat)
    euler = rot.as_euler('xyz', degrees=False)

    # Euler angles
    psi = euler[2]

    # Create rotational matrix z
    R_z = R.from_euler('z', psi, degrees=False)
    R_z_data = R_z.as_matrix()

    # Create vector linear velocity world frame
    qp = np.array([[qp[0]], [qp[1]], [qp[2]]], dtype=np.double)

    # Velocity with respect to the body frame only rotation z
    velocity = np.linalg.inv(R_z_data)@qp
    
    # Velocity with respect to the body frame complete rotation matrix
    rot_matrix = rot.as_matrix()
    rot_inv = np.linalg.inv(rot_matrix)
    v_body = rot_inv@qp

    # Desired velocity and real velocity proyection
    xpd = qdp[0]
    #xp = np.cos(theta)*v_body[0]
    xp = velocity[0]

    # PID controller for lateral velocity y
    pitch_d, v_c = pid(xpd, xp, v_c, 0.12, 0.0, 0.001, ts)
    return pitch_d, v_c

def controller_attitude_roll(qdp, qp, ts, v_c):
    # Rotational matrix
    quat = np.array([qx, qy, qz, qw], dtype=np.double)
    rot = R.from_quat(quat)
    euler = rot.as_euler('xyz', degrees=False)

    # Euler Angles
    psi = euler[2]

    # Create rotational matrix z
    R_z = R.from_euler('z', psi, degrees=False)
    R_z_data = R_z.as_matrix()
    
    # Create vector of the velocities respect to the frame W
    qp = np.array([[qp[0]], [qp[1]], [qp[2]]], dtype=np.double)

    # Linear velocity projection over B frame using on Z rotation
    velocity = np.linalg.inv(R_z_data)@qp

    # Linear velocity projection over B frame using complete rotation matrix
    rot_matrix = rot.as_matrix()
    rot_inv = np.linalg.inv(rot_matrix)
    v_body = rot_inv@qp

    # Desired Velocity and velocty body proyection
    ypd = qdp[1]
    #yp = np.cos(theta)*v_body[1]
    yp = velocity[1]

    roll_d, v_c = pid(ypd, yp, v_c, 0.12, 0.0, 0.001, ts)
    return -roll_d, v_c

def inverse_kinematics_controller(hd, h, k1, k2):
    # Inverse Kinematics controller
    # generate Jacobian matrix
    J = jacobian_matrix(h)
    # Error vector
    h_new = np.array([h[0], h[1], h[2], h[9]], dtype=np.double)
    he = hd - h_new
    he = he.reshape(4, 1)

    # Create gain matrcies
    K1 = k1*np.eye(4, 4)
    K2 = k2*np.eye(4, 4)

    u = np.linalg.inv(J)@(K2@np.tanh(np.linalg.inv(K2)@K1@he))
    u = u.reshape(4, )

    return u

def init_system(control_pub, ref_drone, velocity_z):
    # Function to Initialize the system
    vx_c = np.array([[0.0], [0.0], [0.0]])
    vy_c = np.array([[0.0], [0.0], [0.0]])
    vz_c = np.array([[0.0], [0.0], [0.0]])
    for k in range(0, 350):
        tic = time.time()
        condicion = axes[5]
        if condicion == -10000.0:
            None
        else:
            break
        # Get system velocites Respect to W frame
        velocities = get_system_velocity_sensor()

        # PID control associated to the desired velocities
        u_global, vz_c = controller_z(mass, gravity, [0, 0, velocity_z], velocities[0:3], vz_c)
        u_ref_pitch, vx_c = controller_attitude_pitch([0, 0, velocity_z], velocities[0:3], ts, vx_c)
        u_ref_roll, vy_c = controller_attitude_roll([0, 0, velocity_z], velocities[0:3], ts, vy_c)
        ref_drone = get_reference([u_global, u_ref_roll, u_ref_pitch, 0], ref_drone)

        # Send control action to the aerial robotic system
        send_reference(ref_drone, control_pub)

        # Loop_rate.sleep()
        while (time.time() - tic <= ts):
                None
        toc = time.time() - tic 
    
    return None
    
def init_system_z(control_pub, ref_drone, k1, k2):
    # Function to Initialize the system
    vx_c = np.array([[0.0], [0.0], [0.0]])
    vy_c = np.array([[0.0], [0.0], [0.0]])
    vz_c = np.array([[0.0], [0.0], [0.0]])

    # Desired Z
    hd = np.zeros((4, 1), dtype=np.double)
    hd[0, 0] = 0.0
    hd[1, 0] = 0.0
    hd[2, 0] = 5.0

    # Read pose system
    pose = get_system_states_sensor()

    # Rad actual value angle in order to avoid problems of initial rotations
    hd[3, 0] = pose[9]

    for k in range(0, 350):
        tic = time.time()
        condicion = axes[5]
        if condicion == -10000.0:
            None
        else:
            break
        # Get system velocites Respect to W frame
        pose = get_system_states_sensor()
        velocities = get_system_velocity_sensor()

        # Control law
        control_action = inverse_kinematics_controller(hd[:, 0], pose, k1, k2)
        linear_velocities = control_action[0:3]
        angular = control_action[3]

        # PID control associated to the desired velocities
        u_global, vz_c = controller_z(mass, gravity, linear_velocities, velocities[0:3], vz_c)
        u_ref_pitch, vx_c = controller_attitude_pitch(linear_velocities, velocities[0:3], ts, vx_c)
        u_ref_roll, vy_c = controller_attitude_roll(linear_velocities, velocities[0:3], ts, vy_c)
        ref_drone = get_reference([u_global, u_ref_roll, u_ref_pitch, angular], ref_drone)

        # Send control action to the aerial robotic system
        send_reference(ref_drone, control_pub)

        # Loop_rate.sleep()
        while (time.time() - tic <= ts):
                None
        toc = time.time() - tic 
    
    return None

def controller_z(mass, gravity, qdp, qp, v_c):
    # Control Function only z velocity
    # Control Gains
    Kp = 15*np.eye(3, 3)

    # Control error
    error = qdp - qp
    error_vector = error.reshape((3,1))

    # Split values
    zpd = qdp[2]
    zp = qp[2]

    # Control Law
    #aux_control = Kp@error_vector
    aux_control, v_c = pid(zpd, zp, v_c, 15, 0.05, 0.001, ts)

    # Gravity + compensation velocity
    control_value = mass*gravity + aux_control
    
    return control_value, v_c

# Get system states
def get_system_states_sensor():
    # Get values of the system
    quat = np.array([qx, qy, qz, qw], dtype=np.double)
    rot = R.from_quat(quat)
    euler = rot.as_euler('xyz', degrees=False)

    # Complete pose of the system including postion, quaternions, and euler
    x = np.array([xd, yd, zd, qw, qx, qy, qz, euler[0], euler[1], euler[2]], dtype=np.double)
    return x

def get_system_velocity_sensor():
    # This function read the Velocities of the system
    x = np.array([vxd, vyd, vzd, wxd, wyd, wzd], dtype=np.double)
    return x

def get_reference(ref, ref_msg):
        # This function send the control actions  to the system
        # THis function sends Force in z axis Body and the angles commands
        ref_msg.twist.linear.x = 0
        ref_msg.twist.linear.y = 0
        ref_msg.twist.linear.z = ref[0]

        ref_msg.twist.angular.x = ref[1]
        ref_msg.twist.angular.y = ref[2]
        ref_msg.twist.angular.z = ref[3]
        return ref_msg

def send_reference(ref_msg, ref_pu):
    # This function send the control action to the aerial system
    ref_pu.publish(ref_msg)
    return None

def odometry_call_back(odom_msg):
    # This function reads the odometry of the system
    global xd, yd, zd, qx, qy, qz, qw, time_message, vxd, vyd, vzd, wxd, wyd, wzd

    time_message = odom_msg.header.stamp
    xd = odom_msg.pose.pose.position.x 
    yd = odom_msg.pose.pose.position.y
    zd = odom_msg.pose.pose.position.z
    vxd = odom_msg.twist.twist.linear.x
    vyd = odom_msg.twist.twist.linear.y
    vzd = odom_msg.twist.twist.linear.z


    qx = odom_msg.pose.pose.orientation.x
    qy = odom_msg.pose.pose.orientation.y
    qz = odom_msg.pose.pose.orientation.z
    qw = odom_msg.pose.pose.orientation.w

    wxd = odom_msg.twist.twist.angular.x
    wyd = odom_msg.twist.twist.angular.y
    wzd = odom_msg.twist.twist.angular.z
    return None

def get_values_rc(hdp, rdp):
    # This Function read the commands od the RC controller or the internal desired values
    condicion = axes[5]
    if condicion  == -10000.0:
        xref_ul = axes[0]
        xref_um = axes[1]
        xref_un = 2*axes[3]
        xref_wx = 0
        xref_wy = 0
        xref_wz = axes[2]
    elif condicion == -4545.0:        
        xref_ul = hdp[0]
        xref_um = hdp[1]
        xref_un = hdp[2]
        xref_wx = 0
        xref_wy = 0
        xref_wz = rdp[2]
    else:
        xref_ul = 0
        xref_um = 0
        xref_un = 0
        xref_wz = 0
        xref_wx = 0
        xref_wy = 0
    return np.array([xref_ul, xref_um, xref_un], dtype=np.double), np.array([xref_wx, xref_wy, xref_wz], dtype=np.double)

def main(control_pub):
    global massm, gravity, ts, flag_system
    # Definition Twist message in order to control the system
    ref_drone = TwistStamped()
    flag_system = 1

    # Simulation time parameters
    tf = 60
    t = np.arange(0, tf+ts, ts, dtype=np.double)

    # COntrol gains
    k1 = 1 
    k2 = 1

    # Definition vector where the odometry will be allocated
    h = np.zeros((10, t.shape[0]+1), dtype=np.double)

    # # Definition of the vector where the Velocities will be allocated
    hp = np.zeros((6, t.shape[0]+1), dtype=np.double)
    hp_b = np.zeros((6, t.shape[0]+1), dtype=np.double)

    # Definition control signal empty vector
    u_ref = np.zeros((4, t.shape[0]), dtype=np.double)

    # Set reference linear (x, y, z) velocities respect to the body frame
    hdp = np.zeros((3, t.shape[0]), dtype=np.double)
    hdp[0,:] = 0.0
    hdp[1,:] = 0.0
    hdp[2,:] = 0.0

    # Set reference angular (wx, wy, wz) velocities respect to the body frame
    rdp = np.zeros((3, t.shape[0]), dtype=np.double)
    rdp[0, :] = 0.0
    rdp[1, :] = 0.0
    rdp[2,:] = 0.0

    # Initial random point of the controller
    number_desired_points = 10
    hdx = np.random.uniform(low= -2, high= 2, size=(1, number_desired_points))
    hdy = np.random.uniform(low= -2, high= 2, size=(1, number_desired_points))
    hdz = np.random.uniform(low= 4, high=6, size=(1, number_desired_points))
    hdpsi = np.random.uniform(low= -2.5, high=2.5, size=(1, number_desired_points))

    # General vector desired points
    hd = np.zeros((4, number_desired_points), dtype=np.double)
    hd[0, :] = hdx
    hd[1, :] = hdy
    hd[2, :] = hdz
    hd[3, :] = hdpsi

    # Definiton of the number of the experiment
    experiment_number = 5

    # Initialization of the system in order to establish a proper communication and set the velocities
    init_system(control_pub, ref_drone, velocity_z=2.0)

    # Set the values of the initial conditions of the system
    h[:, 0] = get_system_states_sensor()
    hp[:, 0] = get_system_velocity_sensor()
    hp_b[:, 0] = get_system_velocity_sensor_body()

    # Auxiliar variables Time
    t_k = 0

    # Aulixar variables PID frontal and lateral velocites (vx, vy)
    vx_c = np.array([[0.0], [0.0], [0.0]])
    vy_c = np.array([[0.0], [0.0], [0.0]])
    vz_c = np.array([[0.0], [0.0], [0.0]])

    # Simulation Data
    for k in range(0, t.shape[0]):
        tic = time.time()
        condicion = axes[5]
        if condicion == -10000.0:
            None
        else:
            break
        ## Get valalues RC or Desired Signal
        hdp[:, k], rdp[:, k] = get_values_rc(hdp[:, k], rdp[:, k])

        # Control section in order to track the desired velocites generated by the RC or by the controller
        # PD z velocity body
        u_ref[0, k], vz_c = controller_z(mass, gravity, hdp[:, k], hp_b[0:3, k], vz_c)
        # PD x velocity body
        u_ref[2, k], vx_c = controller_attitude_pitch(hdp[:, k], hp[:, k], ts, vx_c)
        # PD y velocity body
        u_ref[1, k], vy_c = controller_attitude_roll(hdp[:, k], hp[:, k], ts, vy_c)
        # Angular velocity Wz body frame
        u_ref[3, k] = rdp[2, k]

        # Set Message information
        ref_drone = get_reference(u_ref[:, k], ref_drone)

        # Send the informatio to the system
        send_reference(ref_drone, control_pub)

        # Loop_rate.sleep()
        while (time.time() - tic <= ts):
                None
        toc = time.time() - tic 
        print(toc)

        # Read odometry and update information
        h[:, k+1] = get_system_states_sensor()
        hp[:, k+1] = get_system_velocity_sensor()
        hp_b[:, k+1] = get_system_velocity_sensor_body()

        # Auxiliar time variable
        t_k = t_k + toc

    # Set Velocities to zero
    init_system(control_pub, ref_drone, velocity_z=0.0)

    # Save information of the entire system
    mdic_h = {"h": h, "label": "experiment_h"}
    mdic_hp = {"hp": hp, "label": "experiment_hp"}
    mdic_hdp = {"hdp": hdp, "label": "experiment_hdp"}
    mdic_rdp = {"rdp": rdp, "label": "experiment_rdp"}
    mdic_u = {"u_ref": u_ref, "label": "experiment_u"}
    mdic_t = {"t": t, "label": "experiment_t"}

    savemat("h_"+ str(experiment_number) + ".mat", mdic_h)
    savemat("hp_"+ str(experiment_number) + ".mat", mdic_hp)
    savemat("hdp_"+ str(experiment_number) + ".mat", mdic_hdp)
    savemat("rdp_"+ str(experiment_number) + ".mat", mdic_rdp)
    savemat("u_ref_" + str(experiment_number) + ".mat", mdic_u)
    savemat("t_"+ str(experiment_number) + ".mat", mdic_t)

    return None

if __name__ == '__main__':
    try:
        # Initialization Node
        rospy.init_node("Python_Node",disable_signals=True, anonymous=True)

        # Odometry topic
        odometry_webots = "/dji_sdk/odometry"
        odometry_subscriber = rospy.Subscriber(odometry_webots, Odometry, odometry_call_back)

        # Cmd Vel topic
        velocity_topic = "/m100/velocityControl"
        velocity_publisher = rospy.Publisher(velocity_topic, TwistStamped, queue_size = 10)

        # RC controller topic
        RC_sub = rospy.Subscriber("/dji_sdk/rc", Joy, rc_callback, queue_size=10)


        while True and flag_system == 0:
            condicion = axes[5]
            if condicion == -10000.0:
                print("Run")
                main(velocity_publisher)
            else:
                print("No Run")
                ref_drone = TwistStamped()
                ref_drone = get_reference([5, 0, 0, 0], ref_drone)
                send_reference(ref_drone, velocity_publisher)
                None

    except(rospy.ROSInterruptException, KeyboardInterrupt):
        print("Error System")
        ref_drone = TwistStamped()
        init_system(velocity_publisher, ref_drone, velocity_z = 0.0)
        pass
    else:
        print("Complete Execution")
        ref_drone = TwistStamped()
        init_system(velocity_publisher, ref_drone, velocity_z = 0.0)
        pass