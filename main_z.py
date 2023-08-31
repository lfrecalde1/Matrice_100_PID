import sys
import time
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped
from scipy.spatial.transform import Rotation as R
from scipy.io import savemat
from sensor_msgs.msg import Joy
from scipy.interpolate import interp1d

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
mass = 3.2
gravity = 9.81
ts = 0.02

## Global rc
axes = [0, 0, 0, 0, 0, 0]
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


def get_system_velocity_sensor_body():
    q = np.array([qx, qy, qz, qw], dtype=np.double)
    rot = R.from_quat(q)
    rot = rot.as_matrix()
    v_world = np.array([[vxd], [vyd], [vzd]], dtype=np.double)
    rot_inv = np.linalg.inv(rot)
    v_body = rot_inv@v_world
    x = np.array([v_body[0,0], v_body[1, 0], v_body[2, 0], wxd, wyd, wzd], dtype=np.double)
    return x

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

def get_values_rc_f():
    # This Function read the commands od the RC controller or the internal desired values
    condicion = axes[5]
    if axes[3] <=0:
        auxiliar = 0
    else: 
        auxiliar = axes[3]

    if condicion  == -4545.0:
        inter = interp1d([0, 1], [20, 40])
        xref_un = inter(auxiliar)
    elif condicion == -10000.0:        
        xref_un = 0
    else:
        xref_un = 0
    return np.array([xref_un, 0, 0, 0], dtype=np.double)
def main(control_pub):
    global massm, gravity, ts
    # Definition Twist message in order to control the system
    ref_drone = TwistStamped()

    # Simulation time parameters
    tf = 250
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

    # Definiton of the number of the experiment
    experiment_number = 5
    experiment_name = "Empuje_"

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
        ## Get valalues RC or Desired Signal
        u_ref[:, k] = get_values_rc_f()

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

    ref_drone = get_reference([0, 0, 0, 0], ref_drone)
    send_reference(ref_drone, control_pub)

    # Save information of the entire system
    mdic_h = {"h": h, "label": "experiment_h"}
    mdic_hp = {"hp": hp, "label": "experiment_hp"}
    mdic_hdp = {"hdp": hdp, "label": "experiment_hdp"}
    mdic_rdp = {"rdp": rdp, "label": "experiment_rdp"}
    mdic_u = {"u_ref": u_ref, "label": "experiment_u"}
    mdic_t = {"t": t, "label": "experiment_t"}

    savemat("h_"+ experiment_name + str(experiment_number) + ".mat", mdic_h)
    savemat("hp_"+ experiment_name + str(experiment_number) + ".mat", mdic_hp)
    savemat("hdp_"+ experiment_name + str(experiment_number) + ".mat", mdic_hdp)
    savemat("rdp_"+ experiment_name + str(experiment_number) + ".mat", mdic_rdp)
    savemat("u_ref_" + experiment_name + str(experiment_number) + ".mat", mdic_u)
    savemat("t_"+  experiment_name + str(experiment_number) + ".mat", mdic_t)

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

        main(velocity_publisher)



    except(rospy.ROSInterruptException, KeyboardInterrupt):
        print("Error System")
        ref_drone = TwistStamped()
        ref_drone = get_reference([0, 0, 0, 0], ref_drone)
        send_reference(ref_drone, velocity_publisher)
        pass
    else:
        print("Complete Execution")
        ref_drone = TwistStamped()
        ref_drone = get_reference([0, 0, 0, 0], ref_drone)
        send_reference(ref_drone, velocity_publisher)
        pass