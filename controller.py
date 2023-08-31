import numpy as np
import time as time
## Reference system
def get_reference(ref, ref_msg):
        ref_msg.twist.linear.x = 0
        ref_msg.twist.linear.y = 0
        ref_msg.twist.linear.z = ref[0]

        ref_msg.twist.angular.x = ref[1]
        ref_msg.twist.angular.y = ref[2]
        ref_msg.twist.angular.z = ref[3]
        return ref_msg

def send_reference(ref_msg, ref_pu):
    ref_pu.publish(ref_msg)
    return None

def init_system(control_pub, ref_drone, velocity_z):
    vx_c = np.array([[0.0], [0.0], [0.0]])
    vy_c = np.array([[0.0], [0.0], [0.0]])
    for k in range(0, 350):
        tic = time.time()
        velocities = get_system_velocity_sensor()
        ## Get Contol Action or nothing
        u_global = controller_z(mass, gravity, [0, 0, velocity_z], velocities[0:3])
        u_ref_pitch, vx_c = controller_attitude_pitch([0, 0, velocity_z], velocities[0:3], ts, vx_c)
        u_ref_roll, vy_c = controller_attitude_roll([0, 0, velocity_z], velocities[0:3], ts, vy_c)
        ref_drone = get_reference([u_global, u_ref_roll, u_ref_pitch, 0], ref_drone)
        send_reference(ref_drone, control_pub)

        # Loop_rate.sleep()
        while (time.time() - tic <= ts):
                None
        toc = time.time() - tic 
        print("Init System " + str(velocity_z))

    
def controller_z(mass, gravity, qdp, qp):
    # Control Gains
    Kp = 10*np.eye(3, 3)
    # Split values
    xp = qp[0]
    yp = qp[1]
    zp = qp[2]

    xdp = qdp[0]
    ydp = qdp[1]
    zdp = qdp[2]

    # Control error
    error = qdp - qp

    error_vector = error.reshape((3,1))

    # Control Law
    aux_control = Kp@error_vector

    # Gravity + compensation velocity
    control_value = mass*gravity + aux_control[2,0]
    
    return control_value