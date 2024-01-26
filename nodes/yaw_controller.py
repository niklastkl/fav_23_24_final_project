#!/usr/bin/env python3
import math

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from hippo_msgs.msg import ActuatorSetpoint, Float64Stamped
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from tf_transformations import euler_from_quaternion


class YawController(Node):

    def __init__(self):
        super().__init__(node_name='yaw_controller')
        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST,
                         depth=1)

        # default value for the yaw setpoint
        self.setpoint = math.pi / 2.0
        self.setpoint_timed_out = True
        self.dsetpoint = 0.0
        self.last_setpoint_time = self.get_clock().now()

        self.error_integral = 0.0
        self.got_first_state = False
        self.got_first_setpoint = False
        self.last_time = self.get_clock().now()
        self.last_yaw = 0.0
        self.last_derror = 0.0

        self.declare_parameters(namespace='',
                                parameters=[('gains.p', rclpy.Parameter.Type.DOUBLE),
                                            ('gains.i', rclpy.Parameter.Type.DOUBLE),
                                            ('gains.d', rclpy.Parameter.Type.DOUBLE),
                                            ('filter_gain', rclpy.Parameter.Type.DOUBLE),
                                            ])
        param = self.get_parameter('gains.p')
        self.get_logger().info(f'{param.name}={param.value}')
        self.K_p = param.value

        param = self.get_parameter('gains.i')
        self.get_logger().info(f'{param.name}={param.value}')
        self.K_i = param.value

        param = self.get_parameter('gains.d')
        self.get_logger().info(f'{param.name}={param.value}')
        self.K_d = param.value

        param = self.get_parameter('filter_gain')
        self.get_logger().info(f'{param.name}={param.value}')
        self.alpha = param.value

        self.add_on_set_parameters_callback(self.on_params_changed)


        self.p_component_pub = self.create_publisher(msg_type=Float64Stamped,
                                                     topic='~/p_component',
                                                     qos_profile=1)

        self.i_component_pub = self.create_publisher(msg_type=Float64Stamped,
                                                     topic='~/i_component',
                                                     qos_profile=1)

        self.d_component_pub = self.create_publisher(msg_type=Float64Stamped,
                                                     topic='~/d_component',
                                                     qos_profile=1)

        self.d_component_unfiltered_pub = self.create_publisher(msg_type=Float64Stamped,
                                                                topic='~/d_component_unfiltered',
                                                                qos_profile=1)

        self.vision_pose_sub = self.create_subscription(
            msg_type=PoseWithCovarianceStamped,
            topic='vision_pose_cov',
            callback=self.on_vision_pose,
            qos_profile=qos)
        self.setpoint_sub = self.create_subscription(Float64Stamped,
                                                     topic='~/setpoint',
                                                     callback=self.on_setpoint,
                                                     qos_profile=qos)

        self.torque_pub = self.create_publisher(msg_type=ActuatorSetpoint,
                                                topic='torque_setpoint',
                                                qos_profile=1)
        self.setpoint_timeout_timer = self.create_timer(0.5, self.on_setpoint_timeout)
        self.state_timeout_timer = self.create_timer(0.5, self.on_state_timeout)

    def on_params_changed(self, params):
        param: rclpy.Parameter
        for param in params:
            self.get_logger().info(f'Try to set [{param.name}] = {param.value}')
            if param.name == 'gains.p':
                self.K_p = param.value
            elif param.name == 'gains.i':
                self.error_integral = 0.0
                self.K_i = param.value
            elif param.name == 'gains.d':
                self.K_d = param.value
            elif param.name == 'filter_gain':
                self.alpha = param.value
            else:
                continue
        return SetParametersResult(succesful=True, reason='Parameter set')

    def on_state_timeout(self):
        self.state_timeout_timer.cancel()
        self.get_logger().warn('yaw state timed out. waiting for states.')
        self.last_yaw = None

    def on_setpoint_timeout(self):
        self.setpoint_timeout_timer.cancel()
        self.get_logger().warn('Setpoint timed out. Waiting for new setpoints')
        self.setpoint = None
        self.last_derror = 0.0
        self.error_integral = 0.0
        self.setpoint_timed_out = True

    def wrap_pi(self, value: float):
        """Normalize the angle to the range [-pi; pi]."""
        if (-math.pi < value) and (value < math.pi):
            return value
        range = 2 * math.pi
        num_wraps = math.floor((value + math.pi) / range)
        return value - range * num_wraps

    def on_setpoint(self, msg: Float64Stamped):
        self.setpoint_timeout_timer.reset()
        if self.setpoint_timed_out:
            self.get_logger().info('Setpoint received! Getting back to work.')
        if not self.got_first_setpoint:
            self.got_first_setpoint = True
        self.setpoint_timed_out = False
        setpoint = self.wrap_pi(msg.data)
        if self.setpoint is None:
            self.dsetpoint = 0.0
        else:
            dt = (rclpy.time.Time.from_msg(msg.header.stamp) - self.last_setpoint_time).nanoseconds * 1e-9
            self.dsetpoint = self.wrap_pi(setpoint - self.setpoint) / max(dt, 1e-6)
        self.setpoint = setpoint
        self.last_setpoint_time = rclpy.time.Time.from_msg(msg.header.stamp)

    def on_vision_pose(self, msg: PoseWithCovarianceStamped):
        self.state_timeout_timer.reset()
        # get the vehicle orientation expressed as quaternion
        q = msg.pose.pose.orientation
        # convert the quaternion to euler angles
        (roll, pitch, yaw) = euler_from_quaternion([q.x, q.y, q.z, q.w])
        #yaw = self.wrap_pi(yaw)
        if not self.got_first_state:
            self.got_first_state = True
            self.last_time = rclpy.time.Time.from_msg(msg.header.stamp)
            self.last_yaw = yaw
            return

        timestamp = rclpy.time.Time.from_msg(msg.header.stamp)
        control_output = self.compute_control_output(yaw, timestamp)
        self.publish_control_output(control_output, timestamp)

    def moving_average_filter(self, derror):
        return self.alpha * derror + (1 - self.alpha) * self.last_derror

    def compute_control_output(self, yaw: float, time_now: rclpy.time.Time):
        dt = (time_now - self.last_time).nanoseconds * 1e-9
        if (not self.got_first_setpoint) | (self.last_yaw is None) | self.setpoint_timed_out | (dt <= 0.0):
            self.last_time = time_now
            self.last_yaw = yaw
            return 0.0
        # very important: normalize the angle error!
        error = self.wrap_pi(self.setpoint - yaw)
        self.error_integral += dt * error
        dyaw = self.wrap_pi(yaw - self.last_yaw) / max(dt, 1e-6)
        derror = self.moving_average_filter(self.dsetpoint - dyaw)
        p_component = self.K_p * error
        d_component = self.K_d * derror
        i_component = self.K_i * self.error_integral
        torque = p_component + d_component + i_component

        debug_msg = Float64Stamped()
        debug_msg.header.stamp = time_now.to_msg()
        debug_msg.data = p_component
        self.p_component_pub.publish(debug_msg)

        debug_msg.data = i_component
        self.i_component_pub.publish(debug_msg)

        debug_msg.data = d_component
        self.d_component_pub.publish(debug_msg)

        d_component_unfiltered = self.K_d * (self.dsetpoint - dyaw)
        debug_msg.data = d_component_unfiltered
        self.d_component_unfiltered_pub.publish(debug_msg)

        self.last_time = time_now
        self.last_yaw = yaw
        self.last_derror = derror

        return torque

    def publish_control_output(self, control_output: float,
                               timestamp: rclpy.time.Time):
        msg = ActuatorSetpoint()
        msg.header.stamp = timestamp.to_msg()
        msg.ignore_x = True
        msg.ignore_y = True
        msg.ignore_z = False  # yaw is the rotation around the vehicle's z axis

        msg.z = control_output
        self.torque_pub.publish(msg)


def main():
    rclpy.init()
    node = YawController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
