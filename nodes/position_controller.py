#!/usr/bin/env python3

import math
import numpy as np
import rclpy
from geometry_msgs.msg import Point, PointStamped, PoseWithCovarianceStamped, Vector3Stamped
from rcl_interfaces.msg import SetParametersResult

from hippo_msgs.msg import ActuatorSetpoint
from rclpy.node import Node
from tf_transformations import euler_from_quaternion


class PositionController(Node):

    def __init__(self):
        super().__init__(node_name='position_controller')

        self.setpoint = np.zeros((3, 1))
        self.setpoint_timed_out = True
        self.dsetpoint = np.zeros((3, 1))
        self.last_setpoint_time = self.get_clock().now()

        self.error_integral = np.zeros((3, 1))
        self.got_first_state = False
        self.got_first_setpoint = False
        self.last_time = self.get_clock().now()
        self.last_position = np.zeros((3, 1))
        self.last_derror = np.zeros((3, 1))

        self.declare_parameters(namespace='',
                                parameters=[('gains.p', rclpy.Parameter.Type.DOUBLE),
                                            ('gains.i', rclpy.Parameter.Type.DOUBLE),
                                            ('gains.d', rclpy.Parameter.Type.DOUBLE),
                                            ('filter_gain', rclpy.Parameter.Type.DOUBLE),
                                            ])
        param = self.get_parameter('gains.p')
        self.get_logger().info(f'{param.name}={param.value}')
        self.K_p = param.value * np.eye(3)

        param = self.get_parameter('gains.i')
        self.get_logger().info(f'{param.name}={param.value}')
        self.K_i = param.value * np.eye(3)

        param = self.get_parameter('gains.d')
        self.get_logger().info(f'{param.name}={param.value}')
        self.K_d = param.value * np.eye(3)

        param = self.get_parameter('filter_gain')
        self.get_logger().info(f'{param.name}={param.value}')
        self.alpha = param.value

        self.add_on_set_parameters_callback(self.on_params_changed)

        self.p_component_pub = self.create_publisher(msg_type=Vector3Stamped,
                                                     topic='~/p_component',
                                                     qos_profile=1)

        self.i_component_pub = self.create_publisher(msg_type=Vector3Stamped,
                                                     topic='~/i_component',
                                                     qos_profile=1)

        self.d_component_pub = self.create_publisher(msg_type=Vector3Stamped,
                                                     topic='~/d_component',
                                                     qos_profile=1)

        self.d_component_unfiltered_pub = self.create_publisher(msg_type=Vector3Stamped,
                                                                topic='~/d_component_unfiltered',
                                                                qos_profile=1)

        self.thrust_pub = self.create_publisher(ActuatorSetpoint,
                                                'thrust_setpoint', 1)
        self.position_setpoint_sub = self.create_subscription(
            PointStamped, '~/setpoint', self.on_position_setpoint, 1)
        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped,
                                                 'vision_pose_cov',
                                                 self.on_pose, 1)
        self.setpoint_timeout_timer = self.create_timer(0.5, self.on_setpoint_timeout)
        self.state_timeout_timer = self.create_timer(0.5, self.on_state_timeout)

    def on_params_changed(self, params):
        param: rclpy.Parameter
        for param in params:
            self.get_logger().info(f'Try to set [{param.name}] = {param.value}')
            if param.name == 'gains.p':
                self.K_p = param.value * np.eye(3)
            elif param.name == 'gains.i':
                self.error_integral = np.zeros((3, 1))
                self.K_i = param.value * np.eye(3)
            elif param.name == 'gains.d':
                self.K_d = param.value * np.eye(3)
            elif param.name == 'filter_gain':
                self.alpha = param.value
            else:
                continue
        return SetParametersResult(succesful=True, reason='Parameter set')

    def on_state_timeout(self):
        self.state_timeout_timer.cancel()
        self.get_logger().warn('states timed out. waiting for states.')
        self.last_position = None

    def on_setpoint_timeout(self):
        self.setpoint_timeout_timer.cancel()
        self.get_logger().warn('setpoint timed out. waiting for new setpoints.')
        self.setpoint = None
        self.last_derror = np.zeros((3, 1))
        self.error_integral = np.zeros((3, 1))
        self.setpoint_timed_out = True

    def on_position_setpoint(self, msg: PointStamped):
        self.setpoint_timeout_timer.reset()
        if self.setpoint_timed_out:
            self.get_logger().info('Setpoint received! Getting back to work.')
        if not self.got_first_setpoint:
            self.got_first_setpoint = True
        self.setpoint_timed_out = False
        setpoint = np.array([[msg.point.x], [msg.point.y], [msg.point.z]])
        if self.setpoint is None:
            self.dsetpoint = np.zeros((3, 1))
        else:
            dt = (rclpy.time.Time.from_msg(msg.header.stamp) - self.last_setpoint_time).nanoseconds * 1e-9
            self.dsetpoint = (setpoint - self.setpoint) / max(dt, 1e-6)
        self.setpoint = setpoint
        self.last_setpoint_time = rclpy.time.Time.from_msg(msg.header.stamp)

    def on_pose(self, msg: PoseWithCovarianceStamped):
        self.state_timeout_timer.reset()
        position = np.array(
            [[msg.pose.pose.position.x], [msg.pose.pose.position.y], [msg.pose.pose.position.z]])
        if not self.got_first_state:
            self.got_first_state = True
            self.last_time = rclpy.time.Time.from_msg(msg.header.stamp)
            self.last_position = position
            return
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        timestamp = rclpy.time.Time.from_msg(msg.header.stamp)
        thrust = self.apply_control(position, yaw, timestamp)
        self.publish_thrust(thrust=thrust, timestamp=timestamp)

    def publish_thrust(self, thrust: np.ndarray,
                       timestamp: rclpy.time.Time) -> None:
        msg = ActuatorSetpoint()
        # we want to set the vertical thrust exlusively. mask out xy-components.
        msg.ignore_x = False
        msg.ignore_y = False
        msg.ignore_z = False

        msg.x = thrust[0, 0]
        msg.y = thrust[1, 0]
        msg.z = thrust[2, 0]

        # Let's add a time stamp
        msg.header.stamp = timestamp.to_msg()

        self.thrust_pub.publish(msg)

    def moving_average_filter(self, derror):
        return self.alpha * derror + (1 - self.alpha) * self.last_derror

    def apply_control(self, current_position: np.ndarray, yaw: float, time_now: rclpy.time.Time):
        dt = (time_now - self.last_time).nanoseconds * 1e-9
        if (not self.got_first_setpoint) | (self.last_position is None) | self.setpoint_timed_out | (dt <= 0.0):
            self.last_time = time_now
            self.last_position = current_position
            return np.zeros((3, 1))
        error = self.setpoint - current_position
        self.error_integral += dt * error
        # use of derivatives of position and setpoint instead of deravive of error such that the different time stamps
        # of setpoints and positions are considered
        dposition = (current_position - self.last_position) / max(dt, 1e-6)
        derror = self.moving_average_filter(self.dsetpoint - dposition)
        p_component = self.K_p @ error
        d_component = self.K_d @ derror
        i_component = self.K_i @ self.error_integral
        R = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                      [np.sin(yaw), np.cos(yaw), 0.0],
                      [0.0, 0.0, 1.0]])
        p_component = R.transpose() @ p_component
        d_component = R.transpose() @ d_component
        i_component = R.transpose() @ i_component
        thrust = p_component + d_component + i_component

        debug_msg = Vector3Stamped()
        debug_msg.header.stamp = time_now.to_msg()
        debug_msg.vector.x = p_component[0, 0]
        debug_msg.vector.y = p_component[1, 0]
        debug_msg.vector.z = p_component[2, 0]
        self.p_component_pub.publish(debug_msg)

        debug_msg.vector.x = i_component[0, 0]
        debug_msg.vector.y = i_component[1, 0]
        debug_msg.vector.z = i_component[2, 0]
        self.i_component_pub.publish(debug_msg)

        debug_msg.vector.x = d_component[0, 0]
        debug_msg.vector.y = d_component[1, 0]
        debug_msg.vector.z = d_component[2, 0]
        self.d_component_pub.publish(debug_msg)

        d_component_unfiltered = R.transpose() @ self.K_d @ (self.dsetpoint - dposition)
        debug_msg.vector.x = d_component_unfiltered[0, 0]
        debug_msg.vector.y = d_component_unfiltered[1, 0]
        debug_msg.vector.z = d_component_unfiltered[2, 0]
        self.d_component_unfiltered_pub.publish(debug_msg)

        self.last_time = time_now
        self.last_position = current_position
        self.last_derror = derror
        return thrust


def main():
    rclpy.init()
    node = PositionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
