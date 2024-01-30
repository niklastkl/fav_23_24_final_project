#!/usr/bin/env python3
import geometry_msgs.msg
import numpy as np
import rclpy
from geometry_msgs.msg import (
    PointStamped,
    Pose,
    PoseStamped,
    PoseWithCovarianceStamped,
)
from hippo_msgs.msg import Float64Stamped
from nav_msgs.msg import Path
from rclpy.node import Node
from scenario_msgs.srv import SetPath
from std_srvs.srv import Trigger
from tf_transformations import euler_from_quaternion


def distance(p0: Pose, p1: Pose):
    return np.sqrt((p1.position.x - p0.position.x) ** 2 + (p1.position.y - p0.position.y) ** 2 + (
            p1.position.z - p0.position.z) ** 2)


class Trajectory:
    def __init__(self, path: [PoseStamped], max_velocity, max_acceleration, max_yaw_velocity, max_yaw_acceleration):
        # for the positions, assign time stamps to the positions such that a "smooth" trajectory from rest to rest is generated
        self.positions = [np.array([[path[i].pose.position.x], [path[i].pose.position.y], [path[i].pose.position.z]])
                          for
                          i in range(len(path))]

        # for yaw, only take first and last orientation, as intermediate frames don't matter (not important for obstacles)
        # a yaw-trajectory is then generated, considering maximum allowed angular velocity / angular acceleration such that path
        # and yaw progress towards goal pose are synchronized
        self.yawbounds = [0.0, 0.0]
        q = path[0].pose.orientation
        _, _, self.yawbounds[0] = euler_from_quaternion([q.x, q.y, q.z, q.w])
        q = path[-1].pose.orientation
        _, _, self.yawbounds[1] = euler_from_quaternion([q.x, q.y, q.z, q.w])

        mutual_distances = [distance(path[i].pose, path[i + 1].pose) for i in range(len(path) - 1)]
        self.cumulative_distances = [0.0] + [sum(mutual_distances[0:i + 1]) for i in range(len(mutual_distances))]
        # calculation of path segments as in https://www.osrobotics.org/osr/planning/time_parameterization.html,
        # a setpoint on the trajectory is then calculated from the progress of distance along the scalar path trajectory
        # and interpolated between key frames
        ds_max = max_velocity / self.cumulative_distances[-1]
        dds_max = max_acceleration / self.cumulative_distances[-1]
        self.path_trajectory = PathTrajectory(ds_max, dds_max)

        # check if resulting yaw trajectory is within limits and scale path trajectory accordingly,
        # depending on bounds on velocity and acceleration determine the number of path segments
        time_scaling = 1.0
        path_keypoints = [self.path_trajectory.path_variable(t) for t in self.path_trajectory.key_times]
        yaw_keypoints = [keypoint * (self.yawbounds[1] - self.yawbounds[0]) + self.yawbounds[0] for keypoint in
                         path_keypoints]
        if len(path_keypoints) == 3:
            acceleration = 2 * (yaw_keypoints[0] - self.yawbounds[0]) / np.power(self.path_trajectory.key_times[0], 2)
            time_scaling = max(time_scaling, np.sqrt(abs(acceleration / max_yaw_acceleration)))
            velocity = (yaw_keypoints[1] - yaw_keypoints[0]) / (
                    self.path_trajectory.key_times[1] - self.path_trajectory.key_times[0])
            time_scaling = max(time_scaling, abs(velocity / max_yaw_velocity))
            deceleration = abs(2 * (yaw_keypoints[2] - yaw_keypoints[1]) / np.power(
                self.path_trajectory.key_times[2] - self.path_trajectory.key_times[1], 2))
            time_scaling = max(time_scaling, np.sqrt(abs(deceleration / max_yaw_acceleration)))
        self.path_trajectory.set_scaling(time_scaling)

    def get_pose(self, t):
        path_variable = self.path_trajectory.path_variable(t)
        idx = 0
        while True:
            if path_variable * self.cumulative_distances[-1] <= self.cumulative_distances[idx]:
                break

            idx += 1
            if idx >= len(self.cumulative_distances):
                return self.positions[-1], self.yawbounds[-1]
        if idx == 0:
            return self.positions[0], self.yawbounds[0]

        position = self.positions[idx - 1] + (self.positions[idx] - self.positions[idx - 1]) * (
                path_variable * self.cumulative_distances[-1] - self.cumulative_distances[idx - 1]) / (
                           self.cumulative_distances[idx] - self.cumulative_distances[idx - 1])
        yaw = self.yawbounds[0] + path_variable * (self.yawbounds[1] - self.yawbounds[0])
        return position, yaw


class PathTrajectory:
    # scalar rest-to-rest-trajectory consisting of a constant acceleration, a constant velocity and a constant deceleration segment
    # https://www.osrobotics.org/osr/planning/time_parameterization.html
    def __init__(self, ds_max, dds_max):
        self.ds_max = ds_max
        self.dds_max = dds_max
        if ds_max >= np.sqrt(dds_max):  # acceleration up to time stamp ts, deceleration from ts to 2*ts
            ts = 1.0 / np.sqrt(dds_max)
            T = 2 * ts
            self.key_times = [ts, T]
        else:
            t0 = ds_max / dds_max
            t1 = 1.0 / ds_max - ds_max / dds_max
            T = 2 * t0 + t1
            self.key_times = [t0, t0 + t1, T]
        self.scaling = 1.0

    def set_scaling(self, scaling):
        # scale trajectory in time
        self.scaling = scaling
        self.key_times = [time * self.scaling for time in self.key_times]
        self.dds_max /= scaling ** 2
        self.ds_max /= scaling

    def path_variable(self, t):
        if t < 0:
            return 0
        elif t < self.key_times[0]:
            return 1 / 2.0 * self.dds_max * t ** 2
        elif t > self.key_times[-1]:  # trajectory duration exceeded
            return self.path_variable(self.key_times[-1])

        if len(self.key_times) == 2:
            return self.dds_max * self.key_times[0] ** 2 - 1 / 2.0 * self.dds_max * (self.key_times[1] - t) ** 2
        else:  # acceleration up to time stamp t0, constant velocity between t0 and t0 + t1, deceleration up to 2 * t0 + t1
            if t < self.key_times[1]:
                return 1 / 2.0 * self.dds_max * self.key_times[0] ** 2 + self.ds_max * (t - self.key_times[0])
            else:
                return self.dds_max * self.key_times[0] ** 2 - 1 / 2.0 * self.dds_max * (
                        self.key_times[2] - t) ** 2 + self.ds_max * (self.key_times[1] - self.key_times[0])


class PathFollower(Node):

    def __init__(self):
        super().__init__(node_name='path_follower')
        self.look_ahead_distance = 0.3
        self.target_index = -1
        self.path: list[PoseStamped] = None
        self.trajectory = None
        self.start_time = None

        self.max_velocity = 1e-6
        self.max_acceleration = 1e-6
        self.max_yaw_velocity = 1e-6
        self.max_yaw_acceleration = 1e-6

        self.declare_parameters(namespace='',
                                parameters=[('max_velocity', rclpy.Parameter.Type.DOUBLE),
                                            ('max_acceleration', rclpy.Parameter.Type.DOUBLE),
                                            ('max_yaw_velocity', rclpy.Parameter.Type.DOUBLE),
                                            ('max_yaw_acceleration', rclpy.Parameter.Type.DOUBLE),
                                            ])
        param = self.get_parameter('max_velocity')
        self.get_logger().info(f'{param.name}={param.value}')
        self.max_velocity = param.value

        param = self.get_parameter('max_acceleration')
        self.get_logger().info(f'{param.name}={param.value}')
        self.max_acceleration = param.value

        param = self.get_parameter('max_yaw_velocity')
        self.get_logger().info(f'{param.name}={param.value}')
        self.max_yaw_velocity = param.value

        param = self.get_parameter('max_yaw_acceleration')
        self.get_logger().info(f'{param.name}={param.value}')
        self.max_yaw_acceleration = param.value

        self.init_services()
        self.yaw_pub = self.create_publisher(Float64Stamped,
                                             'yaw_controller/setpoint', 1)
        self.position_pub = self.create_publisher(
            PointStamped, 'position_controller/setpoint', 1)
        self.path_pub = self.create_publisher(Path, '~/current_path', 1)
        self.look_ahead_distance = 0.3
        frequency = float(50)
        self.trajectory_timer = self.create_timer(1 / frequency, self.send_setpoint)

    def init_services(self):
        self.set_path_service = self.create_service(SetPath, '~/set_path',
                                                    self.serve_set_path)
        self.path_finished_service = self.create_service(
            Trigger, '~/path_finished', self.serve_path_finished)

    def serve_set_path(self, request, response):
        self.path = request.path.poses
        self.trajectory = None
        self.target_index = 0
        self.get_logger().info(
            f'New path with {len(self.path)} poses has been set.')
        response.success = True
        return response

    def serve_path_finished(self, request, response: Trigger.Response):
        self.path = None
        self.trajectory = None
        response.success = True
        self.get_logger().info('Path finished. Going to idle mode.')
        return response

    def send_setpoint(self):
        if not self.path:
            return
        if not self.update_setpoint():
            return
        stamp = self.get_clock().now().to_msg()

        msg = Float64Stamped()
        msg.data = self.target_yaw
        msg.header.stamp = stamp
        msg.header.frame_id = 'map'
        self.yaw_pub.publish(msg)

        position = geometry_msgs.msg.Point(x=self.target_position[0, 0], y=self.target_position[1, 0],
                                           z=self.target_position[2, 0])

        msg = PointStamped(header=msg.header, point=position)
        self.position_pub.publish(msg)
        if self.path:
            msg = Path()
            msg.header.frame_id = 'map'
            msg.header.stamp = stamp
            msg.poses = self.path
            self.path_pub.publish(msg)

    def update_setpoint(self):
        if not self.path:
            self.target_index = 0
            self.target_position = None
            self.target_yaw = None
            return False
        if not self.trajectory:
            self.trajectory = Trajectory(self.path, self.max_velocity, self.max_acceleration, self.max_yaw_velocity,
                                         self.max_yaw_acceleration)
            self.start_time = self.get_clock().now()

        t = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
        self.target_position, self.target_yaw = self.trajectory.get_pose(t)
        return True


def main():
    rclpy.init()
    node = PathFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
