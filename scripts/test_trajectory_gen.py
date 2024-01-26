import copy

import geometry_msgs.msg
import numpy as np
import matplotlib.pyplot as plt

from geometry_msgs.msg import (
    PointStamped,
    Pose,
    PoseStamped,
    PoseWithCovarianceStamped,
)
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
        # calculation of path segments as in https://www.osrobotics.org/osr/planning/time_parameterization.html
        ds_max = max_velocity / self.cumulative_distances[-1]
        dds_max = max_acceleration / self.cumulative_distances[-1]
        self.path_trajectory = PathTrajectory(ds_max, dds_max)

        # check if resulting yaw trajectory is within limits and scale path trajectory accordingly
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
    # rest-to-rest-trajectory consisting of a constant acceleration, a constant velocity and a constant deceleration segment
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
        self.scaling = scaling
        self.key_times = [time * self.scaling for time in self.key_times]
        self.dds_max /= scaling**2
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

def main():
    max_velocity = 2.0
    max_acceleration = 0.1
    max_yaw_velocity = 0.5
    max_yaw_acceleration = 0.2
    path = list()
    pose_stamped = PoseStamped()
    pose_stamped.pose.position.x = 0.1
    pose_stamped.pose.position.y = 0.2
    pose_stamped.pose.position.z = 0.3
    pose_stamped.pose.orientation.w = 1.0
    pose_stamped.pose.orientation.x = 0.0
    pose_stamped.pose.orientation.y = 0.0
    pose_stamped.pose.orientation.z = 0.0
    path.append(copy.deepcopy(pose_stamped))
    pose_stamped.pose.position.x = 0.6
    pose_stamped.pose.position.y = 0.2
    pose_stamped.pose.position.z = 0.5
    pose_stamped.pose.orientation.w = 3 / np.sqrt(2)
    pose_stamped.pose.orientation.x = 0.0
    pose_stamped.pose.orientation.y = 0.0
    pose_stamped.pose.orientation.z = 1 / 2
    path.append(copy.deepcopy(pose_stamped))
    pose_stamped.pose.position.x = 1.1
    pose_stamped.pose.position.y = 0.2
    pose_stamped.pose.position.z = 1.0
    pose_stamped.pose.orientation.w = 1 / np.sqrt(2)
    pose_stamped.pose.orientation.x = 0.0
    pose_stamped.pose.orientation.y = 0.0
    pose_stamped.pose.orientation.z = 1 / np.sqrt(2)
    path.append(copy.deepcopy(pose_stamped))


    trajectory = Trajectory(path, max_velocity, max_acceleration, max_yaw_velocity, max_yaw_acceleration)
    t_samp = np.linspace(0, 20, 2000)
    p_samp = np.zeros((3, len(t_samp)))
    yaw_samp = np.zeros_like(t_samp)
    for i in range(len(t_samp)):
        tmp_position, yaw_samp[i] = trajectory.get_pose(t_samp[i])
        p_samp[:, i] = tmp_position.reshape(-1)

    plt.figure()
    plt.plot(t_samp, p_samp[0, :], label="x")
    plt.plot(t_samp, p_samp[1, :], label="y")
    plt.plot(t_samp, p_samp[2, :], label="z")
    plt.plot(t_samp, yaw_samp, label="yaw")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()