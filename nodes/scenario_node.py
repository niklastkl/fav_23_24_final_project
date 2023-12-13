#!/usr/bin/env python3
import math
import os

import rclpy
import yaml
from ament_index_python.packages import get_package_share_path
from final_project.msg import PolygonsStamped, Viewpoint, Viewpoints
from geometry_msgs.msg import (
    Point,
    Point32,
    Polygon,
    Pose,
    PoseWithCovarianceStamped,
)
from rclpy.node import Node
from tf_transformations import euler_from_quaternion
from visualization_msgs.msg import Marker, MarkerArray


class ScenarioNode(Node):

    def __init__(self):
        super().__init__(node_name='scenario_node')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('scenario', rclpy.Parameter.Type.INTEGER),
                ('viewpoints', rclpy.Parameter.Type.INTEGER),
            ],
        )
        param = self.get_parameter('scenario')
        self.scenario = param.value
        self.get_logger().info(f'Using scenario #{self.scenario}.')
        self.read_scenario_description()
        self.read_viewpoints()

        self.obstacles_pub = self.create_publisher(msg_type=PolygonsStamped,
                                                   topic='obstacles',
                                                   qos_profile=1)
        self.viewpoints_pub = self.create_publisher(Viewpoints, 'viewpoints', 1)

        self.marker_pub = self.create_publisher(msg_type=MarkerArray,
                                                topic='markers',
                                                qos_profile=1)
        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped,
                                                 'vision_pose_cov',
                                                 self.on_pose, 1)
        self.timer = self.create_timer(1.0 / 50, self.on_timer)
        self.finished_viewpoints = False

    def on_pose(self, msg: PoseWithCovarianceStamped):
        pose = msg.pose.pose
        i = self.find_viewpoint_in_tolerance(pose)
        # nothing to do if no viewpoint in tolerance margin
        if i < 0:
            return
        if self.is_viewpoint_completed(i, self.viewpoints.viewpoints[i], pose):
            self.viewpoints.viewpoints[i].completed = True
            self.get_logger().info(
                f'Viewpoint[{i}] completed. Uncompleted waypoints: '
                f'{self.uncompleted_viewpoints}')
        uncompleted_waypoints = self.uncompleted_viewpoints()
        if not uncompleted_waypoints:
            self.finished_viewpoints = True

    def wrap_pi(self, value: float):
        """Normalize the angle to the range [-pi; pi]."""
        if (-math.pi < value) and (value < math.pi):
            return value
        range = 2 * math.pi
        num_wraps = math.floor((value + math.pi) / range)
        return value - range * num_wraps

    def are_all_viewpoints_completed(self):
        x = [True for x in self.viewpoints.viewpoints if not x.completed]
        return not bool(x)

    def uncompleted_viewpoints(self):
        return [
            i for i, x in enumerate(self.viewpoints.viewpoints) if x.completed
        ]

    def find_viewpoint_in_tolerance(self, pose: Pose):
        yaw_tolerance = 0.1
        position_tolerance = 0.1
        p = pose.position
        q = pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        for i, viewpoint in enumerate(self.viewpoints.viewpoints):
            # do not consider already completed viewpoints
            if viewpoint.completed:
                continue
            p_target = viewpoint.position
            q = viewpoint.orientation
            _, _, yaw_target = euler_from_quaternion([q.x, q.y, q.z, q.w])
            d = math.sqrt((p.x - p_target.y)**2 + (p.y - p_target.y)**2 +
                          (p.z - p_target.z)**2)
            yaw_error = self.wrap_pi(abs(yaw - yaw_target))
            if d < position_tolerance and yaw_error < yaw_tolerance:
                return i
        return -1

    def is_viewpoint_completed(self, i, viewpoint: Viewpoint, pose: Pose):
        return True

    def on_timer(self):
        wall_thickness = 0.3
        y_pos = 2.0
        x_pos = 0.0

        corners_1 = [(x_pos, y_pos), (x_pos + 1.77, 2.0),
                     (x_pos + 1.0, y_pos + wall_thickness),
                     (x_pos, y_pos + wall_thickness)]
        obstacle_list = [corners_1]

        polygons_msg = PolygonsStamped()
        polygons_msg.header.stamp = self.get_clock().now().to_msg()
        polygons_msg.header.frame_id = 'map'

        marker_array_msg = MarkerArray()

        for obstacle in obstacle_list:

            polygon_i = Polygon()
            marker_i = Marker()

            for point in obstacle:
                point_i = Point32()
                point_i.x = point[0]
                point_i.y = point[1]
                polygon_i.points.append(point_i)

                # marker msg wants Point() instead of Point32() ....
                marker_point = Point()
                marker_point.x = point[0]
                marker_point.y = point[1]
                marker_i.points.append(marker_point)

            marker_i.points.append(marker_i.points[0])

            marker_i.header.frame_id = 'map'
            marker_i.header.stamp = self.get_clock().now().to_msg()
            marker_i.type = Marker.LINE_STRIP
            marker_i.color.r = 1.0
            marker_i.color.a = 1.0
            marker_i.scale.x = 0.02
            # marker_i.scale.y = 0.02
            marker_array_msg.markers.append(marker_i)

            polygons_msg.polygons.append(polygon_i)

        self.obstacles_pub.publish(polygons_msg)
        self.marker_pub.publish(marker_array_msg)
        self.viewpoints_pub.publish(self.viewpoints)

    def read_scenario_description(self):
        filepath = os.path.join(get_package_share_path('final_project'),
                                f'config/scenario_{self.scenario}.yaml')
        self.get_logger().info(filepath)

        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
            for obstacle in data['obstacles']:
                thickness = obstacle['thickness']
                self.get_logger().info(f"Wall thickness: {thickness}")

    def read_viewpoints(self):
        viewpoints = self.get_parameter('viewpoints').value
        path = os.path.join(get_package_share_path('final_project'),
                            f'config/viewpoints_{viewpoints}.yaml')
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
            self.viewpoints = Viewpoints()
            for waypoint in data['viewpoints']:
                pose = Pose()
                pose.position.x = waypoint['x']
                pose.position.y = waypoint['y']
                pose.position.z = waypoint['z']
                pose.orientation.w = waypoint['qw']
                pose.orientation.x = waypoint['qx']
                pose.orientation.y = waypoint['qy']
                pose.orientation.z = waypoint['qz']
                viewpoint = Viewpoint()
                viewpoint.pose = pose
                viewpoint.completed = False
                self.viewpoints.viewpoints.append(viewpoint)


def main():
    rclpy.init()
    node = ScenarioNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
