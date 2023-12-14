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
from std_srvs.srv import Trigger
from tf_transformations import euler_from_quaternion
from visualization_msgs.msg import Marker, MarkerArray

DEPTH = -0.5


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
        self.polygons: list[Polygon] = None
        self.scenario = param.value
        self.get_logger().info(f'Using scenario #{self.scenario}.')
        self.read_scenario_description()
        self.read_viewpoints()

        self.obstacles_pub = self.create_publisher(msg_type=PolygonsStamped,
                                                   topic='obstacles',
                                                   qos_profile=1)
        self.viewpoints_pub = self.create_publisher(Viewpoints, 'viewpoints', 1)

        self.marker_pub = self.create_publisher(msg_type=MarkerArray,
                                                topic='~/marker_array',
                                                qos_profile=1)
        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped,
                                                 'vision_pose_cov',
                                                 self.on_pose, 1)
        self.start_service = self.create_service(Trigger, '~/start',
                                                 self.serve_start)
        self.reset_service = self.create_service(Trigger, '~/reset',
                                                 self.serve_reset)
        self.timer = self.create_timer(1.0 / 50, self.on_timer)
        self.finished_viewpoints = False
        self.previous_close_viewpoint = {'index': -1, 'count': 0}
        self.running = False
        self.t_start: rclpy.time.Time
        self.start_position_reached = False

    def reset(self):
        for viewpoint in self.viewpoints.viewpoints:
            viewpoint.completed = False
        self.previous_close_viewpoint = {'index': -1, 'count': 0}
        self.start_position_reached = False
        self.running = False

    def serve_start(self, request, response: Trigger.Response):
        if self.running:
            response.success = False
            response.message = "Already running."
            return response
        self.reset()
        response.success = True
        self.running = True
        response.message = "Started."
        return response

    def serve_reset(self, request, response: Trigger.Response):
        self.reset()
        response.success = True
        response.message = "Reset"
        return response

    def on_pose(self, msg: PoseWithCovarianceStamped):
        pose = msg.pose.pose
        i = self.find_viewpoint_in_tolerance(pose)
        # nothing to do if no viewpoint in tolerance margin
        if i < 0:
            self.previous_close_viewpoint['index'] = -1
            self.previous_close_viewpoint['count'] = 0
            return
        if self.is_viewpoint_completed(i, self.viewpoints.viewpoints[i], pose):
            self.viewpoints.viewpoints[i].completed = True
            self.get_logger().info(
                f'Viewpoint[{i}] completed. Uncompleted waypoints: '
                f'{self.uncompleted_viewpoints()}')
        uncompleted_waypoints = self.uncompleted_viewpoints()
        if not uncompleted_waypoints:
            self.finished_viewpoints = True
            self.running = False
            t = (self.get_clock().now() - self.t_start).nanoseconds * 1e-9
            self.get_logger().info(f'Finished after {t:.2f}s')

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
            i for i, x in enumerate(self.viewpoints.viewpoints)
            if not x.completed
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
            p_target = viewpoint.pose.position
            q = viewpoint.pose.orientation
            _, _, yaw_target = euler_from_quaternion([q.x, q.y, q.z, q.w])
            d = math.sqrt((p.x - p_target.x)**2 + (p.y - p_target.y)**2)
            yaw_error = abs(self.wrap_pi(abs(yaw - yaw_target)))
            if d < position_tolerance and yaw_error < yaw_tolerance:
                return i
        return -1

    def is_viewpoint_completed(self, i, viewpoint: Viewpoint, pose: Pose):
        # ignore waypoints until start position has been reached
        if not self.start_position_reached:
            if i != 0:
                return
        if self.previous_close_viewpoint['index'] != i:
            self.previous_close_viewpoint['count'] = 0
        self.previous_close_viewpoint['index'] = i
        self.previous_close_viewpoint['count'] += 1
        completed = self.previous_close_viewpoint['count'] >= 40
        if completed and not self.start_position_reached:
            self.start_position_reached = True
            self.t_start = self.get_clock().now()
        return completed

    def publish_marker_array(self, markers):
        self.marker_pub.publish(MarkerArray(markers=markers))

    def create_polygon_markers(self, polygons):
        stamp = self.get_clock().now().to_msg()
        markers = []
        for i, polygon in enumerate(polygons):
            marker = Marker()
            for p in polygon.points:
                marker.points.append(Point(x=p.x, y=p.y, z=p.z))
            marker.points.append(marker.points[0])
            marker.type = Marker.LINE_STRIP
            if not self.running:
                marker.action = Marker.DELETEALL
            else:
                marker.action = Marker.ADD
            marker.id = i
            marker.color.r = 1.0
            marker.color.a = 1.0
            marker.scale.x = 0.02
            marker.scale.y = 0.02
            marker.scale.z = 0.02
            marker.header.stamp = stamp
            marker.header.frame_id = 'map'
            marker.ns = 'obstacles'
            markers.append(marker)
        return markers

    def create_viewpoint_marker(self):
        markers = []
        viewpoint: Viewpoint
        stamp = self.get_clock().now().to_msg()
        for i, viewpoint in enumerate(self.viewpoints.viewpoints):
            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = 'map'
            marker.pose = viewpoint.pose
            if not self.running:
                marker.action = Marker.DELETEALL
            else:
                marker.action = Marker.ADD
            marker.type = Marker.ARROW
            marker.id = i
            marker.ns = 'viewpoints'
            marker.color.a = 1.0
            marker.color.g = 1.0 * viewpoint.completed
            marker.color.r = 1.0 - viewpoint.completed
            marker.scale.x = 0.3
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            markers.append(marker)
        return markers

    def on_timer(self):
        polygons_msg = PolygonsStamped()
        polygons_msg.header.stamp = self.get_clock().now().to_msg()
        polygons_msg.header.frame_id = 'map'
        polygons_msg.polygons = self.polygons

        if self.running:
            self.obstacles_pub.publish(polygons_msg)
            self.viewpoints_pub.publish(self.viewpoints)
        markers = self.create_polygon_markers(self.polygons)
        markers.extend(self.create_viewpoint_marker())
        self.publish_marker_array(markers)

    def read_scenario_description(self):
        filepath = os.path.join(get_package_share_path('final_project'),
                                f'config/scenario_{self.scenario}.yaml')
        self.get_logger().info(filepath)

        def obstacle_to_polygon(obstacle):
            polygon = Polygon()
            for corner in obstacle['corners']:
                p = Point32(x=corner[0], y=corner[1], z=DEPTH)
                polygon.points.append(p)
            return polygon

        polygons = []
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
            for obstacle in data['obstacles']:
                polygon = obstacle_to_polygon(obstacle)
                polygons.append(polygon)
        self.polygons = polygons

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
