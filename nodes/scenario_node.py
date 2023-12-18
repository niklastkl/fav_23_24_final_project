#!/usr/bin/env python3
import math
import os
from collections import deque
from statistics import mean

import rclpy
import yaml
from ament_index_python.packages import get_package_share_path
from final_project.msg import PolygonsStamped, Viewpoint, Viewpoints
from final_project.srv import MoveToStart
from geometry_msgs.msg import (
    Point,
    Point32,
    Polygon,
    Pose,
    PoseWithCovarianceStamped,
)
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rviz_2d_overlay_msgs.msg import OverlayText
from std_msgs.msg import Float32
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
        self.init_clients()
        self.init_services()
        self.viewpoint_in_tolerance_index = -1
        self.completed_queues = [
            deque([0.0] * 100, maxlen=100) for _ in self.viewpoints.viewpoints
        ]

        self.obstacles_pub = self.create_publisher(msg_type=PolygonsStamped,
                                                   topic='obstacles',
                                                   qos_profile=1)
        self.viewpoints_pub = self.create_publisher(Viewpoints, 'viewpoints', 1)
        self.gauge_pub = self.create_publisher(Float32, '~/viewpoint_progress',
                                               1)
        self.rviz_overlay_status_pub = self.create_publisher(
            OverlayText, '~/overlay_status_text', 1)
        self.rviz_t_finish_pub = self.create_publisher(
            OverlayText, '~/overlay_t_finish_text', 1)

        self.marker_pub = self.create_publisher(msg_type=MarkerArray,
                                                topic='~/marker_array',
                                                qos_profile=1)
        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped,
                                                 'vision_pose_cov',
                                                 self.on_pose, 1)
        self.timer = self.create_timer(1.0 / 50, self.on_timer)
        self.finished_viewpoints = False
        self.previous_close_viewpoint = {'index': -1, 'count': 0}
        self.running = False
        self.t_start: rclpy.time.Time
        self.start_position_reached = False
        self.vehicle_pose = Pose()

    def init_services(self):
        self.start_service = self.create_service(Trigger, '~/start',
                                                 self.serve_start)
        self.reset_service = self.create_service(Trigger, '~/reset',
                                                 self.serve_reset)

    def init_clients(self):
        # require separate cb group to make synchronous service calls
        cb_group = rclpy.callback_groups.MutuallyExclusiveCallbackGroup()
        self.start_path_planner_client = self.create_client(
            Trigger, 'path_planner/start', callback_group=cb_group)
        self.move_to_start_client = self.create_client(
            MoveToStart, 'path_planner/move_to_start', callback_group=cb_group)
        self.stop_path_planner_client = self.create_client(
            Trigger, 'path_planner/stop', callback_group=cb_group)

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
        req = MoveToStart.Request()
        req.current_pose = self.vehicle_pose
        req.target_pose = self.viewpoints.viewpoints[0].pose
        self.move_to_start_client.call(req)
        self.reset()
        response.success = True
        self.running = True
        response.message = "Started."
        return response

    def serve_reset(self, request, response: Trigger.Response):
        self.reset()
        self.stop_path_planner_client.call(Trigger.Request())
        response.success = True
        response.message = "Reset"
        return response

    def on_pose(self, msg: PoseWithCovarianceStamped):
        pose = msg.pose.pose
        self.vehicle_pose = pose
        i = self.find_viewpoint_in_tolerance(pose)
        self.viewpoint_in_tolerance_index = i
        # nothing to do if no viewpoint in tolerance margin
        for j, queue in enumerate(self.completed_queues):
            if j == i:
                queue.append(1.0)
            else:
                queue.append(0.0)
        if i < 0:
            self.previous_close_viewpoint['index'] = -1
            self.previous_close_viewpoint['count'] = 0
            self.gauge_pub.publish(Float32(data=0.0))
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
            msg = OverlayText()
            msg.horizontal_alignment = msg.LEFT
            msg.horizontal_distance = 10
            msg.vertical_alignment = msg.TOP
            msg.vertical_distance = 70
            msg.fg_color.a = 1.0
            msg.fg_color.g = 1.0
            msg.line_width = 10
            msg.text_size = 28.0
            msg.width = 1000
            msg.height = 50
            msg.text = f'Finished after {t:.2f}s'
            self.rviz_t_finish_pub.publish(msg)
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
        self.gauge_pub.publish(
            Float32(data=mean(self.completed_queues[i]) / 0.8))
        completed = self.previous_close_viewpoint['count'] >= 40
        completed = mean(self.completed_queues[i]) > 0.8
        if completed and not self.start_position_reached:
            self.start_position_reached = True
            self.t_start = self.get_clock().now()
            self.start_path_planner_client.call(Trigger.Request())
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
            marker.color.r = 1.0 - viewpoint.completed
            marker.color.g = 1.0 * viewpoint.completed
            marker.scale.x = 0.3
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            markers.append(marker)
        i = self.viewpoint_in_tolerance_index
        if i >= 0 and not self.viewpoints.viewpoints[i].completed:
            markers[i].color.r = 1.0
            markers[i].color.g = 1.0
            markers[i].color.b = 0.0

        return markers

    def on_timer(self):
        polygons_msg = PolygonsStamped()
        polygons_msg.header.stamp = self.get_clock().now().to_msg()
        polygons_msg.header.frame_id = 'map'
        polygons_msg.polygons = self.polygons

        status_msg = OverlayText()
        status_msg.action = status_msg.ADD
        status_msg.horizontal_alignment = status_msg.LEFT
        status_msg.horizontal_distance = 10
        status_msg.vertical_alignment = status_msg.TOP
        status_msg.vertical_distance = 10
        status_msg.fg_color.a = 1.0
        status_msg.fg_color.r = 1.0
        status_msg.line_width = 10
        status_msg.text_size = 28.0
        status_msg.width = 1000
        status_msg.height = 50
        status_msg.text = "Not running"
        if self.running:
            t_finish_msg = OverlayText()
            t_finish_msg.action = t_finish_msg.DELETE
            t_finish_msg.width = 1
            t_finish_msg.height = 1
            self.rviz_t_finish_pub.publish(t_finish_msg)
            status_msg.fg_color.r = 1.0
            status_msg.fg_color.g = 1.0
            status_msg.fg_color.b = 0.0
            self.obstacles_pub.publish(polygons_msg)
            self.viewpoints_pub.publish(self.viewpoints)
            status_msg.text = 'Moving to start position.'

            if self.start_position_reached:
                t = (self.get_clock().now() - self.t_start).nanoseconds * 1e-9
                status_msg.text = f'Time: {t:.2f}s'
        self.rviz_overlay_status_pub.publish(status_msg)

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
    exec = MultiThreadedExecutor()
    exec.add_node(node)
    try:
        exec.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
