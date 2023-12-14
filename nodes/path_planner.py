#!/usr/bin/env python3

import numpy as np
import rclpy
from final_project.msg import Viewpoint, Viewpoints
from final_project.srv import SetPath
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from visualization_msgs.msg import Marker


class PathSegment:

    def __init__(self):
        self.points = []

        self.has_collision: bool = False

    def world_points(self, grid_size):
        return [[x[0] * grid_size, x[1] * grid_size] for x in self.points]

    def detect_collision(self, occupancy_matrix):
        self.has_collision = False
        for point in self.points:
            if occupancy_matrix[point[1], point[0]] >= 50:
                self.has_collision = True
                break
        return self.has_collision


def occupancy_grid_to_matrix(grid: OccupancyGrid):
    data = np.array(grid.data, dtype=np.uint8)
    data = data.reshape(grid.info.height, grid.info.width)
    return data


def world_to_matrix(x, y, grid_size):
    return [round(x / grid_size), round(y / grid_size)]


def matrix_index_to_world(x, y, grid_size):
    return [x * grid_size, y * grid_size]


def multiple_matrix_index_to_world(points, grid_size):
    world_points = []
    for point in points:
        world_points.append([point[0] * grid_size, point[1] * grid_size])
    return world_points


class PathPlanner(Node):

    def __init__(self):
        super().__init__(node_name='path_planner')
        self.path_marker: Marker
        self.init_path_marker()
        self.path_marker_pub = self.create_publisher(Marker, '~/marker', 1)
        self.viewpoints = []
        self.waypoints = []
        self.orientations = []
        self.occupancy_grid: OccupancyGrid = None
        self.occupancy_matrix: np.ndarray = None
        self.progress = -1.0
        self.remaining_segments = []
        self.set_path_client = self.create_client(SetPath,
                                                  'path_follower/set_path')
        self.grid_map_sub = self.create_subscription(OccupancyGrid,
                                                     'occupancy_grid',
                                                     self.on_occupancy_grid, 1)
        self.viewpoints_sub = self.create_subscription(Viewpoints, 'viewpoints',
                                                       self.on_viewpoints, 1)

    def create_initial_segment(self, viewpoint: Viewpoint):
        p = viewpoint.pose.position
        q = viewpoint.pose.orientation
        self.waypoints = [[p.x, p.y, p.z], [p.x, p.y, p.z]]
        self.orientations = [q, q]
        segment = PathSegment()
        segment.points = [
            world_to_matrix(p.x, p.y, self.occupancy_grid.info.resolution),
            world_to_matrix(p.x, p.y, self.occupancy_grid.info.resolution)
        ]
        return segment

    def on_viewpoints(self, msg: Viewpoints):
        if not self.occupancy_grid:
            return
        start_viewpoint = msg.viewpoints[0]
        if not start_viewpoint.completed and not self.remaining_segments:
            self.remaining_segments = [
                self.create_initial_segment(start_viewpoint),
            ]
            path = self.create_path()
            if not self.set_new_path(path):
                self.get_logger().error('Failed to set new path')
            return
        current_segment = self.remaining_segments[0]
        self.viewpoints = msg.viewpoints
        i = self.find_first_uncompleted_viewpoint(msg)
        if i < 0:
            self.get_logger().warn(
                'If we are not done, something unexpected happend!', once=True)
            return
        if i < 1:
            return
        self.waypoints = []
        self.orientations = []
        self.store_viewpoint(msg.viewpoints[i - 1], self.waypoints,
                             self.orientations)

        self.store_viewpoints(msg.viewpoints[i:])
        self.update_segments()
        if not self.remaining_segments:
            return
        if self.remaining_segments[0].points == current_segment.points:
            return
        path = self.create_path()
        if not path:
            return
        self.set_new_path(path)

    def store_viewpoints(self, viewpoints):
        for viewpoint in viewpoints:
            if viewpoint.completed:
                continue
            self.store_viewpoint(viewpoint, self.waypoints, self.orientations)

    def store_viewpoint(self, viewpoint, waypoints, orientations):
        p = viewpoint.pose.position
        waypoints.append([p.x, p.y, p.z])
        q = viewpoint.pose.orientation
        orientations.append(q)

    def find_first_uncompleted_viewpoint(self, viewpoints: Viewpoints):
        for i, viewpoint in enumerate(viewpoints.viewpoints):
            if not viewpoint.completed:
                return i
        # This should not happen!
        return -1

    def create_path(self):
        if not self.remaining_segments:
            self.get_logger().warn('No remaining segments.')
            return None
        segment: PathSegment = self.remaining_segments[0]
        world_points = segment.world_points(self.occupancy_grid.info.resolution)
        z = self.waypoints[1][2]
        q = self.orientations[1]
        path = Path()
        stamp = self.get_clock().now().to_msg()
        for p_xy in world_points:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = stamp
            pose.pose.position.x = p_xy[0]
            pose.pose.position.y = p_xy[1]
            pose.pose.position.z = z
            pose.pose.orientation = q
            path.poses.append(pose)
        path.header.frame_id = 'map'
        path.header.stamp = stamp
        return path

    def update_segments(self):
        segments = self.compute_path_segments()
        if not segments:
            self.get_logger().warn(
                'Could not create any path segment. '
                'Maybe no waypoints have been received yet.',
                throttle_duration_sec=5.0)
            return
        collision_indices = self.compute_path_collisions(segments)
        collision_detected = bool(collision_indices)

        if not collision_detected:
            self.get_logger().info('Path is collision free :-)')
            self.remaining_segments = segments
        else:
            self.get_logger().info(
                f'Collisions in segments: {collision_indices}')

    def on_occupancy_grid(self, msg: OccupancyGrid):
        self.occupancy_grid = msg
        self.occupancy_matrix = occupancy_grid_to_matrix(self.occupancy_grid)
        self.update_segments()
        if not self.remaining_segments:
            return
        self.publish_path_marker(self.remaining_segments)
        self.occupancy_grid.data = self.occupancy_matrix.flatten()

    def init_path_marker(self):
        msg = Marker()
        msg.action = Marker.ADD
        msg.ns = 'path'
        msg.id = 0
        msg.type = Marker.LINE_STRIP
        msg.header.frame_id = 'map'
        msg.color.a = 1.0
        msg.color.r = 0.0
        msg.color.g = 1.0
        msg.color.b = 0.0
        msg.scale.x = 0.02
        msg.scale.y = 0.02
        msg.scale.z = 0.02
        self.path_marker = msg

    def set_new_path(self, path):
        request = SetPath.Request()
        if not path:
            return False
        request.path = path
        self.set_new_path_future = self.set_path_client.call_async(request)
        return True

    def segments_to_world_points(self, segments: list[PathSegment]):
        world_points = []
        for _i, segment in enumerate(segments):
            world_points.extend(
                segment.world_points(self.occupancy_grid.info.resolution))
        return world_points

    def publish_path_marker(self, segments: list[PathSegment]):
        msg = self.path_marker
        world_points = self.segments_to_world_points(segments)
        msg.points = [Point(x=p[0], y=p[1], z=-0.5) for p in world_points]
        msg.header.stamp = self.get_clock().now().to_msg()
        self.path_marker_pub.publish(msg)

    def get_line_points(self, x0, y0, x1, y1):
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        error = dx + dy

        x = x0
        y = y0
        points = []
        while True:
            points.append([int(x), int(y)])
            if x == x1 and y == y1:
                break
            doubled_error = 2 * error
            if doubled_error >= dy:
                if x == x1:
                    break
                error += dy
                x += sx
            if doubled_error <= dx:
                if y == y1:
                    break
                error += dx
                y += +sy
        return points

    def detect_collision(self, grid_points):
        for pixel in grid_points:
            if self.occupancy_matrix[pixel[1], pixel[0]] >= 50:
                return True
        return False

    def compute_path_segments(self):
        n = len(self.waypoints)
        if n < 2:
            return []
        p0_world = self.waypoints[0]
        prev_point = world_to_matrix(p0_world[0], p0_world[1],
                                     self.occupancy_grid.info.resolution)
        self.get_logger().info(f'Start grid: {prev_point}')
        segments = []
        for i in range(1, n):
            self.get_logger().info(f'Computing path segment {i}/{n-1}')
            point = self.waypoints[i]
            x, y = world_to_matrix(point[0], point[1],
                                   self.occupancy_grid.info.resolution)
            grid_points = self.get_line_points(prev_point[0], prev_point[1], x,
                                               y)
            segment = PathSegment()
            segment.points = grid_points
            segments.append(segment)
            prev_point = [x, y]
        return segments

    def compute_path_collisions(self, segments):
        collision_indices = []
        segment: PathSegment
        for i, segment in enumerate(segments):
            if segment.detect_collision(self.occupancy_matrix):
                collision_indices.append(i)
        return collision_indices


def main():
    rclpy.init()
    node = PathPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
