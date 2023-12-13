#!/usr/bin/env python3
import numpy as np
import rclpy
from geometry_msgs.msg import Point, Pose
from nav_msgs.msg import MapMetaData, OccupancyGrid
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

GRID_SIZE = 0.05
TANK_WIDTH = 2.0


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
        self.test_grid_pub = self.create_publisher(OccupancyGrid, 'test_grid',
                                                   1)
        self.path_marker: Marker
        self.init_path_marker()
        self.path_marker_pub = self.create_publisher(Marker, '~/marker', 10)
        self.waypoints = [
            [1.5, 0.5],
            [0.5, 0.75],
            [0.5, 3.25],
            [1.5, 3.5],
        ]
        self.grid_map_sub = self.create_subscription(OccupancyGrid,
                                                     'occupancy_grid',
                                                     self.on_occupancy_grid, 1)
        self.occupancy_grid = self.create_test_grid()
        self.occupancy_matrix = occupancy_grid_to_matrix(self.occupancy_grid)
        self.timer = self.create_timer(0.5, self.on_timer)

    def on_occupancy_grid(self, msg: OccupancyGrid):
        self.occupancy_grid = msg
        self.occupancy_matrix = occupancy_grid_to_matrix(self.occupancy_grid)

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

    def create_test_grid(self):
        grid = OccupancyGrid()

        info = MapMetaData()
        info.resolution = GRID_SIZE
        info.width = int(TANK_WIDTH / info.resolution)
        info.height = int(2 * info.width)
        pose = Pose()
        info.origin = pose

        grid.info = info
        data = np.zeros([info.height, info.width], dtype=np.uint8)
        data[int(info.height * 0.25):int(info.height * 0.75),
             int(info.width * 0.4):int(info.width * 0.6)] = 100
        grid.data = data.flatten()
        grid.header.frame_id = 'map'
        return grid

    def segments_to_world_points(self, segments: list[PathSegment]):
        world_points = []
        for _i, segment in enumerate(segments):
            world_points.extend(
                segment.world_points(self.occupancy_grid.info.resolution))
        return world_points

    def publish_path_marker(self, segments: list[PathSegment]):
        msg = self.path_marker
        msg.points = []
        msg.colors = []
        world_points = self.segments_to_world_points(segments)
        n = len(world_points)
        for i, point in enumerate(world_points):
            x, y = point[0], point[1]
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.0
            msg.points.append(p)
            color = ColorRGBA()
            #color.g = 1.0 - (i / n)
            color.g = 1.0
            color.r = i / n
            color.a = 1.0
            msg.colors.append(color)
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
        segments = []
        n = len(self.waypoints)
        assert (n >= 2)
        p0_world = self.waypoints[0]
        prev_point = world_to_matrix(*p0_world,
                                     self.occupancy_grid.info.resolution)
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

    def on_timer(self):
        self.occupancy_grid = self.create_test_grid()
        self.occupancy_matrix = occupancy_grid_to_matrix(self.occupancy_grid)

        segments = self.compute_path_segments()
        collision_indices = self.compute_path_collisions(segments)
        collision_detected = bool(collision_indices)

        if not collision_detected:
            self.get_logger().info('Path is collision free :-)')
        else:
            self.get_logger().info(
                f'Collisions in segments: {collision_indices}')
        self.publish_path_marker(segments)
        self.occupancy_grid.data = self.occupancy_matrix.flatten()
        self.test_grid_pub.publish(self.occupancy_grid)


def main():
    rclpy.init()
    node = PathPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
