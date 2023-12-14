#!/usr/bin/env python3

from enum import Enum, auto

import numpy as np
import rclpy
from final_project.msg import Viewpoints
from final_project.srv import MoveToStart, SetPath
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Header
from std_srvs.srv import Trigger
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from visualization_msgs.msg import Marker


class State(Enum):
    UNSET = auto()
    INIT = auto()
    IDLE = auto()
    MOVE_TO_START = auto()
    NORMAL_OPERATION = auto()


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


def compute_discrete_line(x0, y0, x1, y1):
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


class PathPlanner(Node):

    def __init__(self):
        super().__init__(node_name='path_planner')
        self.state = State.UNSET
        self.cell_size = 0.2
        self.recomputation_required = True
        self.target_viewpoint_index = -1
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
        self.init_clients()
        self.init_services()
        self.grid_map_sub = self.create_subscription(OccupancyGrid,
                                                     'occupancy_grid',
                                                     self.on_occupancy_grid, 1)
        self.viewpoints_sub = self.create_subscription(Viewpoints, 'viewpoints',
                                                       self.on_viewpoints, 1)

    def init_services(self):
        self.move_to_start_service = self.create_service(
            MoveToStart, '~/move_to_start', self.serve_move_to_start)
        self.start_service = self.create_service(Trigger, '~/start',
                                                 self.serve_start)
        self.stop_service = self.create_service(Trigger, '~/stop',
                                                self.serve_stop)

    def init_clients(self):
        cb_group = rclpy.callback_groups.MutuallyExclusiveCallbackGroup()
        self.set_path_client = self.create_client(SetPath,
                                                  'path_follower/set_path',
                                                  callback_group=cb_group)
        self.path_finished_client = self.create_client(
            Trigger, 'path_follower/path_finished', callback_group=cb_group)

    def serve_move_to_start(self, request, response):
        self.state = State.MOVE_TO_START
        self.start_pose = request.target_pose
        self.current_pose = request.current_pose
        # we do not care for collisions while going to the start position
        # in the simulation 'collisions' do not matter. In the lab, we
        # can manually make sure that we avoid collisions, by bringing the
        # vehicle in a safe position manually before starting anything.
        response.success = self.move_to_start(request.current_pose,
                                              request.target_pose)
        return response

    def move_to_start(self, p0: Pose, p1: Pose):
        path_segment = self.compute_path_segment(p0, p1, check_collision=False)
        request = SetPath.Request()
        request.path = path_segment['path']
        answer = self.set_path_client.call(request)
        if answer.success:
            self.get_logger().info('Moving to start position')
            return True
        else:
            self.get_logger().info(
                'Asked to move to start position. '
                'But the path follower did not accept the new path.')
            return False

    def has_collisions(self, points_2d):
        if not self.occupancy_grid:
            return []
        collision_indices = [
            i for i, p in enumerate(points_2d)
            if self.occupancy_matrix[p[1], p[0]] >= 50
        ]
        return collision_indices

    def compute_path_segment(self, p0: Pose, p1: Pose, check_collision=True):
        p0_2d = world_to_matrix(p0.position.x, p0.position.y, self.cell_size)
        p1_2d = world_to_matrix(p1.position.x, p1.position.y, self.cell_size)
        # now we should/could apply some sophisticated algorithm to compute
        # the path that brings us from p0_2d to p1_2d. For this dummy example
        # we simply go in a straight line. Not very clever, but a straight
        # line is the shortes path between two points, isn't it?
        line_points_2d = compute_discrete_line(p0_2d[0], p0_2d[1], p1_2d[0],
                                               p1_2d[1])
        if check_collision:
            collision_indices = self.has_collisions(line_points_2d)
        else:
            collision_indices = []

        # Convert back our matrix/grid_map points to world coordinates. Since
        # the grid_map does not contain information about the z-coordinate,
        # the following list of points only contains the x and y component.
        xy_3d = multiple_matrix_index_to_world(line_points_2d, self.cell_size)

        # it might be, that only a single grid point brings us from p0 to p1.
        # in this duplicate this point. this way it is easier to handle.
        if len(xy_3d) == 1:
            xy_3d.append(xy_3d[0])
        z0 = p0.position.z
        z1 = p1.position.z
        z_step = (z1 - z0) / (len(xy_3d) - 1)
        points_3d = [
            Point(x=p[0], y=p[1], z=z0 + i * z_step)
            for i, p in enumerate(xy_3d)
        ]
        # Replace the last point with the exac value stored in p1.position
        # instead of the grid map discretized world coordinate
        points_3d[-1] = p1.position
        # Now we have a waypoint path with the x and y component computed by
        # our path finding algorithm and z is a linear interpolation between
        # the z coordinate of the start and the goal pose.

        # now we need to compute our desired heading (yaw angle) while we
        # follow the waypoints. We choose a not-so-clever approach by
        # keeping the yaw angle from our start pose and only set the yaw
        # angle to the desired yaw angle from the goal pose for the very last
        # waypoint
        q0 = p0.orientation
        _, _, yaw0 = euler_from_quaternion([q0.x, q0.y, q0.z, q0.w])
        q1 = p1.orientation
        _, _, yaw1 = euler_from_quaternion([q1.x, q1.y, q1.z, q1.w])

        # replace the very last orientation with the orientation of our
        # goal pose p1.
        q = quaternion_from_euler(0.0, 0.0, yaw0)
        orientations = [Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
                        ] * len(points_3d)
        q = quaternion_from_euler(0.0, 0.0, yaw1)
        orientations[-1] = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        path = Path()
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'map'
        path.poses = [
            PoseStamped(header=header, pose=Pose(position=p, orientation=q))
            for p, q in zip(points_3d, orientations)
        ]
        return {'path': path, 'collision_indices': collision_indices}

    def reset_internals(self):
        self.target_viewpoint_index = -1
        self.recomputation_required = True
        self.state = State.UNSET

    def serve_start(self, request, response):
        if self.state != State.NORMAL_OPERATION:
            self.get_logger().info('Starting normal operation.')
            self.reset_internals()
        self.state = State.NORMAL_OPERATION
        response.success = True
        return response

    def serve_stop(self, request, response):
        if self.state != State.IDLE:
            self.get_logger().info('Asked to stop. Going to idle mode.')
        response.success = self.do_stop()
        return response

    def do_stop(self):
        self.state = State.IDLE
        if self.path_finished_client.call(Trigger.Request()).success:
            self.reset_internals()
            self.state = State.IDLE
            return True
        return False

    def on_viewpoints(self, msg: Viewpoints):  # noqa: C901
        if self.state == State.IDLE:
            return
        if self.state == State.UNSET:
            if self.do_stop():
                self.state = State.IDLE
            else:
                self.get_logger().error('Failed to stop.')
            return
        if self.state == State.MOVE_TO_START:
            # nothing to be done here. We already did the setup when the
            # corresponding service was called
            return
        if self.state == State.NORMAL_OPERATION:
            # what we need to do:
            # - check if the viewpoints changed, if so, recalculate the path
            i = self.find_first_uncompleted_viewpoint(msg)
            # we completed our mission!
            if i < 0:
                self.get_logger().info('Mission completed.')
                if not self.do_stop():
                    self.get_logger().error(
                        'All waypoints completed, but could not '
                        'stop the path_follower. Trying again...',
                        throttle_duration_sec=1.0)
                    return
                self.state = State.IDLE
                return

            if (not self.recomputation_required
                ) or self.target_viewpoint_index == i:
                # we are still chasing the same viewpoint. Nothing to do.
                return
            self.get_logger().info('Computing new path segments')
            self.target_viewpoint_index = i
            if i == 0:
                p = msg.viewpoints[0].pose
                if not self.move_to_start(p, p):
                    self.get_logger().fatal(
                        'Could not move to first viewpoint. Giving up...')
                    if self.do_stop():
                        self.state = State.IDLE
                    else:
                        self.state = State.UNSET
                return
            # start position and is treated differently. we could bring them
            # in an appropriate order or do whatever with them. The task is to
            # complete them all.
            # We do nothing smart here. We keep the order in which we received
            # them and connect them by straight lines. No collision avoidance.
            # We only perform a collision detection and give up in case that
            # our path of straight lines collides with anything.
            # Not very clever, eh?

            # now get the remaining uncompleted viewpoints. In general, we can
            # assume that all the following viewpoints are uncompleted, since
            # we complete them in the same order as we get them in the
            # viewpoints message. But it is not that hard to double-check it.
            viewpoint_poses = [
                v.pose for v in msg.viewpoints[i:] if not v.completed
            ]
            # get the most recently visited viewpoint. Since we do not change
            # the order of the viewpoints, this will be the viewpoint right
            # before the first uncompleted viewpoint in the list, i.e. i-1
            p0 = msg.viewpoints[i - 1].pose
            viewpoint_poses.insert(0, p0)

            # now we can finally call our super smart function to compute
            # the path piecewise between the viewpoints
            path_segments = [
                self.compute_path_segment(viewpoint_poses[i - 1],
                                          viewpoint_poses[i])
                for i in range(1, len(viewpoint_poses))
            ]
            if path_segments[0]['collision_indices']:
                self.get_logger().fatal(
                    'We have a collision in our current segment!'
                    'Giving up...')
                if self.do_stop():
                    self.state = State.IDLE
                else:
                    self.state = State.UNSET
                return
            self.set_new_path(path_segments[0]['path'])
            return

    def on_viewpoints_old(self, msg: Viewpoints):
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
        if msg.info.resolution != self.cell_size:
            self.get_logger().info('Cell size changed. Recomputation required.')
            self.recomputation_required = True
            self.cell_size = msg.info.resolution
        # TODO(lennartalff): Check if grid map changed?

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
