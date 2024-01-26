#!/usr/bin/env python3

from enum import Enum, auto

import numpy as np
import rclpy
import heapq
import time
from itertools import permutations
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from scenario_msgs.msg import Viewpoints
from scenario_msgs.srv import MoveToStart, SetPath
from std_msgs.msg import Header
from std_srvs.srv import Trigger
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from visualization_msgs.msg import Marker


class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority, *args):
        entry = (priority, self.count, item, *args)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item, *args) = heapq.heappop(self.heap)
        if not args:
            return item
        else:
            return item, *args

    def is_empty(self):
        return len(self.heap) == 0

    def update(self, item, priority, *args):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i, _) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item, *args))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority, *args)


k_directions = ["left", "right", "up", "down", "left_up", "left_down", "right_up", "right_down"]

k_direction2vector = {"left": (-1, 0), "right": (1, 0), "up": (0, 1), "down": (0, -1),
                      "left_up": (-1, 1), "left_down": (-1, -1), "right_up": (1, 1), "right_down": (1, -1)}
k_direction2cost = {"left": 1, "right": 1, "up": 1, "down": 1,
                    "left_up": np.sqrt(2), "left_down": np.sqrt(2), "right_up": np.sqrt(2), "right_down": np.sqrt(2)}


def action_sequence_to_points(start_position, actions):
    points = [list(start_position)]
    for action in actions:
        direction = k_direction2vector[action]
        points.append([points[-1][0] + direction[0], points[-1][1] + direction[1]])
    return points


class SearchProblem:
    def get_start_state(self):
        pass

    def is_goal_state(self, state):
        pass

    def get_successors(self, state):
        pass


class Point2PointProblem(SearchProblem):
    def __init__(self, start_position, goal_position, gridmap):
        self.start_state = start_position
        self.goal_state = goal_position
        self.gridmap = gridmap

    def get_start_state(self):
        return self.start_state

    def is_goal_state(self, state):
        return self.goal_state == state

    def get_successors(self, state):
        successors = []
        for action in k_directions:
            dx, dy = k_direction2vector[action]
            next_state = (int(state[0] + dx), int(state[1] + dy))
            if self.gridmap[next_state[1], next_state[0]] > 0:
                continue
            cost = k_direction2cost[action]
            successors.append((next_state, action, cost))
        return successors


class ViewpointsProblem(SearchProblem):
    def __init__(self, start_position, viewpoints, gridmap):
        self.start_position = start_position
        self.viewpoints = viewpoints
        self.gridmap = gridmap

    def get_start_state(self):
        return (self.start_position, tuple([0 for i in range(len(self.viewpoints))]))

    def is_goal_state(self, state):
        return sum(list(state[1])) == len(self.viewpoints)

    def get_successors(self, state):
        successors = []
        for action in k_directions:
            dx, dy = k_direction2vector[action]
            next_position = (int(state[0][0] + dx), int(state[0][1] + dy))
            if self.gridmap[next_position[1], next_position[0]] > 0:
                continue
            visited_viewpoints = list(state[1])
            for i, viewpoint in enumerate(self.viewpoints):
                if next_position == viewpoint:
                    visited_viewpoints[i] |= 1
            cost = k_direction2cost[action]
            successors.append(((next_position, tuple(visited_viewpoints)), action, cost))
        return successors


class State(Enum):
    UNSET = auto()
    INIT = auto()
    IDLE = auto()
    MOVE_TO_START = auto()
    NORMAL_OPERATION = auto()


def occupancy_grid_to_matrix(grid: OccupancyGrid):
    data = np.array(grid.data, dtype=np.uint8)
    data = data.reshape(grid.info.height, grid.info.width)
    return data


def world_to_matrix(x, y, grid_size):
    return [round(x / grid_size), round(y / grid_size)]


def matrix_index_to_world(x, y, grid_size):
    return [x * grid_size, y * grid_size]


def multiple_matrix_indeces_to_world(points, grid_size):
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


def null_heuristic(state, problem: SearchProblem):
    return 0


def euclidean_heuristic(state, problem: Point2PointProblem):
    return np.sqrt(np.power(problem.goal_state[0] - state[0], 2) + np.power(problem.goal_state[1] - state[1], 2))


def viewpoint_heuristic(state, problem: ViewpointsProblem):
    total_dist = float("inf")
    active_viewpoints = [problem.viewpoints[i] for i in range(len(problem.viewpoints)) if (state[1][i] == 0)]
    if not active_viewpoints:
        return 0
    n_active_viewpoints = len(active_viewpoints)
    n_points = len(active_viewpoints) + 1
    x_points = np.vstack(
        (np.array([[state[0][0]]]),
         np.array([active_viewpoints[i][0] for i in range(len(active_viewpoints))]).reshape(-1, 1)))
    y_points = np.vstack(
        (np.array([[state[0][1]]]),
         np.array([active_viewpoints[i][1] for i in range(len(active_viewpoints))]).reshape(-1, 1)))
    distance_matrix = np.sqrt(
        np.power(x_points.transpose() - x_points, 2) + np.power(y_points.transpose() - y_points, 2))

    permutation_object = permutations(range(1, n_points), n_active_viewpoints)
    for order in permutation_object:
        points = [0] + list(order)
        dist = 0.0
        for i in range(1, n_points):
            dist += distance_matrix[points[i - 1], points[i]]
        total_dist = min(dist, total_dist)
    return total_dist  # Default to trivial solution


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
        self.path_segments = []

        self.compute_full_path = False
        self.search_to_start = False
        self.init_params()

        self.init_clients()
        self.init_services()
        self.grid_map_sub = self.create_subscription(OccupancyGrid,
                                                     'occupancy_grid',
                                                     self.on_occupancy_grid, 1)
        self.viewpoints_sub = self.create_subscription(Viewpoints, 'viewpoints',
                                                       self.on_viewpoints, 1)

    def init_params(self):
        self.declare_parameters(
            namespace='',
            parameters=[
                ('compute_full_path', rclpy.Parameter.Type.BOOL),
                ('search_to_start', rclpy.Parameter.Type.BOOL),
            ])
        param = self.get_parameter('compute_full_path')
        self.get_logger().info(f'{param.name}={param.value}')
        self.compute_full_path = param.value
        param = self.get_parameter('search_to_start')
        self.get_logger().info(f'{param.name}={param.value}')
        self.search_to_start = param.value

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
        print("start_pose: ", request.current_pose)
        print("target_pose: ", request.target_pose)
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
        if self.search_to_start:
            if not self.occupancy_grid:
                self.get_logger().info('Moving to start position failed, waiting for occupancy grid!')
                return False
            path_segment = self.compute_a_star_segment(p0, p1)
        else:
            path_segment = self.compute_simple_path_segment(p0,
                                                            p1,
                                                            check_collision=False)
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

    def a_star(self, problem: SearchProblem, heuristic=null_heuristic, consistent_heuristic=False):
        if consistent_heuristic:
            expanded = set()
            fringe = PriorityQueue()
            fringe.push((problem.get_start_state(), [], 0), heuristic(problem.get_start_state(), problem))
            while (not fringe.is_empty()):
                node = fringe.pop()
                state, actions, cost = node
                if problem.is_goal_state(state):
                    return actions
                if state in expanded:
                    continue
                expanded.add(state)
                successors = problem.get_successors(state)
                for next_state, next_action, next_cost in successors:
                    fringe.push((next_state, actions + [next_action], cost + next_cost),
                                cost + next_cost + heuristic(next_state, problem))
        else:
            expanded = set()
            fringe = PriorityQueue()
            fringe.push(problem.get_start_state(), heuristic(problem.get_start_state(), problem), ([], 0))
            while (not fringe.is_empty()):
                node = fringe.pop()
                state, (actions, cost) = node
                if problem.is_goal_state(state):
                    return actions
                if state in expanded:
                    continue
                expanded.add(state)
                successors = problem.get_successors(state)
                for next_state, next_action, next_cost in successors:
                    fringe.update(next_state, cost + next_cost + heuristic(next_state, problem),
                                  (actions + [next_action], cost + next_cost))
        print("No feasible solution found!")
        return None

    def is_walkable(self, p0, p1):
        # https://www.gamedeveloper.com/programming/toward-more-realistic-pathfinding
        # http://eugen.dedu.free.fr/projects/bresenham/
        infeasible_point = lambda x, y: self.occupancy_matrix[x, y] > 0

        traversed_fields = []
        x1 = p0[0]
        y1 = p0[1]
        x2 = p1[0]
        y2 = p1[1]
        # i = np.nan            # loop counter
        ystep = np.nan
        xstep = np.nan  # the step on y and x axis
        error = np.nan  # the error accumulated during the increment
        errorprev = np.nan  # *vision the previous value of the error variable
        y = y1
        x = x1  # the line points
        ddy = np.nan
        ddx = np.nan  # compulsory variables: the double values of dy and dx
        dx = x2 - x1;
        dy = y2 - y1;
        if infeasible_point(y1, x1):  # first point
            return False
        # NB the last point can't be here, because of its previous point (which has to be verified)
        if (dy < 0):
            ystep = -1
            dy = -dy
        else:
            ystep = 1
        if (dx < 0):
            xstep = -1
            dx = -dx
        else:
            xstep = 1
        ddy = 2 * dy  # work with double values for full precision
        ddx = 2 * dx
        if (ddx >= ddy):  # first octant (0 <= slope <= 1)
            # compulsory initialization (even for errorprev, needed when dx==dy)
            error = dx
            errorprev = dx  # start in the middle of the square
            for i in range(dx):  # do not use the first point (already done)
                x += xstep
                error += ddy
                if (error > ddx):  # increment y if AFTER the middle ( > )
                    y += ystep
                    error -= ddx
                    # three cases (octant == right->right-top for directions below):
                    if (error + errorprev < ddx):  # bottom square also
                        if infeasible_point(y - ystep, x):
                            return False
                    elif (error + errorprev > ddx):  # left square also
                        if infeasible_point(y, x - xstep):
                            return False
                    else:  # corner: bottom and left squares also
                        if infeasible_point(y - ystep, x):
                            return False
                        if infeasible_point(y, x - xstep):
                            return False
                if infeasible_point(y, x):
                    return False
                errorprev = error
        else:  # the same as above
            error = dy
            errorprev = dy
            for i in range(dy):
                y += ystep;
                error += ddx;
                if (error > ddy):
                    x += xstep;
                    error -= ddy;
                    if (error + errorprev < ddy):
                        if infeasible_point(y, x - xstep):
                            return False
                    elif (error + errorprev > ddy):
                        if infeasible_point(y - ystep, x):
                            return False
                    else:
                        if infeasible_point(y, x - xstep):
                            return False
                        if infeasible_point(y - ystep, x):
                            return False
                if infeasible_point(y, x):
                    return False
                errorprev = error
        if (not y == y2) | (
        not x == x2):  # the last point (y2,x2) has to be the same with the last point of the algorithm
            print("Algorithm did not end at endpoint!")
            return False
        return True

    def straighten_path(self, points):
        idxs = [0]
        i = 1
        while (i + 1 < len(points)):
            if not self.is_walkable(points[idxs[-1]], points[i + 1]):
                idxs.append(i)
            i += 1
        idxs.append(len(points) - 1)
        key_points = [points[idx] for idx in idxs]

        #resample points between key_points
        resampled_points = []
        for i in range(len(key_points) - 1):
            n_resampling = int(np.ceil(np.sqrt((key_points[i+1][0] - key_points[i][0])**2 + (key_points[i+1][1] - key_points[i][1])**2)))
            resampled_x = np.linspace(key_points[i][0], key_points[i+1][0], n_resampling)
            resampled_y = np.linspace(key_points[i][1], key_points[i+1][1], n_resampling)
            if i == 0:
                idx_range = range(0, n_resampling)
            else:
                idx_range = range(1, n_resampling)
            resampled_points = resampled_points + [[resampled_x[idx], resampled_y[idx]] for idx in idx_range]
        return resampled_points

    def postprocess_path(self, p0: Pose, p1: Pose, points: [tuple], check_collisions=True):
        if check_collisions:
            collision_indices = self.has_collisions(points)
        else:
            collision_indices = []
        if not collision_indices:  # post process path further such that unnecessary zic-zac behaviour is eliminated
            # and shortest straight paths are chosen
            points = self.straighten_path(points)
        # Convert back our matrix/grid_map points to world coordinates. Since
        # the grid_map does not contain information about the z-coordinate,
        # the following list of points only contains the x and y component.
        xy_3d = multiple_matrix_indeces_to_world(points, self.cell_size)

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
        yaws = np.linspace(yaw0, yaw1, len(points_3d))
        orientations = []
        for yaw in yaws:
            q = quaternion_from_euler(0.0, 0.0, yaw)
            orientations.append(Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]))

        path = Path()
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'map'
        path.poses = [
            PoseStamped(header=header, pose=Pose(position=p, orientation=q))
            for p, q in zip(points_3d, orientations)
        ]
        return {'path': path, 'collision_indices': collision_indices}

    def compute_full_a_star(self, p0: Pose, viewpoints: [Pose]):
        p0_2d = tuple(world_to_matrix(p0.position.x, p0.position.y, self.cell_size))
        viewpoints_2d = [tuple(world_to_matrix(p.position.x, p.position.y, self.cell_size)) for p in viewpoints]
        search_problem = ViewpointsProblem(p0_2d, viewpoints_2d, self.occupancy_matrix)
        search_function = lambda problem: self.a_star(problem, heuristic=viewpoint_heuristic, consistent_heuristic=True)
        start_time = time.time()
        actions = search_function(search_problem)
        end_time = time.time()
        print("Processed in {} seconds.".format(end_time - start_time))
        points = action_sequence_to_points(p0_2d, actions)
        idxs = [0]
        for viewpoint_2d in viewpoints_2d:
            idxs.append(points.index(list(viewpoint_2d)))
        order = list(np.argsort(idxs))
        idxs = [idxs[i] for i in order]
        poses = [p0] + viewpoints
        path_segments = []
        start_time = time.time()
        for i in range(len(order) - 1):
            path_segments.append(
                self.postprocess_path(poses[order[i]], poses[order[i + 1]], points[idxs[i]:idxs[i + 1] + 1]))

        end_time = time.time()
        print("Postprocessed in {} seconds.".format(end_time - start_time))
        return path_segments

    def compute_a_star_segment(self, p0: Pose, p1: Pose):
        # TODO: implement your algorithms
        # you probably need the gridmap: self.occupancy_grid
        p0_2d = tuple(world_to_matrix(p0.position.x, p0.position.y, self.cell_size))
        p1_2d = tuple(world_to_matrix(p1.position.x, p1.position.y, self.cell_size))
        search_problem = Point2PointProblem(p0_2d, p1_2d, self.occupancy_matrix)
        search_function = lambda problem: self.a_star(problem, heuristic=euclidean_heuristic, consistent_heuristic=False)
        start_time = time.time()
        actions = search_function(search_problem)
        end_time = time.time()
        print("Processed in {} seconds.".format(end_time - start_time))
        points = action_sequence_to_points(p0_2d, actions)
        path_segments = self.postprocess_path(p0, p1, points)
        return path_segments

    def compute_simple_path_segment(self,
                                    p0: Pose,
                                    p1: Pose,
                                    check_collision=True):
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
        xy_3d = multiple_matrix_indeces_to_world(line_points_2d, self.cell_size)

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
        yaws = np.linspace(yaw0, yaw1, len(points_3d))
        orientations = []
        for yaw in yaws:
            q = quaternion_from_euler(0.0, 0.0, yaw)
            orientations.append(Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]))

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

    def handle_mission_completed(self):
        self.get_logger().info('Mission completed.')
        if not self.do_stop():
            self.get_logger().error(
                'All waypoints completed, but could not '
                'stop the path_follower. Trying again...',
                throttle_duration_sec=1.0)
            return
        self.state = State.IDLE

    def compute_new_path(self, viewpoints: Viewpoints, calculate_full=False):
        i = self.find_first_uncompleted_viewpoint(viewpoints)
        # start position and is treated differently.
        if i < 1:
            return
        # complete them all.
        # We do nothing smart here. We keep the order in which we received
        # the waypoints and connect them by straight lines.
        # No collision avoidance.
        # We only perform a collision detection and give up in case that
        # our path of straight lines collides with anything.
        # Not very clever, eh?

        # now get the remaining uncompleted viewpoints. In general, we can
        # assume that all the following viewpoints are uncompleted, since
        # we complete them in the same order as we get them in the
        # viewpoints message. But it is not that hard to double-check it.
        viewpoint_poses = [
            v.pose for v in viewpoints.viewpoints[i:] if not v.completed
        ]
        # get the most recently visited viewpoint. Since we do not change
        # the order of the viewpoints, this will be the viewpoint right
        # before the first uncompleted viewpoint in the list, i.e. i-1
        p0 = viewpoints.viewpoints[i - 1].pose
        if calculate_full:
            path_segments = self.compute_full_a_star(p0, viewpoint_poses)
            return path_segments
        viewpoint_poses.insert(0, p0)

        # now we can finally call our super smart function to compute
        # the path piecewise between the viewpoints
        path_segments = []
        for i in range(1, len(viewpoint_poses)):
            # segment = self.compute_simple_path_segment(viewpoint_poses[i - 1],
            #                                           viewpoint_poses[i])

            # alternatively call your own implementation
            segment = self.compute_a_star_segment(viewpoint_poses[i - 1],
                                                  viewpoint_poses[i])
            path_segments.append(segment)
        return path_segments

    def handle_no_collision_free_path(self):
        self.get_logger().fatal('We have a collision in our current segment!'
                                'Giving up...')
        if self.do_stop():
            self.state = State.IDLE
        else:
            self.state = State.UNSET

    def do_normal_operation(self, viewpoints: Viewpoints):
        # what we need to do:
        # - check if the viewpoints changed, if so, recalculate the path
        i = self.find_first_uncompleted_viewpoint(viewpoints)
        # we completed our mission!
        if i < 0:
            self.handle_mission_completed()
            return

        if (not self.recomputation_required) or self.target_viewpoint_index == i:
            # we are still chasing the same viewpoint. Nothing to do.
            return
        self.get_logger().info('Computing new path segments')
        self.target_viewpoint_index = i
        if i == 0:
            p = viewpoints.viewpoints[0].pose
            if not self.move_to_start(p, p):
                self.get_logger().fatal(
                    'Could not move to first viewpoint. Giving up...')
                if self.do_stop():
                    self.state = State.IDLE
                else:
                    self.state = State.UNSET
            return

        path_segments = self.compute_new_path(viewpoints, calculate_full=False)
        if not path_segments:
            self.get_logger().error(
                'This is a logic error. The cases that would have lead to '
                'no valid path_segments should have been handled before')
            return
        if path_segments[0]['collision_indices']:
            self.handle_no_collision_free_path()
            return
        self.set_new_path(path_segments[0]['path'])
        return

    def do_normal_operation_full_path(self, viewpoints: Viewpoints):
        # what we need to do:
        # - check if the viewpoints changed, if so, recalculate the path
        i = self.count_completed_viewpoints(viewpoints)
        # we completed our mission!
        if i == len(viewpoints.viewpoints):
            self.handle_mission_completed()
            return
        if (not self.recomputation_required) or self.target_viewpoint_index == i:
            # we are still chasing the same viewpoint. Nothing to do.
            return
        self.get_logger().info('Computing new path segments')
        self.target_viewpoint_index = i
        if i == 0:
            p = viewpoints.viewpoints[0].pose
            if not self.move_to_start(p, p):
                self.get_logger().fatal(
                    'Could not move to first viewpoint. Giving up...')
                if self.do_stop():
                    self.state = State.IDLE
                else:
                    self.state = State.UNSET
            return
        if i == 1:
            self.path_segments = self.compute_new_path(viewpoints, calculate_full=True)
            self.path_counter = 0
            if not self.path_segments:
                self.get_logger().error(
                    'This is a logic error. The cases that would have lead to '
                    'no valid path_segments should have been handled before')
                return
        if self.path_segments[self.path_counter]['collision_indices']:
            self.handle_no_collision_free_path()
            return
        self.set_new_path(self.path_segments[self.path_counter]['path'])
        self.path_counter += 1
        return

    def on_viewpoints(self, msg: Viewpoints):
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
            if self.compute_full_path:
                self.do_normal_operation_full_path(msg)
            else:
                self.do_normal_operation(msg)

    def find_first_uncompleted_viewpoint(self, viewpoints: Viewpoints):
        for i, viewpoint in enumerate(viewpoints.viewpoints):
            if not viewpoint.completed:
                return i
        # This should not happen!
        return -1

    def count_completed_viewpoints(self, viewpoints: Viewpoints):
        counter = 0
        for i, viewpoint in enumerate(viewpoints.viewpoints):
            if viewpoint.completed:
                counter += 1
        # This should not happen!
        return counter

    def on_occupancy_grid(self, msg: OccupancyGrid):
        self.occupancy_grid = msg
        self.occupancy_matrix = occupancy_grid_to_matrix(self.occupancy_grid)
        if msg.info.resolution != self.cell_size:
            self.get_logger().info('Cell size changed. Recomputation required.')
            self.recomputation_required = True
            self.cell_size = msg.info.resolution

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

    def publish_path_marker(self, segments):
        msg = self.path_marker
        world_points = self.segments_to_world_points(segments)
        msg.points = [Point(x=p[0], y=p[1], z=-0.5) for p in world_points]
        msg.header.stamp = self.get_clock().now().to_msg()
        self.path_marker_pub.publish(msg)


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
