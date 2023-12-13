#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose, Polygon
from final_project.msg import PolygonsStamped

import numpy as np
import cv2

OCCUPIED = 100


class MapperNode(Node):

    def __init__(self):
        super().__init__(node_name='occupancy_grid')

        self.tank_size_x = None
        self.tank_size_y = None
        self.cell_resolution = None
        self.init_params()

        self.num_cells_x = int(self.tank_size_x / self.cell_resolution)
        self.num_cells_y = 2 * self.num_cells_x

        # x: columns, y: rows --> switch order here
        self.cells = np.zeros((self.num_cells_y, self.num_cells_x),
                              dtype='int8')
        self.cells = self.get_tank_walls(self.cells)

        self.get_logger().info(
            f'Using a cell resolution of {self.cell_resolution} m per cell, ' +
            f'resulting in {self.num_cells_x} cells in x-direction and ' +
            f'{self.num_cells_y} cells in y-direction.')

        self.map_pub = self.create_publisher(msg_type=OccupancyGrid,
                                             topic='grid_map',
                                             qos_profile=1)

        self.obstacle_sub = self.create_subscription(msg_type=PolygonsStamped,
                                                     topic='obstacles',
                                                     callback=self.on_obstacles,
                                                     qos_profile=1)

    def init_params(self):
        self.declare_parameters(
            namespace='',
            parameters=[
                ('tank_size_x', rclpy.Parameter.Type.DOUBLE),
                ('tank_size_y', rclpy.Parameter.Type.DOUBLE),
                ('cell_resolution', rclpy.Parameter.Type.DOUBLE),
            ])
        param = self.get_parameter('tank_size_x')
        self.get_logger().info(f'{param.name}={param.value}')
        self.tank_size_x = param.value
        param = self.get_parameter('tank_size_y')
        self.get_logger().info(f'{param.name}={param.value}')
        self.tank_size_y = param.value
        param = self.get_parameter('cell_resolution')
        self.get_logger().info(f'{param.name}={param.value}')
        self.cell_resolution = param.value

        # TODO: Wir wollen eigentlich kein Online Ändern der Parameter, oder?

    def on_obstacles(self, msg: PolygonsStamped):
        num_obstacles = len(msg.polygons)
        if not num_obstacles:
            return

        for obstacle in msg.polygons:
            self.include_obstacle_in_map(obstacle)

        self.publish_map()

    def include_obstacle_in_map(self, obstacle: Polygon):
        num_points = len(obstacle.points)
        obstacle_cells = np.zeros((num_points, 2), dtype='int8')

        for index, point in enumerate(obstacle.points):
            obstacle_cells[index, 0] = int(point.x / self.cell_resolution)
            obstacle_cells[index, 1] = int(point.y / self.cell_resolution)

        img = np.copy(self.cells)
        self.cells = cv2.fillPoly(img,
                                  pts=np.int32([obstacle_cells]),
                                  color=(OCCUPIED))

    def publish_map(self):
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        meta_data = MapMetaData()
        meta_data.map_load_time = self.get_clock().now().to_msg()
        meta_data.resolution = self.cell_resolution
        meta_data.width = self.num_cells_x
        meta_data.height = self.num_cells_y
        # origin is at 0,0,0, same orientation as map frame
        meta_data.origin = Pose()
        msg.info = meta_data

        # map data in row-major order, starting with (0,0)
        msg.data = self.cells.flatten()
        self.map_pub.publish(msg)

    def get_tank_walls(self, cells: np.ndarray) -> np.ndarray:
        # we will set all borders of the grid as occupied
        cells[:, 0] = OCCUPIED
        cells[:, -1] = OCCUPIED
        cells[0, :] = OCCUPIED
        cells[-1, :] = OCCUPIED
        return cells


def main():
    rclpy.init()
    node = MapperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
