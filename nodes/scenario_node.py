#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from geometry_msgs.msg import Polygon, Point32, Point
from final_project.msg import PolygonsStamped
from visualization_msgs.msg import MarkerArray, Marker


class ScenarioNode(Node):

    def __init__(self):
        super().__init__(node_name='occupancy_grid')

        self.obstacles_pub = self.create_publisher(msg_type=PolygonsStamped,
                                                   topic='obstacles',
                                                   qos_profile=1)

        self.marker_pub = self.create_publisher(msg_type=MarkerArray,
                                                topic='markers',
                                                qos_profile=1)

        self.timer = self.create_timer(1.0 / 50, self.on_timer)

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


def main():
    rclpy.init()
    node = ScenarioNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
