#!/usr/bin/env python3
import cv2
import numpy as np
import rclpy
from final_project.msg import PolygonsStamped
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node


class MapperNode(Node):

    def __init__(self):
        super().__init__(node_name='mapper')

        self.marker_pub = self.create_publisher(msg_type=Marker,
                                                topic='robot_marker',
                                                qos_profile=1)

        self.obstacle_sub = self.create_subscription(
            msg_type=PoseWithCovarianceStamped,
            topic='vision_pose_cov',
            callback=self.on_pose,
            qos_profile=1)

    def on_pose(self, msg: PoseWithCovarianceStamped):
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = 'map'
        marker.type = Marker.MESH_RESOURCE
        marker.pose = msg.pose.pose
        marker.color.b = 0.9
        marker.color.g = 0.5
        marker.color.r = 0.1
        marker.color.a = 1.0
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        mesh_resource = 'package://hippo_sim/models/bluerov/meshes/bluerov.dae'
        marker.mesh_resource = mesh_resource

        self.marker_pub.publish(marker)


def main():
    rclpy.init()
    node = MapperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
