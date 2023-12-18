#!/usr/bin/env python3
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from visualization_msgs.msg import Marker


class MapperNode(Node):

    def __init__(self):
        super().__init__(node_name='mapper')

        self.marker_pub = self.create_publisher(msg_type=Marker,
                                                topic='robot_marker',
                                                qos_profile=1)

        self.pose_sub = self.create_subscription(
            msg_type=PoseWithCovarianceStamped,
            topic='vision_pose_cov',
            callback=self.on_pose,
            qos_profile=1)

        self.timer = self.create_timer(1.0, self.on_timer)

    def on_pose(self, msg: PoseWithCovarianceStamped):
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = 'map'
        marker.type = Marker.MESH_RESOURCE
        marker.ns = 'bluerov'
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

    def on_timer(self):
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = 'map'
        marker.type = Marker.MESH_RESOURCE
        marker.ns = 'pool'
        marker.pose.position.x = 1.0
        marker.pose.position.y = 2.0
        marker.pose.position.z = -1.5
        marker.color.r = 0.435
        marker.color.g = 0.725
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        mesh_resource = 'package://hippo_sim/models/pool/meshes/pool.dae'
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
