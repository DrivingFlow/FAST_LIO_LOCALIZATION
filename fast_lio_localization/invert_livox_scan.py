#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import rclpy.parameter
import rclpy.parameter_service
from sensor_msgs.msg import PointCloud2, Imu
from livox_ros_driver2.msg import CustomMsg
from sensor_msgs.msg import PointCloud2
import ros2_numpy
from rcl_interfaces.srv import GetParameters


class LivoxLaserToPointcloud(Node):
    def __init__(self):
        super().__init__("Invert_Livox_Scan")

        xfer_format = self.declare_parameter("xfer_format", 0).value

        if xfer_format == 0:
            self.pub_scan = self.create_publisher(PointCloud2, "/livox/lidar", 10)
            self.sub_scan = self.create_subscription(PointCloud2, "/livox/inverted_lidar", self.pointcloud2_callback, 10)

        elif xfer_format == 1:
            self.pub_scan = self.create_publisher(CustomMsg, "/livox/lidar", 10)
            self.sub_scan = self.create_subscription(CustomMsg, "/livox/inverted_lidar", self.custom_msg_callback, 10)

        else:
            self.get_logger().error(f"Method undefined for xfer_format = {xfer_format}")
            self.destroy_node()
            
            return

        self.pub_imu = self.create_publisher(Imu, "/livox/imu", 10)
        self.sub_imu = self.create_subscription(Imu, "/livox/inverted_imu", self.imu_callback, 10)
        
    # def get_xfer_format(self):
    #     """Get the 'xfer_format' parameter from the Livox ROS 2 driver."""
    #     try:            
    #         param_client = self.create_client(GetParameters, "/livox_lidar_publisher/get_parameters")
    #         self.get_logger().info("Waiting for Livox driver parameter service...")
            
            
    #         while not param_client.wait_for_service(timeout_sec=2.0):
    #             self.get_logger().warn("Waiting for Livox driver parameter service...")

    #         request = GetParameters.Request()
    #         request.names = ["xfer_format"]
    #         future = param_client.call_async(request)
    #         rclpy.spin_until_future_complete(self, future)

    #         if future.result() is not None and len(future.result().values) > 0:
    #             return future.result().values[0].integer_value  # Extract integer parameter value

    #     except Exception as e:
    #         self.get_logger().error(f"Failed to get xfer_format from Livox driver: {e}")

    #     return -1  # Default to -1 if parameter fetch fails

    def pointcloud2_callback(self, msg: PointCloud2):
        data = ros2_numpy.numpify(msg)
        # print(data)
        
        pc = data['xyz']
        # print(pc)
        pc[:, 1] = -pc[:, 1]
        pc[:, 2] = -pc[:, 2]
        # print(pc)
        
        data = {"xyz": pc}  # Invert Y, Z
        # print(data)

        out_msg = ros2_numpy.msgify(PointCloud2, data)
        out_msg.header = msg.header
        # out_msg.header.stamp = self.get_clock().now().to_msg()
        out_msg.point_step = 12
        self.pub_scan.publish(out_msg)

    def custom_msg_callback(self, msg: CustomMsg):
        for p in msg.points:
            p.y = -p.y
            p.z = -p.z
            
        # msg.header.stamp = self.get_clock().now().to_msg()

        self.pub_scan.publish(msg)

    def imu_callback(self, msg: Imu):
        msg.angular_velocity.y = -msg.angular_velocity.y
        msg.angular_velocity.z = -msg.angular_velocity.z
        
        # msg.header.stamp = self.get_clock().now().to_msg()

        self.pub_imu.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LivoxLaserToPointcloud()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()