#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import struct
import os


class PCDPublisher(Node):
    def __init__(self):
        super().__init__('pcd_publisher')
        
        # Declare parameters
        self.declare_parameter('file_name', '')
        self.declare_parameter('tf_frame', 'map')
        self.declare_parameter('cloud_topic', '/map')
        self.declare_parameter('period_ms', 500)
        
        # Get parameters
        self.pcd_file = self.get_parameter('file_name').value
        self.frame_id = self.get_parameter('tf_frame').value
        cloud_topic = self.get_parameter('cloud_topic').value
        period_ms = self.get_parameter('period_ms').value
        
        # Validate PCD file path
        if not self.pcd_file or not os.path.exists(self.pcd_file):
            self.get_logger().error(f'PCD file not found: {self.pcd_file}')
            raise FileNotFoundError(f'PCD file not found: {self.pcd_file}')
        
        # Create publisher
        self.publisher = self.create_publisher(PointCloud2, cloud_topic, 10)
        
        # Load PCD file
        self.get_logger().info(f'Loading PCD file: {self.pcd_file}')
        self.point_cloud_msg = self.load_pcd(self.pcd_file)
        self.get_logger().info(f'Loaded {self.count_points(self.point_cloud_msg)} points from PCD file')
        
        # Create timer for publishing
        timer_period = period_ms / 1000.0  # Convert ms to seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.get_logger().info(f'Publishing PCD map on topic "{cloud_topic}" every {period_ms}ms')
    
    def load_pcd(self, filename):
        """Load a PCD file and convert it to PointCloud2 message"""
        points = []
        
        # First, read header in text mode to determine format
        header = {}
        header_lines = []
        with open(filename, 'rb') as f:
            while True:
                line = f.readline()
                try:
                    line_str = line.decode('ascii').strip()
                except:
                    break
                    
                if line_str.startswith('#'):
                    continue
                    
                header_lines.append(line_str)
                
                if line_str.startswith('VERSION'):
                    header['version'] = line_str.split()[1]
                elif line_str.startswith('FIELDS'):
                    header['fields'] = line_str.split()[1:]
                elif line_str.startswith('SIZE'):
                    header['size'] = [int(x) for x in line_str.split()[1:]]
                elif line_str.startswith('TYPE'):
                    header['type'] = line_str.split()[1:]
                elif line_str.startswith('COUNT'):
                    header['count'] = [int(x) for x in line_str.split()[1:]]
                elif line_str.startswith('WIDTH'):
                    header['width'] = int(line_str.split()[1])
                elif line_str.startswith('HEIGHT'):
                    header['height'] = int(line_str.split()[1])
                elif line_str.startswith('VIEWPOINT'):
                    header['viewpoint'] = line_str.split()[1:]
                elif line_str.startswith('POINTS'):
                    header['points'] = int(line_str.split()[1])
                elif line_str.startswith('DATA'):
                    header['data'] = line_str.split()[1]
                    # After DATA line, we have the actual data
                    if header['data'] == 'binary':
                        # Read remaining binary data
                        binary_data = f.read()
                        points = self.parse_binary_pcd(binary_data, header)
                    elif header['data'] == 'ascii':
                        # Read ASCII data
                        for line in f:
                            try:
                                line_str = line.decode('ascii').strip()
                                if line_str:
                                    values = [float(x) for x in line_str.split()]
                                    if len(values) >= 3:
                                        points.append(values[:3])
                            except:
                                continue
                    break
        
        if not points:
            self.get_logger().error('No valid points loaded from PCD file')
            raise ValueError('No valid points in PCD file')
        
        # Convert to PointCloud2 message
        points_array = np.array(points, dtype=np.float32)
        return self.create_pointcloud2(points_array)
    
    def parse_binary_pcd(self, binary_data, header):
        """Parse binary PCD data"""
        points = []
        num_points = header['points']
        
        # Find x, y, z field indices
        fields = header['fields']
        try:
            x_idx = fields.index('x')
            y_idx = fields.index('y')
            z_idx = fields.index('z')
        except ValueError:
            self.get_logger().error('PCD file must contain x, y, z fields')
            raise
        
        # Calculate point step (total bytes per point)
        point_step = sum(header['size'])
        
        # Calculate offsets for x, y, z fields
        x_offset = sum(header['size'][:x_idx])
        y_offset = sum(header['size'][:y_idx])
        z_offset = sum(header['size'][:z_idx])
        
        # Determine data type for x, y, z (usually 'F' for float)
        x_type = header['type'][x_idx]
        y_type = header['type'][y_idx]
        z_type = header['type'][z_idx]
        x_size = header['size'][x_idx]
        y_size = header['size'][y_idx]
        z_size = header['size'][z_idx]
        
        # Map PCD types to struct format
        type_map = {
            ('F', 4): 'f',  # float32
            ('F', 8): 'd',  # float64
            ('I', 1): 'B',  # uint8
            ('I', 2): 'H',  # uint16
            ('I', 4): 'I',  # uint32
            ('U', 1): 'b',  # int8
            ('U', 2): 'h',  # int16
            ('U', 4): 'i',  # int32
        }
        
        x_format = type_map.get((x_type, x_size), 'f')
        y_format = type_map.get((y_type, y_size), 'f')
        z_format = type_map.get((z_type, z_size), 'f')
        
        # Parse binary data
        for i in range(num_points):
            offset = i * point_step
            if offset + point_step > len(binary_data):
                break
            
            try:
                x = struct.unpack(x_format, binary_data[offset + x_offset:offset + x_offset + x_size])[0]
                y = struct.unpack(y_format, binary_data[offset + y_offset:offset + y_offset + y_size])[0]
                z = struct.unpack(z_format, binary_data[offset + z_offset:offset + z_offset + z_size])[0]
                
                # Check for NaN or inf values
                if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                    points.append([x, y, z])
            except:
                continue
        
        return points
    
    def create_pointcloud2(self, points):
        """Create a PointCloud2 message from numpy array of points"""
        msg = PointCloud2()
        msg.header.frame_id = self.frame_id
        
        # Define fields
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        
        msg = pc2.create_cloud(msg.header, fields, points)
        return msg
    
    def count_points(self, msg):
        """Count the number of points in a PointCloud2 message"""
        return msg.width * msg.height if msg.height > 0 else msg.width
    
    def timer_callback(self):
        """Publish the point cloud periodically"""
        # Update timestamp
        self.point_cloud_msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(self.point_cloud_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PCDPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
