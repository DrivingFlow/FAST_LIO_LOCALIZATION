#!/usr/bin/env python3

import copy
import threading
import time

import open3d as o3d
import torch
import pypose as pp
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Point, Quaternion
from nav_msgs.msg import Odometry
# from rclpy.wait_for_message import wait_for_message
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
import numpy as np
import tf2_ros
import tf_transformations
import ros2_numpy

# Optional plotting (guarded by parameters)
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


class FastLIOLocalization(Node):
    def __init__(self):
        super().__init__("fast_lio_localization")
        self.global_map = None
        self.T_map_to_odom = np.eye(4)
        self.cur_odom = None
        self.cur_scan = None
        self.initialized = False
        self.prev_pose = None

        self.declare_parameters(
            namespace="",
            parameters=[
                ("map_voxel_size", 0.4),
                ("scan_voxel_size", 0.1),
                ("freq_localization", 0.5),
                ("freq_global_map", 0.25),
                ("localization_threshold", 0.8),
                ("fov", 6.28319),
                ("fov_far", 300),
                ("pcd_map_topic", "/map"),
                ("pcd_map_path", ""),
                ("icp_method", "open3d"),  # 'torch', 'open3d', or 'pypose'
                ("plot_rmse", True),
                ("plot_rmse_rate", 10.0),  # Hz
                ("plot_rmse_history", 200),
            ],
        )

        # RMSE plotting state
        self._rmse_hist = []
        self._rmse_t_hist = []
        self._dtrans_hist = []
        self._rmse_plot_last_t = 0.0
        self._rmse_fig = None
        self._rmse_ax = None
        self._rmse_ax2 = None
        self._rmse_line = None
        self._dtrans_line = None

        if self.get_parameter("plot_rmse").value and plt is None:
            self.get_logger().warn("plot_rmse is true but matplotlib is not available; disabling RMSE plot.")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # self.pub_global_map = self.create_publisher(PointCloud2, self.get_parameter("pcd_map_topic").value, 10)
        self.pub_pc_in_map = self.create_publisher(PointCloud2, "/cur_scan_in_map", 10)
        self.pub_submap = self.create_publisher(PointCloud2, "/submap", 10)
        self.pub_map_to_odom = self.create_publisher(Odometry, "/map_to_odom", 10)

        self.get_logger().info("Waiting for global map...")
        # global_map_msg = wait_for_message(msg_type = PointCloud2, node = self, topic = "/cloud_pcd")[1]
        # self.initialize_global_map(global_map_msg)
        
        self.initialize_global_map()
        self.get_logger().info("Global map received.")
        
        self.create_subscription(PointCloud2, "/cloud_registered", self.cb_save_cur_scan, 10)
        self.create_subscription(Odometry, "/Odometry", self.cb_save_cur_odom, 10)
        self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self.cb_initialize_pose, 10)

        self.timer_localisation = self.create_timer(1.0 / self.get_parameter("freq_localization").value, self.localisation_timer_callback)
        # self.timer_global_map = self.create_timer(1/ self.get_parameter("freq_global_map").value, self.global_map_callback)

    def global_map_callback(self):
        # self.get_logger().info(np.array(self.global_map.points).shape)
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "map"
        self.publish_point_cloud(self.pub_global_map, header, np.array(self.global_map.points))
        
    def pose_to_mat(self, pose):
        trans = np.eye(4)
        trans[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
        quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        trans[:3, :3] = tf_transformations.quaternion_matrix(quat)[:3, :3]
        return trans
    
    def msg_to_array(self, pc_msg):
        try:
            pc_array = ros2_numpy.numpify(pc_msg)
            return pc_array["xyz"]
        except Exception as e:
            # Fallback for malformed/unusual PointCloud2 layouts: use sensor_msgs_py.point_cloud2
            # This is slower but more tolerant of non-standard field layouts.
            # self.get_logger().warn(f"ros2_numpy.numpify failed: {e}; falling back to read_points")
            try:
                from sensor_msgs_py import point_cloud2 as pc2
            except Exception:
                # sensor_msgs_py might not be available; re-raise original error
                raise

            pts = []
            for p in pc2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=True):
                pts.append([p[0], p[1], p[2]])

            if len(pts) == 0:
                # keep original behavior to raise a useful error
                raise ValueError("PointCloud2 contains no valid x,y,z points")

            return np.array(pts, dtype=np.float64)
    
    def get_torch_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def run_icp_torch(self, src_np, tgt_np, device, init_T_np=None, max_iters=20):
        """
        Simple point-to-point ICP implemented in PyTorch.
        - Nearest neighbors via torch.cdist (O(N^2)).
        - Runs on the given device (CPU or CUDA).
        - Uses all provided points (no subsampling inside).
        """
        device = torch.device(device)
        src_np = src_np.astype(np.float32, copy=False)
        tgt_np = tgt_np.astype(np.float32, copy=False)
        src = torch.from_numpy(src_np).to(device)
        tgt = torch.from_numpy(tgt_np).to(device)
        if init_T_np is None:
            R = torch.eye(3, device=device)
            t = torch.zeros(3, device=device)
            src_tf = src.clone()
        else:
            init_T = torch.from_numpy(init_T_np.astype(np.float32)).to(device)
            R = init_T[:3, :3]
            t = init_T[:3, 3]
            src_tf = (src @ R.T) + t
        for i in range(max_iters):
            d = torch.cdist(src_tf, tgt)
            nn_idx = d.argmin(dim=1)
            tgt_match = tgt[nn_idx]
            mu_src = src_tf.mean(0)
            mu_tgt = tgt_match.mean(0)
            X = src_tf - mu_src
            Y = tgt_match - mu_tgt
            H = X.T @ Y
            U, S, Vt = torch.linalg.svd(H)
            R_d = Vt.T @ U.T
            if torch.det(R_d) < 0:
                Vt[-1, :] *= -1
                R_d = Vt.T @ U.T
            t_d = mu_tgt - R_d @ mu_src
            R = R_d @ R
            t = R_d @ t + t_d
            src_tf = (src @ R.T) + t
        T = torch.eye(4, device=device, dtype=torch.float32)
        T[:3, :3] = R
        T[:3, 3] = t

        # Compute rmse
        d_final = torch.cdist(src_tf, tgt)
        min_distances = d_final.min(dim=1)[0]
        rmse = torch.sqrt((min_distances ** 2).mean())
        return T.cpu().numpy(), rmse

    def run_icp_pypose(self, src_np, tgt_np, device, init_T_np=None, max_iters=20):
        """
        Simple point-to-point ICP implemented in PyPose.
        Ensures all tensors are on the same device.
        """
        device = torch.device(device)
        src_np = src_np.astype(np.float32, copy=False)
        tgt_np = tgt_np.astype(np.float32, copy=False)
        src = torch.from_numpy(src_np).to(device)
        tgt = torch.from_numpy(tgt_np).to(device)
        if init_T_np is None:
            init_T_pypose = pp.identity_SE3(device=device, dtype=torch.float32)[0]
        else:
            # Convert 4x4 SE(3) matrix to (x, y, z, qx, qy, qz, qw)
            T = init_T_np.astype(np.float32)
            R = T[:3, :3]
            t = T[:3, 3]
            # Quaternion from rotation matrix (xyzw, w last)
            R_np = R
            qw = np.sqrt(1 + R_np[0,0] + R_np[1,1] + R_np[2,2]) / 2
            qx = (R_np[2,1] - R_np[1,2]) / (4*qw)
            qy = (R_np[0,2] - R_np[2,0]) / (4*qw)
            qz = (R_np[1,0] - R_np[0,1]) / (4*qw)
            vec7 = np.array([t[0], t[1], t[2], qx, qy, qz, qw], dtype=np.float32)
            vec7_torch = torch.from_numpy(vec7).to(device)
            init_T_pypose = pp.SE3(vec7_torch)
        stepper = pp.utils.ReduceToBason(steps=max_iters, verbose=False)
        icp = pp.module.ICP(stepper=stepper)
        # Ensure init_T_pypose is on the same device as src/tgt
        init_T_pypose = init_T_pypose.to(device)
        T_pypose = icp(src, tgt, init=init_T_pypose)
        # Convert back to 4x4 SE(3) matrix
        vec7_out = T_pypose.tensor().detach().cpu().numpy().squeeze()
        x, y, z, qx, qy, qz, qw = vec7_out
        # Normalize quaternion
        norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
        qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
        # Rotation matrix from quaternion (w last)
        R = np.array([
            [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw),     1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw),         2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
        ], dtype=np.float32)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]

        # Compute rmse
        src_tf = (src @ torch.from_numpy(R).to(device).T) + torch.from_numpy(np.array([x, y, z], dtype=np.float32)).to(device)
        d_final = torch.cdist(src_tf, tgt)
        min_distances = d_final.min(dim=1)[0]
        rmse = torch.sqrt((min_distances ** 2).mean())
        return T, rmse

    def registration_at_scale(self, scan, map, initial, scale, max_iters=20):
        # Downsample point clouds
        scan_down = self.voxel_down_sample(scan, self.get_parameter("scan_voxel_size").value * scale)
        map_down = self.voxel_down_sample(map, self.get_parameter("map_voxel_size").value * scale)
        scan_np = np.asarray(scan_down.points)
        map_np = np.asarray(map_down.points)
        # Use initial as 4x4 numpy
        init_T = initial.astype(np.float32) if isinstance(initial, np.ndarray) else np.eye(4, dtype=np.float32)
        icp_method = self.get_parameter("icp_method").value.lower()
        if icp_method == "torch":
            device = self.get_torch_device()
            self.get_logger().info(f"Using PyTorch ICP on device: {device}")
            T, rmse = self.run_icp_torch(scan_np, map_np, device, init_T_np=init_T, max_iters=max_iters)
            fitness = 1.0
        elif icp_method == "pypose":
            device = self.get_torch_device()
            self.get_logger().info(f"Using PyPose ICP on device: {device}")
            T, rmse = self.run_icp_pypose(scan_np, map_np, device, init_T_np=init_T, max_iters=max_iters)
            fitness = 1.0
        else:
            self.get_logger().info("Using Open3D ICP")
            threshold = 1.0 * scale
            reg = o3d.pipelines.registration.registration_icp(
                scan_down, map_down, threshold, init_T,
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            T = reg.transformation
            fitness = reg.fitness
            rmse = reg.inlier_rmse
        
        # # Update previous pose if rmse is acceptable
        # if rmse < 0.19:
        #     self.prev_pose = T
        
        return T, fitness, rmse

    def inverse_se3(self, trans):
        trans_inverse = np.eye(4)
        # R
        trans_inverse[:3, :3] = trans[:3, :3].T
        # t
        trans_inverse[:3, 3] = -np.matmul(trans[:3, :3].T, trans[:3, 3])
        return trans_inverse

    def publish_point_cloud(self, publisher, header, pc):
        # Ensure pc is at least 2D
        if pc.ndim == 1:
            pc = pc.reshape(-1, 3)
        
        # Create structured array for ros2_numpy.msgify
        if pc.shape[1] >= 4:
            # Has intensity
            cloud_arr = np.zeros(len(pc), dtype=[
                ('x', np.float32),
                ('y', np.float32),
                ('z', np.float32),
                ('intensity', np.float32)
            ])
            cloud_arr['x'] = pc[:, 0]
            cloud_arr['y'] = pc[:, 1]
            cloud_arr['z'] = pc[:, 2]
            cloud_arr['intensity'] = pc[:, 3]
        else:
            # Only xyz
            cloud_arr = np.zeros(len(pc), dtype=[
                ('x', np.float32),
                ('y', np.float32),
                ('z', np.float32)
            ])
            cloud_arr['x'] = pc[:, 0]
            cloud_arr['y'] = pc[:, 1]
            cloud_arr['z'] = pc[:, 2]
        
        msg = ros2_numpy.msgify(PointCloud2, cloud_arr)
        msg.header = header
        publisher.publish(msg)
        
    def crop_global_map_in_FOV(self, pose_estimation):
        T_odom_to_base_link = self.pose_to_mat(self.cur_odom.pose.pose)
        T_map_to_base_link = np.matmul(pose_estimation, T_odom_to_base_link)
        T_base_link_to_map = self.inverse_se3(T_map_to_base_link)

        global_map_in_map = np.array(self.global_map.points)
        global_map_in_map = np.column_stack([global_map_in_map, np.ones(len(global_map_in_map))])
        global_map_in_base_link = np.matmul(T_base_link_to_map, global_map_in_map.T).T

        if self.get_parameter("fov").value > 3.14:
            indices = np.where(
                (global_map_in_base_link[:, 0] < self.get_parameter("fov_far").value)
                & (np.abs(np.arctan2(global_map_in_base_link[:, 1], global_map_in_base_link[:, 0])) < self.get_parameter("fov").value / 2.0)
            )
        else:
            indices = np.where(
                (global_map_in_base_link[:, 0] > 0)
                & (global_map_in_base_link[:, 0] < self.get_parameter("fov_far").value)
                & (np.abs(np.arctan2(global_map_in_base_link[:, 1], global_map_in_base_link[:, 0])) < self.get_parameter("fov").value / 2.0)
            )
        global_map_in_FOV = o3d.geometry.PointCloud()
        global_map_in_FOV.points = o3d.utility.Vector3dVector(np.squeeze(global_map_in_map[indices, :3]))

        header = self.cur_odom.header
        header.frame_id = "map"
        self.publish_point_cloud(self.pub_submap, header, np.array(global_map_in_FOV.points)[::10])

        return global_map_in_FOV

    def global_localization(self, pose_estimation, first_pose=False):
        scan_tobe_mapped = copy.copy(self.cur_scan)
        global_map_in_FOV = self.crop_global_map_in_FOV(pose_estimation)
        t0 = time.time()
        transformation, fitness, rmse = self.registration_at_scale(scan_tobe_mapped, global_map_in_FOV, initial=pose_estimation, scale=1, max_iters=20)
        t1 = time.time()
        self.get_logger().info(f"Global localization took {t1 - t0:.2f} seconds, fitness: {fitness:.4f}, rmse: {rmse:.4f}")

        # Calculate translation delta (vs previous accepted pose) and plot
        delta_trans = 0.0
        if self.prev_pose is not None:
            try:
                delta_trans = float(np.linalg.norm(transformation[:3, 3] - self.prev_pose[:3, 3]))
                self.get_logger().info(f"Translation delta from previous pose: {delta_trans:.4f} m")
            except Exception:
                delta_trans = 0.0

        # Optional plotting
        self._maybe_update_rmse_plot(rmse, delta_trans=delta_trans, stamp=t1)
        
        if first_pose and fitness > self.get_parameter("localization_threshold").value:
            self.T_map_to_odom = transformation
            self.prev_pose = transformation
            self.publish_odom(transformation)
            return
        if fitness > self.get_parameter("localization_threshold").value and delta_trans < 0.15:
            self.T_map_to_odom = transformation
            self.prev_pose = transformation
            self.publish_odom(transformation)
            self.get_logger().info("\033[92mLocalization accepted.\033[0m")
        elif fitness > self.get_parameter("localization_threshold").value and delta_trans > 0.15:
            if self.prev_pose is None:
                self.get_logger().warn(
                    "Localization error too large and prev_pose is None; skipping odom publish this cycle."
                )
                return
            self.T_map_to_odom = self.prev_pose
            self.publish_odom(self.prev_pose)
            self.get_logger().warn("\033[93mFalling back to previous pose\033[0m")
        else:
            self.get_logger().warn(
                "\033[91m"
                f"Fitness score {fitness} less than localization threshold {self.get_parameter('localization_threshold').value} "
                f"or error {delta_trans} too high; localization rejected."
                "\033[0m"
            )

    def _maybe_update_rmse_plot(self, rmse, delta_trans=0.0, stamp=None):
        """Non-blocking RMSE plot for debugging.

        Plots are throttled by plot_rmse_rate and keep a sliding history.
        """
        if not self.get_parameter("plot_rmse").value:
            return
        if plt is None:
            return

        now = time.time() if stamp is None else float(stamp)
        max_h = int(self.get_parameter("plot_rmse_history").value)
        self._rmse_hist.append(float(rmse))
        self._rmse_t_hist.append(now)
        self._dtrans_hist.append(float(delta_trans))
        if max_h > 0 and len(self._rmse_hist) > max_h:
            extra = len(self._rmse_hist) - max_h
            del self._rmse_hist[:extra]
            del self._rmse_t_hist[:extra]
            del self._dtrans_hist[:extra]

        rate_hz = float(self.get_parameter("plot_rmse_rate").value)
        if rate_hz <= 0:
            rate_hz = 1.0
        if (now - self._rmse_plot_last_t) < (1.0 / rate_hz):
            return
        self._rmse_plot_last_t = now

        # Lazy-init plot
        if self._rmse_fig is None:
            plt.ion()
            self._rmse_fig, self._rmse_ax = plt.subplots(num="ICP RMSE")
            self._rmse_line, = self._rmse_ax.plot([], [], "b-", linewidth=2, label="rmse")
            self._rmse_ax2 = self._rmse_ax.twinx()
            self._dtrans_line, = self._rmse_ax2.plot([], [], "r-", linewidth=2, label="Δtranslation")

            self._rmse_ax.set_title("ICP RMSE + Δtranslation")
            self._rmse_ax.set_xlabel("samples")
            self._rmse_ax.set_ylabel("rmse")
            self._rmse_ax2.set_ylabel("Δtranslation (m)")
            self._rmse_ax.grid(True, alpha=0.3)

            lines = [self._rmse_line, self._dtrans_line]
            labels = [l.get_label() for l in lines]
            self._rmse_ax.legend(lines, labels, loc="upper right")

        xs = list(range(len(self._rmse_hist)))
        self._rmse_line.set_data(xs, self._rmse_hist)
        if self._dtrans_line is not None:
            self._dtrans_line.set_data(xs, self._dtrans_hist)
        if len(xs) > 0:
            self._rmse_ax.set_xlim(0, max(xs))
            y_min = min(self._rmse_hist)
            y_max = max(self._rmse_hist)
            pad = 0.05 * (y_max - y_min + 1e-6)
            self._rmse_ax.set_ylim(y_min - pad, y_max + pad)

            if self._rmse_ax2 is not None and len(self._dtrans_hist) > 0:
                dy_min = min(self._dtrans_hist)
                dy_max = max(self._dtrans_hist)
                dpad = 0.05 * (dy_max - dy_min + 1e-6)
                self._rmse_ax2.set_ylim(dy_min - dpad, dy_max + dpad)

        # Draw without blocking ROS callbacks
        self._rmse_fig.canvas.draw_idle()
        self._rmse_fig.canvas.flush_events()

    def voxel_down_sample(self, pcd, voxel_size):
        # print(pcd)
        
        try:
            pcd_down = pcd.voxel_down_sample(voxel_size)
        
        except Exception as e:
            # for opend3d 0.7 or lower
            pcd_down = o3d.geometry.voxel_down_sample(pcd, voxel_size)
            
        return pcd_down

    def cb_save_cur_odom(self, msg):
        self.cur_odom = msg
        
    def cb_save_cur_scan(self, msg):
        pc = self.msg_to_array(msg)
        self.cur_scan = o3d.geometry.PointCloud()
        self.cur_scan.points = o3d.utility.Vector3dVector(pc)
        self.publish_point_cloud(self.pub_pc_in_map, msg.header, pc)
        
    def initialize_global_map(self): #, pc_msg):
        # self.global_map = o3d.geometry.PointCloud()
        # self.global_map.points = o3d.utility.Vector3dVector(self.msg_to_array(pc_msg)[:, :3])
        self.global_map = o3d.io.read_point_cloud(self.get_parameter("pcd_map_path").value)
        self.global_map = self.voxel_down_sample(self.global_map, self.get_parameter("map_voxel_size").value)
        # o3d.io.write_point_cloud("/home/wheelchair2/laksh_ws/pcds/lab_map_with_outside_corridor (with ground pcd)_downsampled.pcd", self.global_map)
        self.get_logger().info("Global map received.")

    def cb_initialize_pose(self, msg):
        initial_pose = self.pose_to_mat(msg.pose.pose)
        self.initialized = True
        self.get_logger().info("Initial pose received.")
        
        if self.cur_scan is not None:
            self.global_localization(initial_pose, first_pose=True)
            
    def publish_odom(self, transform):
        if transform is None:
            self.get_logger().warn("publish_odom called with None transform; skipping publish.")
            return
        odom_msg = Odometry()
        xyz = transform[:3, 3]
        quat = tf_transformations.quaternion_from_matrix(transform)
        odom_msg.pose.pose = Pose(
            position=Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2])),
            orientation=Quaternion(x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3]))
        )
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = "map"
        self.pub_map_to_odom.publish(odom_msg)

    def localisation_timer_callback(self):
        if not self.initialized:
            self.get_logger().info("Waiting for initial pose...")
            return
        
        if self.cur_scan is not None:
            self.global_localization(self.T_map_to_odom, first_pose=False)


def main(args=None):
    rclpy.init(args=args)
    node = FastLIOLocalization()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()