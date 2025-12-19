#!/usr/bin/env python3

"""Executable script to compare /Odometry and /localization as SE(3) deltas.

Run (after sourcing ROS 2 setup):
  python3 compare_odom_localization.py

This subscribes directly (no custom node class) and opens a matplotlib window
showing the norm of translation and rotation increments for both topics.
"""

import math
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

import rclpy
from rclpy.qos import QoSProfile
from nav_msgs.msg import Odometry


def pose_to_T(pose_msg):
    """Convert geometry_msgs/Pose to 4x4 SE(3) matrix."""
    t = np.array(
        [pose_msg.position.x, pose_msg.position.y, pose_msg.position.z],
        dtype=float,
    )
    q = np.array(
        [
            pose_msg.orientation.x,
            pose_msg.orientation.y,
            pose_msg.orientation.z,
            pose_msg.orientation.w,
        ],
        dtype=float,
    )

    norm_q = np.linalg.norm(q)
    if norm_q > 0:
        q = q / norm_q
    x, y, z, w = q

    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def so3_log(R):
    """Log map from SO(3) to R^3."""
    cos_theta = (np.trace(R) - 1.0) * 0.5
    cos_theta = max(min(cos_theta, 1.0), -1.0)
    theta = math.acos(cos_theta)

    if abs(theta) < 1e-9:
        return np.zeros(3)

    omega_hat = (R - R.T) / (2.0 * math.sin(theta))
    return theta * np.array([omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0]])


def se3_log(T):
    """Log map from SE(3) to R^6 (rho, phi)."""
    R = T[:3, :3]
    t = T[:3, 3]
    phi = so3_log(R)
    theta = np.linalg.norm(phi)

    if theta < 1e-9:
        V_inv = np.eye(3)
    else:
        a = phi / theta
        ax = np.array(
            [
                [0, -a[2], a[1]],
                [a[2], 0, -a[0]],
                [-a[1], a[0], 0],
            ]
        )
        A = math.sin(theta) / theta
        B = (1 - math.cos(theta)) / (theta * theta)
        V = np.eye(3) + A * ax + B * (ax @ ax)
        V_inv = np.linalg.inv(V)
    rho = V_inv @ t
    return np.concatenate([rho, phi])  # R^6


def main():
    rclpy.init()
    qos = QoSProfile(depth=10)
    node = rclpy.create_node("compare_odom_localization_script")

    odom_topic = "/Odometry"
    loc_topic = "/localization"

    buffer_size = 1000
    prev_odom_T = {"T": None}
    prev_loc_T = {"T": None}

    odom_deltas = deque(maxlen=buffer_size)
    loc_deltas = deque(maxlen=buffer_size)
    indices = deque(maxlen=buffer_size)
    counter = {"i": 0}

    # Matplotlib setup
    plt.ion()
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.set_ylabel("‖xi‖ (SE3 delta norm)")
    ax.set_xlabel("Step")

    odom_line, = ax.plot([], [], label="odom ‖xi‖")
    loc_line, = ax.plot([], [], label="loc ‖xi‖")
    ax.legend()

    def handle_msg(pose, is_odom: bool):
        T = pose_to_T(pose)

        if is_odom:
            if prev_odom_T["T"] is not None:
                dT = np.linalg.inv(prev_odom_T["T"]) @ T
                xi = se3_log(dT)
                odom_deltas.append(xi)
                indices.append(counter["i"])
                counter["i"] += 1
            prev_odom_T["T"] = T
        else:
            if prev_loc_T["T"] is not None:
                dT = np.linalg.inv(prev_loc_T["T"]) @ T
                xi = se3_log(dT)
                loc_deltas.append(xi)
            prev_loc_T["T"] = T

    def odom_cb(msg: Odometry):
        handle_msg(msg.pose.pose, is_odom=True)

    def loc_cb(msg: Odometry):
        handle_msg(msg.pose.pose, is_odom=False)

    node.create_subscription(Odometry, odom_topic, odom_cb, qos)
    node.create_subscription(Odometry, loc_topic, loc_cb, qos)

    node.get_logger().info(f"Listening to {odom_topic} and {loc_topic}")

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)

            if not indices:
                continue

            # Convert buffers to arrays and align lengths to avoid shape mismatch
            idx = np.array(indices)
            odom_arr = np.array(odom_deltas)
            loc_arr = np.array(loc_deltas)

            # Align odom increments with their indices
            if odom_arr.shape[0] == 0:
                continue
            if odom_arr.shape[0] != idx.shape[0]:
                min_len = min(odom_arr.shape[0], idx.shape[0])
                odom_arr = odom_arr[-min_len:]
                idx_odom = idx[-min_len:]
            else:
                idx_odom = idx

            # Align localization increments with the last part of the index sequence
            if loc_arr.shape[0] > 0:
                if loc_arr.shape[0] <= idx.shape[0]:
                    idx_loc = idx[-loc_arr.shape[0]:]
                else:
                    loc_arr = loc_arr[-idx.shape[0]:]
                    idx_loc = idx
            else:
                idx_loc = np.array([])

            # Full SE3 delta norms
            norm_odom = np.linalg.norm(odom_arr, axis=1)

            if loc_arr.size > 0:
                norm_loc = np.linalg.norm(loc_arr, axis=1)
            else:
                norm_loc = np.array([])

            # Update line data with length-consistent x/y
            odom_line.set_data(idx_odom, norm_odom)

            if norm_loc.size > 0 and idx_loc.size == norm_loc.size:
                loc_line.set_data(idx_loc, norm_loc)

            ax.relim()
            ax.autoscale_view()

            fig.canvas.draw()
            fig.canvas.flush_events()

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
