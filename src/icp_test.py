import time
import copy
import numpy as np
import open3d as o3d
import torch
import pypose as pp


# -----------------------------
#  Utilities
# -----------------------------
def get_torch_device():
    if torch.backends.mps.is_available():
        print("Using MPS device")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA device")
        return torch.device("cuda")
    else:
        print("Using CPU (no MPS/CUDA found)")
        return torch.device("cpu")


def random_se3_transform(max_trans=1.0, max_angle_deg=45.0, device="cpu"):
    """
    Sample a random SE(3) transform (R, t).
    Rotation: random axis, angle in [-max_angle_deg, max_angle_deg]
    Translation: uniform in [-max_trans, max_trans] on each axis.
    """
    device = torch.device(device)
    axis = torch.randn(3, device=device)
    axis = axis / axis.norm()
    angle = (torch.rand(1, device=device) - 0.5) * 2.0 * np.deg2rad(max_angle_deg)
    aa = axis * angle  # so3 tangent
    aa = aa.unsqueeze(0)
    r = pp.so3(aa)
    Rq = r.Exp()
    q = Rq.tensor().squeeze(0)
    x, y, z, w = q
    R = torch.tensor([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ], device=device, dtype=torch.float32)
    t = (torch.rand(3, device=device) * 2.0 - 1.0) * max_trans
    return R, t


def se3_to_mat(R, t):
    """SE(3) torch (R, t) -> 4x4 homogeneous matrix."""
    device = R.device
    T = torch.eye(4, device=device, dtype=torch.float32)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert_se3(T):
    """Invert 4x4 SE(3) matrix (torch)."""
    R = T[:3, :3]
    t = T[:3, 3]
    Rinv = R.T
    tinv = -Rinv @ t
    Tinv = torch.eye(4, device=T.device, dtype=torch.float32)
    Tinv[:3, :3] = Rinv
    Tinv[:3, 3] = tinv
    return Tinv


def apply_transform(points_np, R, t):
    """Apply SE(3) torch (R, t) to Nx3 numpy points."""
    R_np = R.detach().cpu().numpy()
    t_np = t.detach().cpu().numpy()
    return (points_np @ R_np.T) + t_np

def se3_mat_to_xyz_quat(T):
    """
    Convert a 4x4 SE(3) matrix (torch or numpy) to (x, y, z, qx, qy, qz, qw).
    """
    if isinstance(T, torch.Tensor):
        T = T.detach().cpu().numpy()
    R = T[:3, :3]
    t = T[:3, 3]

    # Quaternion from rotation matrix (xyzw, w last)
    qw = np.sqrt(1 + R[0,0] + R[1,1] + R[2,2]) / 2
    qx = (R[2,1] - R[1,2]) / (4*qw)
    qy = (R[0,2] - R[2,0]) / (4*qw)
    qz = (R[1,0] - R[0,1]) / (4*qw)
    return np.array([t[0], t[1], t[2], qx, qy, qz, qw], dtype=np.float32)

def xyz_quat_to_se3_mat(vec7):
    """
    Convert (x, y, z, qx, qy, qz, qw) to a 4x4 SE(3) matrix.
    Accepts numpy array or list.
    """
    x, y, z, qx, qy, qz, qw = vec7
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
    return T


def rms_error_torch(src_np, tgt_np, device, sample_size=10_000):
    """
    Compute RMS nearest-neighbor error between two large point clouds.
    Uses torch.cdist on a uniform subsample.
    Ensures float32 for MPS compatibility.
    """
    device = torch.device(device)

    # MPS does NOT support float64
    src_np = src_np.astype(np.float32, copy=False)
    tgt_np = tgt_np.astype(np.float32, copy=False)

    src = torch.from_numpy(src_np).to(device)
    tgt = torch.from_numpy(tgt_np).to(device)

    N = min(sample_size, len(src))
    idx_src = torch.randperm(len(src), device=device)[:N]
    idx_tgt = torch.randperm(len(tgt), device=device)[:N]

    src_s = src[idx_src]
    tgt_s = tgt[idx_tgt]

    d = torch.cdist(src_s, tgt_s)
    nn = d.min(dim=1)[0]
    return float(torch.sqrt((nn**2).mean()).cpu().item())

def plot_registration(src_points, tgt_points, T=None, title=""):
    """
    Visualize source and target point clouds in Open3D.
    If T is provided, transform src_points by T before plotting.
    """
    src_pcd = o3d.geometry.PointCloud()
    tgt_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(src_points)
    tgt_pcd.points = o3d.utility.Vector3dVector(tgt_points)
    src_pcd.paint_uniform_color([1, 0, 0])  # Red: source
    tgt_pcd.paint_uniform_color([0, 1, 0])  # Green: target

    if T is not None:
        src_pcd.transform(T)

    o3d.visualization.draw_geometries([src_pcd, tgt_pcd], window_name=title)


# -----------------------------
#  Data generation
# -----------------------------
def generate_point_cloud(num_points=200_000, noise_std=0.01, bounds=1.0, device="cpu"):
    """
    Returns:
      base_src: (N,3) noisy source points on a cone
      base_tgt: (N+outliers,3) target with outliers
      R0, t0: torch transform such that (ideal) base_tgt_clean = R0 * base_src + t0
    """
    device = torch.device(device)
    # Parameters for the cone
    h = bounds  # height of the cone
    r = bounds  # base radius of the cone

    # Sample uniformly along the height
    z = np.random.uniform(0, h, num_points)
    # For each z, the radius at that height
    local_r = (z / h) * r
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    x = local_r * np.cos(theta)
    y = local_r * np.sin(theta)

    src = np.stack([x, y, z], axis=1).astype(np.float32)
    src_noisy = src + np.random.normal(scale=noise_std, size=src.shape).astype(np.float32)

    # True base transform from src -> tgt
    R0, t0 = random_se3_transform(2, 40, device)
    tgt = apply_transform(src_noisy, R0, t0)

    return src_noisy, tgt, R0, t0

# -----------------------------
#  Open3D ICP
# -----------------------------
def run_icp_open3d(src_np, tgt_np, init_T_np=None, max_corr=0.3, max_iters=50):
    src_pcd = o3d.geometry.PointCloud()
    tgt_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(src_np)
    tgt_pcd.points = o3d.utility.Vector3dVector(tgt_np)

    if init_T_np is None:
        init = np.eye(4, dtype=np.float64)
    else:
        init = init_T_np.astype(np.float64)

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iters)

    t0 = time.perf_counter()
    result = o3d.pipelines.registration.registration_icp(
        src_pcd, tgt_pcd, max_corr, init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria,
    )
    t1 = time.perf_counter()

    return result.transformation, t1 - t0


# -----------------------------
#  Torch ICP
# -----------------------------
def run_icp_torch(src_np, tgt_np, device, init_T_torch=None,
                  max_iters=20):
    """
    Simple point-to-point ICP implemented in PyTorch.

    - Nearest neighbors via torch.cdist (O(N^2)).
    - Runs on the given device (MPS on your Mac).
    - Uses all provided points (no subsampling inside).
    """
    device = torch.device(device)

    # Force float32 for MPS
    src_np = src_np.astype(np.float32, copy=False)
    tgt_np = tgt_np.astype(np.float32, copy=False)

    src_full = torch.from_numpy(src_np).to(device)
    tgt_full = torch.from_numpy(tgt_np).to(device)

    src = src_full
    tgt = tgt_full

    if init_T_torch is None:
        R = torch.eye(3, device=device)
        t = torch.zeros(3, device=device)
        src_tf = src.clone()
    else:
        init_T_torch = init_T_torch.to(device=device, dtype=torch.float32)
        R = init_T_torch[:3, :3]
        t = init_T_torch[:3, 3]
        src_tf = (src @ R.T) + t

    t0 = time.perf_counter()
    for _ in range(max_iters):
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

    t1 = time.perf_counter()

    T = torch.eye(4, device=device, dtype=torch.float32)
    T[:3, :3] = R
    T[:3, 3] = t
    return T.cpu().numpy(), t1 - t0

# -----------------------------
#  PyPose ICP
# -----------------------------
def run_icp_pypose(src_np, tgt_np, device, init_T_torch=None,
                   max_iters=20):
    """
    Simple point-to-point ICP implemented in PyPose.
    """
    device = torch.device(device)

    # Force float32 for MPS
    src_np = src_np.astype(np.float32, copy=False)
    tgt_np = tgt_np.astype(np.float32, copy=False)

    src_full = torch.from_numpy(src_np).to(device)
    tgt_full = torch.from_numpy(tgt_np).to(device)

    src = src_full
    tgt = tgt_full

    if init_T_torch is None:
        init_T_torch = pp.identity_SE3(device=device, dtype=torch.float32)[0]
    else:
        init_T_torch = se3_mat_to_xyz_quat(init_T_torch)
        init_T_torch = pp.SE3(init_T_torch)

    t0 = time.perf_counter()

    stepper = pp.utils.ReduceToBason(steps=max_iters, verbose=True)
    icp = pp.module.ICP(stepper=stepper)
    T = icp(src, tgt, init=init_T_torch)
    T = xyz_quat_to_se3_mat(T.tensor().cpu().numpy())

    t1 = time.perf_counter()

    return T, t1 - t0


# -----------------------------
#  Make an initial guess near ground truth
# -----------------------------
def make_initial_guess(R0, t0, R_pert, t_pert, device,
                       max_angle_deg_noise=10.0, max_trans_noise=0.05):
    """
    We know:
      base_tgt_clean = T0 * base_src
      src_trial      = Tpert * base_src
    So the true transform from src_trial -> base_tgt_clean is:
      T_gt = T0 * inv(Tpert)

    We then perturb T_gt slightly to get T_init, used as ICP initial guess.
    """
    device = torch.device(device)

    # True base transform and trial perturbation
    T0 = se3_to_mat(R0, t0)        # base_src -> base_tgt_clean
    Tpert = se3_to_mat(R_pert, t_pert)  # base_src -> src_trial
    Tpert_inv = invert_se3(Tpert)

    # Ground-truth transform: src_trial -> base_tgt_clean
    T_gt = T0 @ Tpert_inv

    # Small noise transform around identity
    R_noise, t_noise = random_se3_transform(
        max_trans=max_trans_noise,
        max_angle_deg=max_angle_deg_noise,
        device=device,
    )
    T_noise = se3_to_mat(R_noise, t_noise)

    # Initial guess is noisy version of ground truth
    T_init = T_noise @ T_gt

    return T_gt, T_init


# -----------------------------
#  Benchmark
# -----------------------------
def benchmark_icp(num_points=200_000, noise_std=0.01, bounds=1.0, num_trials=5):
    device = get_torch_device()

    print("\nGenerating base clouds...")
    base_src, base_tgt, R0, t0 = generate_point_cloud(
        num_points=num_points,
        noise_std=noise_std,
        bounds=bounds,
        device=device,
    )

    transforms = []
    src_trials = []
    init_guesses_torch = []
    src_icp_trials = []
    tgt_icp_trials = []

    # Pre-generate src_trial and initial guesses
    for i in range(num_trials):
        R_pert, t_pert = random_se3_transform(0.25, 20, device)
        src_tf = apply_transform(base_src, R_pert, t_pert)

        T_gt, T_init = make_initial_guess(R0, t0, R_pert, t_pert, device, 35.0, 2.0)
        transforms.append((R_pert, t_pert))
        src_trials.append(src_tf)
        init_guesses_torch.append(T_init)

        # Subsample points for ICP (same subsets used by Open3D and Torch)
        sample_size = 10000
        N_src_icp = min(sample_size, src_tf.shape[0])
        N_tgt_icp = min(sample_size, base_tgt.shape[0])
        idx_src_icp = np.random.choice(src_tf.shape[0], N_src_icp, replace=False)
        idx_tgt_icp = np.random.choice(base_tgt.shape[0], N_tgt_icp, replace=False)
        src_icp = src_tf[idx_src_icp]
        tgt_icp = base_tgt[idx_tgt_icp]

        src_icp_trials.append(src_icp)
        tgt_icp_trials.append(tgt_icp)

    # ------------------- Open3D phase -------------------
    print("\n=== Open3D ICP ===")
    total_o3d = 0.0
    for i in range(num_trials):
        print(f"\n[Open3D] Trial {i+1}/{num_trials}")
        src_trial = src_trials[i]
        T_init_torch = init_guesses_torch[i]

        src_icp = src_icp_trials[i]
        tgt_icp = tgt_icp_trials[i]

        # plot_registration(src_trial, base_tgt, T=None, title="Before ICP")

        pre_err = rms_error_torch(src_trial, base_tgt, device)

        # Convert initial guess to numpy float64 for Open3D
        init_np = T_init_torch.detach().cpu().numpy().astype(np.float64)
        T, t_icp = run_icp_open3d(src_icp, tgt_icp, init_T_np=init_np)

        # plot_registration(src_trial, base_tgt, T=T, title="After ICP (Open3D)")

        src_after = (src_trial @ T[:3, :3].T) + T[:3, 3]
        post_err = rms_error_torch(src_after, base_tgt, device)

        print(f"  Pre-ICP error:  {pre_err:.4f}")
        print(f"  Post-ICP error: {post_err:.4f}")
        print(f"  ICP time:       {t_icp:.4f} s")

        total_o3d += t_icp

    # ------------------- Torch phase -------------------
    print("\n=== Torch ICP ===")
    total_torch = 0.0
    for i in range(num_trials):
        print(f"\n[Torch] Trial {i+1}/{num_trials}")
        src_trial = src_trials[i]
        T_init_torch = init_guesses_torch[i]

        src_icp = src_icp_trials[i]
        tgt_icp = tgt_icp_trials[i]

        # plot_registration(src_trial, base_tgt, T=None, title="Before ICP")

        pre_err = rms_error_torch(src_trial, base_tgt, device)

        T, t_icp = run_icp_torch(src_icp, tgt_icp,
                                 device=device,
                                 init_T_torch=T_init_torch,
                                 max_iters=50)
        
        # plot_registration(src_trial, base_tgt, T=T, title="After ICP (Torch)")

        src_after = (src_trial @ T[:3, :3].T) + T[:3, 3]
        post_err = rms_error_torch(src_after, base_tgt, device)

        print(f"  Pre-ICP error:  {pre_err:.4f}")
        print(f"  Post-ICP error: {post_err:.4f}")
        print(f"  ICP time:       {t_icp:.4f} s")

        total_torch += t_icp
    
    # ------------------- PyPose phase -------------------
    print("\n=== PyPose ICP ===")
    total_pypose = 0.0
    for i in range(num_trials):
        print(f"\n[PyPose] Trial {i+1}/{num_trials}")
        src_trial = src_trials[i]
        T_init_torch = init_guesses_torch[i]

        src_icp = src_icp_trials[i]
        tgt_icp = tgt_icp_trials[i]

        # plot_registration(src_trial, base_tgt, T=None, title="Before ICP")

        pre_err = rms_error_torch(src_trial, base_tgt, device)

        T, t_icp = run_icp_pypose(src_icp, tgt_icp,
                                  device=device,
                                  init_T_torch=T_init_torch,
                                  max_iters=50)
        
        # plot_registration(src_trial, base_tgt, T=T, title="After ICP (Torch)")

        src_after = (src_trial @ T[:3, :3].T) + T[:3, 3]
        post_err = rms_error_torch(src_after, base_tgt, device)

        print(f"  Pre-ICP error:  {pre_err:.4f}")
        print(f"  Post-ICP error: {post_err:.4f}")
        print(f"  ICP time:       {t_icp:.4f} s")

        total_pypose += t_icp

    print("\n=== Summary ===")
    print(f"Total Open3D ICP time:       {total_o3d:.4f} s")
    print(f"Total Torch ICP time: {total_torch:.4f} s")
    print(f"Total PyPose ICP time: {total_pypose:.4f} s")

if __name__ == "__main__":
    benchmark_icp(num_points=500000, num_trials=1)
