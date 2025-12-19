import time
import torch


def time_op(fn, n_iters=50, device="cpu", synchronize=False):
    """
    Time a callable `fn` over `n_iters` iterations.
    If `synchronize` is True, calls torch.cuda.synchronize() before and after timing.
    """
    if synchronize:
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iters):
        fn()
    if synchronize:
        torch.cuda.synchronize()
    end = time.perf_counter()

    total = end - start
    return total / n_iters  # average seconds per iter


def benchmark_matmul(device, size=4096):
    """
    Benchmark a large matrix multiplication on the given device.
    """
    print(f"\n[Matmul] Device: {device}, size: {size}x{size}")

    # Create tensors
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # Warm up (especially important for GPU)
    for _ in range(10):
        c = a @ b

    synchronize = device.startswith("cuda")

    def op():
        # matmul
        c = a @ b
        # prevent optimization away
        return c

    avg_sec = time_op(op, n_iters=20, device=device, synchronize=synchronize)
    print(f"  Avg time: {avg_sec * 1000:.3f} ms  ({1.0 / avg_sec:.2f} iters/sec)")
    return avg_sec


def benchmark_mlp(device, batch_size=4096, input_dim=1024, hidden_dim=2048, output_dim=1000):
    """
    Benchmark a simple 3-layer MLP forward pass.
    """
    print(f"\n[MLP] Device: {device}, batch_size={batch_size}, dims={input_dim}->{hidden_dim}->{hidden_dim}->{output_dim}")

    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, output_dim),
    ).to(device)

    x = torch.randn(batch_size, input_dim, device=device)

    # Warm up
    for _ in range(10):
        y = model(x)

    synchronize = device.startswith("cuda")

    def op():
        y = model(x)
        return y

    avg_sec = time_op(op, n_iters=50, device=device, synchronize=synchronize)
    print(f"  Avg time: {avg_sec * 1000:.3f} ms  ({1.0 / avg_sec:.2f} iters/sec)")
    return avg_sec


def main():
    print("PyTorch version:", torch.__version__)
    has_cuda = torch.cuda.is_available()
    print("CUDA available:", has_cuda)

    devices = ["cpu"]
    if has_cuda:
        devices.append("cuda")

    results = {}

    # Run benchmarks
    for d in devices:
        matmul_time = benchmark_matmul(d, size=2048)  # adjust size if OOM/too slow
        mlp_time = benchmark_mlp(d)
        results[d] = {
            "matmul_sec": matmul_time,
            "mlp_sec": mlp_time,
        }

    # Pretty comparison (if GPU exists)
    if has_cuda:
        print("\n" + "=" * 60)
        print("Summary: CPU vs GPU")
        print("=" * 60)
        cpu = results["cpu"]
        gpu = results["cuda"]

        def speedup(cpu_t, gpu_t):
            return cpu_t / gpu_t if gpu_t > 0 else float("inf")

        print(f"Matmul: CPU {cpu['matmul_sec']*1000:.3f} ms vs GPU {gpu['matmul_sec']*1000:.3f} ms "
              f"(speedup x{speedup(cpu['matmul_sec'], gpu['matmul_sec']):.2f})")

        print(f"MLP:    CPU {cpu['mlp_sec']*1000:.3f} ms vs GPU {gpu['mlp_sec']*1000:.3f} ms "
              f"(speedup x{speedup(cpu['mlp_sec'], gpu['mlp_sec']):.2f})")
    else:
        print("\nNo CUDA GPU detected; only CPU results available.")


if __name__ == "__main__":
    main()
