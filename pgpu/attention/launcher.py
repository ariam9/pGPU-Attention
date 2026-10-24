import numpy as np

from pgpu.attention.baseline import run_attention_kernel as baseline_kernel
from pgpu.attention.onlyregs import run_attention_kernel as onlyregs_kernel

def compare(A, A_, atol=1e-4, rtol=1e-3, max_fail_frac=0.001):
    diff = np.abs(A - A_)
    tol = atol + rtol * np.abs(A)
    fail_mask = diff > tol
    n_fail = int(fail_mask.sum())
    frac_fail = n_fail / A.size
    max_diff = float(diff.max())

    ok = frac_fail <= max_fail_frac
    status = "PASS" if ok else "FAIL"
    print(f"{status}: {n_fail}/{A.size} elements ({frac_fail:.4%}) exceed tol "
          f"(atol={atol}, rtol={rtol}), max diff={max_diff:.3e}")
    if n_fail:
        worst = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"  worst element at {worst}: expected={A[worst]:.6f} got={A_[worst]:.6f}")
    return ok


def launch_kernel(B, H, W, debug=False, tracing=False):
    rng = np.random.default_rng(seed=42)
    Q = rng.random(size=(B, H, 8, 32), dtype=np.float32) * 100
    K = rng.random(size=(B, 8, 32), dtype=np.float32) * 100

    KT = K[:, None, :, :].swapaxes(-1, -2)
    scores = (Q @ KT) * (1 / np.sqrt(32)).astype(np.float32)
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    A = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    A_, num_cycles = onlyregs_kernel(Q, K, num_warps=W, debug=debug, tracing=tracing)

    return A, A_, num_cycles


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-B", "--batchs", type=int, default=1, help="batch size")
    parser.add_argument("-H", "--heads", type=int, default=1, help="number of heads")
    parser.add_argument("-W", "--warps", type=int, default=4, help="number of warps")
    parser.add_argument("--debug", action='store_true', help="debug on/off")
    parser.add_argument("--tracer", action='store_true', help="tracer on/off")
    args = parser.parse_args()

    B = args.batchs
    H = args.heads
    W = args.warps
    debug = True if args.debug else False
    tracing = True if args.tracer else False

    A, A_, num_cycles = launch_kernel(B, H, W, debug, tracing)

    print(f"Number of cycles: {num_cycles}")
    print(compare(A, A_))

