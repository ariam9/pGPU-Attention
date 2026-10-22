import numpy as np

from pgpu.attention.baseline import run_attention_kernel

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


def launch_kernel(B=1, H=1, tracer=False, debug=False):
    rng = np.random.default_rng(seed=42)
    Q = rng.random(size=(B, H, 8, 32), dtype=np.float32) * 100
    K = rng.random(size=(8, 32), dtype=np.float32) * 100

    scores = (Q @ K.T) * (1 / np.sqrt(32)).astype(np.float32)
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    A = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    A_, num_cycles = run_attention_kernel(Q, K, tracer=tracer, debug=debug)

    return A, A_, num_cycles


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-B", "--batchs", type=int, default=1, help="batch size")
    parser.add_argument("-H", "--heads", type=int, default=1, help="number of heads")
    parser.add_argument("--tracer", type=bool, default=False, help="tracer on/off")
    parser.add_argument("--debug", type=bool, default=False, help="debug on/off")
    args = parser.parse_args()

    B = args.batchs
    H = args.heads
    tracer = args.tracer
    debug = args.debug

    A, A_, num_cycles = launch_kernel(B, H, tracer, debug)

    print(f"Number of cycles: {num_cycles}")
    print(compare(A, A_))

