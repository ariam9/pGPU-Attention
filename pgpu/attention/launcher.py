import argparse
import importlib
from pathlib import Path

import numpy as np

from pgpu.attention.common import comparison_metrics, reference_attention


def list_kernels():
    #discover kernels
    names = []
    for path in sorted(Path(__file__).parent.glob("*.py")):
        if path.stem.startswith("_") or path.stem == "launcher":
            continue
        module = importlib.import_module(f"pgpu.attention.{path.stem}")
        if callable(getattr(module, "run_attention_kernel", None)):
            names.append(path.stem)
    return names


def load_kernel(name):
    if name.endswith(".py"):
        name = name[:-3]
    if not name.isidentifier() or name not in list_kernels():
        raise ValueError(f"Unknown attention kernel {name!r}. Choose from: {', '.join(list_kernels())}")
    return importlib.import_module(f"pgpu.attention.{name}").run_attention_kernel


def compare(A, A_, atol=1e-4, rtol=1e-3, max_fail_frac=0.001):
    metrics = comparison_metrics(A, A_, atol, rtol, max_fail_frac)
    status = "PASS" if metrics["passed"] else "FAIL"
    print(f"{status}: {metrics['n_fail']}/{metrics['size']} elements "
          f"({metrics['fail_fraction']:.4%}) exceed tol "
          f"(atol={atol}, rtol={rtol}), max diff={metrics['max_abs_error']:.3e}")
    if metrics["reason"]:
        print(f"  {metrics['reason']}")
    return metrics["passed"]


def launch_kernel(B, H, W, debug=False, tracing=False, schedule=False, kernel="onlyregs_softmax", seed=42, input_scale=100.0, trace_dir="artifacts/traces"):
    if min(B, H, W) <= 0:
        raise ValueError("B, H, and W must be positive")
    if not np.isfinite(input_scale) or input_scale < 0:
        raise ValueError("input_scale must be finite and nonnegative")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    run = load_kernel(kernel)
    rng = np.random.default_rng(seed=seed)
    Q = rng.random(size=(B, H, 8, 32), dtype=np.float32) * input_scale
    K = rng.random(size=(B, 8, 32), dtype=np.float32) * input_scale
    expected = reference_attention(Q, K)
    actual, cycles = run(Q, K, num_warps=W, debug=debug, tracing=tracing,
                         schedule=schedule, trace_dir=trace_dir)
    return expected, actual, cycles


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", default="onlyregs_softmax", help="kernel filename or module stem")
    parser.add_argument("--list-kernels", action="store_true")
    parser.add_argument("-B", "--batches", "--batchs", type=positive_int, default=1)
    parser.add_argument("-H", "--heads", type=positive_int, default=1)
    parser.add_argument("-W", "--warps", type=positive_int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-scale", type=float, default=100.0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--trace", "--tracer", dest="tracing", action="store_true")
    parser.add_argument("--trace-dir", default="artifacts/traces")
    parser.add_argument("--schedule", action="store_true", help="enable instruction scheduler")
    args = parser.parse_args(argv)
    if args.list_kernels:
        print("\n".join(list_kernels()))
        return 0
    try:
        expected, actual, cycles = launch_kernel(
            args.batches, args.heads, args.warps, args.debug, args.tracing, args.schedule,
            args.kernel, args.seed, args.input_scale, args.trace_dir,
        )
    except (ValueError, MemoryError) as exc:
        parser.error(str(exc))
    print(f"Kernel: {args.kernel.removesuffix('.py')}")
    print(f"B={args.batches} H={args.heads} W={args.warps} N=8 d=32 "
          f"seed={args.seed} input_scale={args.input_scale:g} schedule={args.schedule}")
    print(f"Number of cycles: {cycles}")
    return 0 if compare(expected, actual) else 1


if __name__ == "__main__":
    raise SystemExit(main())
