from numbers import Integral
from pathlib import Path

import numpy as np


def validate_inputs(Q, K, num_warps=4, device=None):
    if isinstance(num_warps, bool) or not isinstance(num_warps, Integral) or num_warps <= 0:
        raise ValueError("num_warps must be a positive integer")
    Q = np.asarray(Q, dtype=np.float32)
    K = np.asarray(K, dtype=np.float32)
    if Q.ndim != 4 or K.ndim != 3:
        raise ValueError("Expected Q with shape (B, H, 8, 32) and K with shape (B, 8, 32)")
    if Q.shape[0] != K.shape[0]:
        raise ValueError("Q and K must have the same batch size")
    if Q.shape[0] <= 0 or Q.shape[1] <= 0:
        raise ValueError("Batch size and number of heads must be positive")
    if Q.shape[2:] != (8, 32) or K.shape[1:] != (8, 32):
        raise ValueError("These kernels support exactly 8 query/key tokens and dimension 32")
    if not np.isfinite(Q).all() or not np.isfinite(K).all():
        raise ValueError("Q and K must contain only finite values")
    if device is not None and device.total_warps != num_warps:
        raise ValueError("num_warps must match the supplied device's total_warps")
    return Q, K


def reference_attention(Q, K):
    #reference numpy implementation of the attention score calculations(float32)
    scores = (Q @ K[:, None, :, :].swapaxes(-1, -2)) * np.float32(1 / np.sqrt(32))
    exponentials = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def comparison_metrics(expected, actual, atol=1e-4, rtol=1e-3, max_fail_frac=0.001):
    expected, actual = np.asarray(expected), np.asarray(actual)
    result = {"passed": False, "n_fail": 0, "size": int(expected.size),
              "fail_fraction": 1.0, "max_abs_error": float("inf"), "reason": ""}
    if expected.shape != actual.shape:
        result["reason"] = f"shape mismatch: expected {expected.shape}, got {actual.shape}"
        return result
    if expected.size == 0:
        result["reason"] = "empty outputs cannot be validated"
        return result
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        invalid = ~np.isfinite(expected) | ~np.isfinite(actual)
        result.update(n_fail=int(invalid.sum()), fail_fraction=float(invalid.mean()))
        result["reason"] = "reference or kernel output contains NaN/Inf"
        return result
    difference = np.abs(expected - actual)
    failed = difference > (atol + rtol * np.abs(expected))
    result.update(n_fail=int(failed.sum()), fail_fraction=float(failed.mean()),
                  max_abs_error=float(difference.max()))
    if np.any(actual < -atol) or np.any(actual > 1 + atol):
        result["reason"] = "attention probabilities are outside [0, 1]"
    elif not np.allclose(actual.sum(axis=-1), 1.0, atol=1e-5, rtol=1e-5):
        result["reason"] = "attention rows do not sum to one"
    elif result["fail_fraction"] > max_fail_frac:
        result["reason"] = "too many elements exceed the numerical tolerance"
    else:
        result["passed"] = True
    return result


def trace_output_base(kernel_file, B, H, num_warps, schedule, trace_dir):
    directory = Path(trace_dir)
    directory.mkdir(parents=True, exist_ok=True)
    name = f"{Path(kernel_file).stem}_B{B}_H{H}_W{num_warps}_schedule{int(schedule)}"
    return str(directory / name)
