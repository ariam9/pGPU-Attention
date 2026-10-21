import numpy as np

from pgpu.attention.baseline import run_attention_kernel

def launch_kernel(B=1, H=1, tracer=False, debug=False):
    rng = np.random.default_rng(seed=42)
    Q = rng.random(size=(B, H, 8, 32)) * 100
    K = rng.random(size=(8, 32)) * 100

    scores = Q @ K.T
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

    A, A_, num_cycles = launch(B, H, tracer, debug)

    print(A_)
    print(A)

