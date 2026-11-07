**CS6023 course project - pGPU Attention**

Attention-score matrix computation on the CPU-based pGPU simulator:

```text
A[b, h] = softmax(Q[b, h] @ K[b].T / sqrt(32))
```

Q has shape `(B, H, 8, 32)`, K has shape `(B, 8, 32)`, and A has shape
`(B, H, 8, 8)`. 

The problem workload is B=16, H=64 with four warps.

### setup:
python 3.10+, from the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### running:
the launcher reports simulated cycles and checks correctness against a reference numpy baseline:

```sh
python -m pgpu.attention.launcher --list-kernels
python -m pgpu.attention.launcher --kernel onlyregs_softmax -B 16 -H 64 -W 4
```

Available kernels are: `baseline`, `onlyregs`, and `onlyregs_softmax`, each is a product of successive iteration.

Add `--schedule` to enable instruction reordering, by a custom scheduling engine which should hopefully decrease cycle counts under the specified workloads. It is off by default. 

Use `--help` for input controls and other options.

Correctness failures return a nonzero exit code.

### tests:

```sh
python -m unittest discover -s pgpu/tests -v
python -m unittest discover -s pgpu/tests/pGPUInfra -v
```

### tracing:
json and html are saved under `artifacts/traces/`:
```sh
python -m pgpu.attention.launcher --kernel onlyregs_softmax -B 1 -H 1 -W 1 --trace
python -m pgpu.devtools.summary artifacts/traces/onlyregs_softmax_B1_H1_W1_schedule0.json
```

Find [here](analysis.md) my analysis for the project!

See [the project description](docs/ATTENTION_ASM.md) for architecture and ISA details. 

Original simulator and documentation by Arjun Menon Vadakkeveedu
([upstream pGPU](https://github.com/arjunmenonv/pGPU)).
