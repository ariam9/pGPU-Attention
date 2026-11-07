# Analysis

## Roofline analysis

theoretical max compute is about 256 FLOPs/cycle for matrix multiplication:
2048 FLOPs in 8 cycles on the MMA unit.

for memory, stage 2 of the memory requests dominates. 8 words every
50 cycles per active channel, across 4 channels:

```text
bandwidth = (4 channels * 8 words * 4 bytes) / 50 cycles
          = 2.56 bytes/cycle
```

operation count is:

```text
FLOPs = 4480 * B * H
4480 = 2048 * 2 + 64 * 6 [matmul + softmax]
```

loads and stores are independent and can overlap. since the stores are less, the estimate uses only the loads:

```text
key loads:    B * 256 words
query loads:  B * H * 256 words
score stores: B * H * 64 words

load words = B * 256 + B * H * 256
AI = (4480 * B * H) / [(B * 256 + B * H * 256) * 4]
roofline = min(2.56 * AI, 256)
```

for B = 16, H = 64:

```text
AI approximately 4.31
speed-of-light estimate = 416,000 cycles
```

## Baseline

the main workload is B = 16, H = 64. the warps work on the (Q, K) of one batch separately/independently.

each warp maintains Q1, Q2, K1, K2 of dimensions 8x16, and A of dimensions 8x8. Q and K are split into halves, and the matmul is performed using `Q1 * K1.T + Q2 * K2.T`.

the DRAM to SRAM load is coalesced for the DRAM read and conflict free for the SRAM write. the baseline has separate loading, matmul, scaling, softmax and manager routines. these create their own register contexts, which leads to thinking about register pressure and kernel fusion type improvements.

early baseline measurement for B = 16, H = 64: **86,985,630 cycles**.

## Optimisation 1: only registers + batches separated

the matrices are small, so keep them in registers. each warp focuses on its
own batch and goes through heads one by one, reusing the keys across heads.
shared memory is only used for softmax reductions. contiguous groups of
8 lanes hold 8 contiguous attention-score elements, so the reduction takes
3 steps over those threads' values in SRAM.

### Driver context-switch fixes

the driver was saving the outgoing warp's vector and predicate registers to SRAM and restoring the incoming warp's registers from SRAM on each switch.
so switching warps to hide a DRAM stall ended up adding a whole lot of SRAM loads/stores and waiting for them instead!

the warps already have their own register state in the simulator, so this save/restore was unnecessary. removing it keeps the registers resident and the switch itself adds no extra cycles. this explains why 4 warps were so much worse than 1 before the fix, and why the single warp result is unchanged, there's no other warp to switch to.

| Warps | Before driver context-switch fixes | After fixes |
| --- | ---: | ---: |
| 1 | 1,364,017 | 1,364,017 |
| 4 | 86,046,454 | 1,261,152 |

initially, the observation was that the warps end up loading from DRAM at around the same time, so context switches don't hide the latency as well as expected. this motivated an instruction scheduler, warp staggering and more GPU-aware memory-dependence analysis.

### The softmax problem

the devtool results after v1 scheduling showed significant SRAM stalls:

| Reported quantity | Value |
| --- | ---: |
| Total cycles | 1,206,980 |
| Instructions issued | 121,536 |
| Average IPC | 0.1007 |
| Cumulative DRAM cycles | 2,477,564 |
| Cumulative SRAM cycles | 944,128 |

some recurring gaps were `sram.ld -> sram.ld` (6 cycles), `sram.st -> sram.ld` (14 cycles), and `sram.ld -> sub` (70 cycles).

the big problem seemed to be softmax reduction, particularly SRAM port contention and the lack of bank broadcasting(which was falsely assumed to exist for this implementation).

rough SRAM accounting per attention-score register was:

```text
6 + (14 + 14 + 6) * 3 + 70 + 6 + (14 + 14 + 6) * 3 + 70
= 356 cycles
```

each thread has 2 attention-score registers, so about 712 cycles per score matrix, or `16 * 64 * 712 = 729,088` cycles over the kernel. 

## Optimisation 2: MMA-based softmax

can we use the MMA operations for reductions instead of SRAM?

- min/max reductions are basically matmul with the sums replaced by min/max.
- attention scores with a matrix of ones can give a row-axis max reduction.
- regular matmul with a matrix of ones can give a row-axis sum.
- A is 8x8, so pad it to 8x16 using negative infinity for max and 0 for sum.

the resulting softmax is:

```python
kb.asm("avx.mmred.max", (r_tmp1, r_tmp2), [d_regs[0], d_regs[1], r_neg_inf, r_neg_inf], [r_1, r_1, r_1, r_1], (r_neg_inf, r_neg_inf))
kb.asm("sub", d_regs[0], d_regs[0], r_tmp1)
kb.asm("sub", d_regs[1], d_regs[1], r_tmp2)

kb.asm("exp.trans", d_regs[0], d_regs[0])
kb.asm("exp.trans", d_regs[1], d_regs[1])

kb.asm("avx.mma", (r_tmp1, r_tmp2), [d_regs[0], d_regs[1], kb.rz, kb.rz], [r_1, r_1, r_1, r_1], (kb.rz, kb.rz))
kb.asm("div", d_regs[0], d_regs[0], r_tmp1)
kb.asm("div", d_regs[1], d_regs[1], r_tmp2)
```

this removes SRAM use entirely!

the reduction from `onlyregs` to `onlyregs_softmax` is around 800k cycles, with the extra difference beyond the SRAM estimate possibly coming from other latency effects.

the workload recorded for these measurements is **B = 16, H = 64**:

| Warps | Without v1 scheduler | With v1 scheduler |
| --- | ---: | ---: |
| 1 | 592,945 | 593,921 |
| 4 | 418,265 | 418,071 |

the scheduler slightly helps with 4 warps, but causes a regression with 1.

## Optimisation 3: less sequential attention-score writes

after changing the attention-score writes, to reduce sequentiality in dram writes(earlier all writes happened to the same channel at once):

| Warps | Without scheduler | With scheduler |
| --- | ---: | ---: |
| 1 | 599,913 | 598,840 |
| 4 | 417,694 | 417,803 |

current best: **417,694 cycles** 

the scheduler doesn't improve this version with 4 warps.

## Instruction scheduler: current approach and limitations

v1 is a basic list scheduler. it divides the program into straight-line blocks at terminators like jumps, syncs and yields. within each block, it calculates dependencies and does a topological sort with a priority heuristic.

it doesn't rename registers or look at the addresses of DRAM/SRAM memory operations yet, so there's room for improvement.

the latency assumptions(for the priority heuristic) are:

| Dependence | Latency / reason |
| --- | --- |
| Register RAW | Producer latency |
| Register WAW | Previous producer latency: issue waits for the previous write |
| Register WAR | 1 cycle: read operands are latched on issue |
| Memory RAW | Producer latency: 14 cycles for SRAM, 90 for DRAM |
| Memory WAW | 1 cycle: in-order store queue |
| Memory WAR | 1 cycle: due to issue order preservation |

VALU, TRANS and MMA are non-pipelined and each has one issue slot. the scheduler doesn't account for this currently. 

memory issue ports also need modelling to avoid losing cycles to contention. context-switch modelling needs more experimentation.

## Things still worth exploring

- software pipelining through loop unrolling: try it manually first and see if the scheduler helps. a more aggressive version would need register renaming, alias analysis, issue-port modelling and a tile abstraction for memory accesses/base addresses(though this won't help with the current amount of parallelism available)
- integer programming for instruction scheduling in python, just as an exploration
- yielding on significant forced stalls, and whether warp staggering can improve latency hiding.
- alternatives to round-robin warp scheduling in the driver, keeping track of the information and computation that the scheduling decision needs.
- more realistic devtools: maybe an `RDTSC`-like instruction to read cycles, instead of relying on trace information that is not available on real hardware, and is thus unrealistic.

## Conclusion
we started at around 87 million cycles and got down to 417,694 cycles by keeping the matrices in registers, reusing keys across heads, removing the unnecessary context saves/restores, using MMA operations for softmax reductions, improving coalescing and building an instruction scheduler to provide (marginal)benefits by reordering instructions.

the roofline estimate is 416,000 cycles, so we're within about 0.4% of it, basically at speed of light for this workload on the simulator. 

at this point we're limited by DRAM load bandwidth, so better scheduling and overlapping computation with memory have very little room to improve this workload further.
