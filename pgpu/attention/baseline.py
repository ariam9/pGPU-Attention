import numpy as np

from pgpu.arch.gpu_device import pGPU, CopyDirection
from pgpu.sw.kernel import build_kernel
from pgpu.devtools.kernel_tracer import KernelTracer
from pgpu.sw.intrinsics import mma_8x16_8x8


def compute_attention_score(kb, sm_Q, sm_K, sm_A):
    with kb.reg_scope("attention_score") as scope:
        d0 = scope.alloc("d0")
        d1 = scope.alloc("d1")
        mma_8x16_8x8(kb, sm_Q, sm_K, sm_A, (d0, d1))

        r_a_base = scope.alloc("a_base")
        kb.asm("add", r_a_base, sm_A, kb.sr_laneid)
        kb.asm("sram.st", r_a_base, d0)

        r_32 = scope.alloc("thirtytwo")
        kb.asm("set.imm", r_32, 32.0)
        kb.asm("add", r_a_base, r_a_base, r_32)
        kb.asm("sram.st", r_a_base, d1)


def scale_attention_score(kb, sm_A, scale):
    with kb.reg_scope("scale_attention_score") as scope:
        r_32 = scope.alloc("thirtytwo")
        kb.asm("set.imm", r_32, 32.0)

        r_scale = scope.alloc("scale")
        kb.asm("set.imm", r_scale, scale)

        r_addr = scope.alloc("addr")
        r_val = scope.alloc("val")

        kb.asm("add", r_addr, sm_A, kb.sr_laneid)
        kb.asm("sram.ld", r_val, r_addr)
        kb.asm("mul", r_val, r_val, r_scale)
        kb.asm("sram.st", r_addr, r_val)

        kb.asm("add", r_addr, r_addr, r_32)
        kb.asm("sram.ld", r_val, r_addr)
        kb.asm("mul", r_val, r_val, r_scale)
        kb.asm("sram.st", r_addr, r_val)


def load_8x32_into_sram(kb, r_base, sm_1, sm_2):
    with kb.reg_scope("sram_load_scope") as sram_load_scope:
        #registers containing constants
        r_16 = sram_load_scope.alloc("sixteen")
        r_32 = sram_load_scope.alloc("thirtytwo")
        r_48 = sram_load_scope.alloc("fortyeight")
        kb.asm("set.imm", r_16, 16.0)
        kb.asm("set.imm", r_32, 32.0)
        kb.asm("set.imm", r_48, 48.0)

        p_first = sram_load_scope.alloc_pred("p_first")
        kb.asm("cmp.lt", p_first, kb.sr_laneid, r_16)

        r_row1 = sram_load_scope.alloc("row1")
        r_row2 = sram_load_scope.alloc("row2")

        r_address = sram_load_scope.alloc("address")
        r_SM1_address = sram_load_scope.alloc("K1_address")
        r_SM2_address = sram_load_scope.alloc("K2_address")

        kb.asm("add", r_address, r_base, kb.sr_laneid)
        kb.asm("add", r_SM1_address, sm_1, kb.sr_laneid)

        kb.asm("add", r_SM2_address, sm_2, kb.sr_laneid)
        kb.asm("add", r_SM2_address, r_SM2_address, r_16, pred=p_first)
        kb.asm("sub", r_SM2_address, r_SM2_address, r_16, pred=p_first, pred_inv=True)

        r_tmp = sram_load_scope.alloc("tmp")

        #automated code generation by using for loop, not cheating
        for rowpair in range(4):
            kb.asm("dram.ld", r_row1, r_address)

            kb.asm("add", r_address, r_address, r_48, pred=p_first)
            kb.asm("add", r_address, r_address, r_16, pred=p_first, pred_inv=True)

            kb.asm("dram.ld", r_row2, r_address)

            #register swap for coalesced/conflict-free shared memory write
            kb.asm("mov", r_tmp, r_row2, pred=p_first, pred_inv=True)
            kb.asm("mov", r_row2, r_row1, pred=p_first, pred_inv=True)
            kb.asm("mov", r_row1, r_tmp, pred=p_first, pred_inv=True)

            kb.asm("sram.st", r_SM1_address, r_row1)
            kb.asm("sram.st", r_SM2_address, r_row2)

            #setting up for next iteration
            kb.asm("add", r_address, r_address, r_48, pred=p_first, pred_inv=True)
            kb.asm("add", r_address, r_address, r_16, pred=p_first)

            kb.asm("add", r_SM1_address, r_SM1_address, r_32)
            kb.asm("add", r_SM2_address, r_SM2_address, r_32)


def compute_softmax(kb, sm_A):
    with kb.reg_scope("softmax_scope") as softmax_scope:
        r_8 = softmax_scope.alloc("eight")
        r_1 = softmax_scope.alloc("one")

        kb.asm("set.imm", r_8, 8.0)
        kb.asm("set.imm", r_1, 1.0)

        p_lt_8 = softmax_scope.alloc_pred("p_lt_8")
        kb.asm("cmp.lt", p_lt_8, kb.sr_laneid, r_8)

        r_idx = softmax_scope.alloc("idx")
        r_val = softmax_scope.alloc("val")
        r_tmp = softmax_scope.alloc("tmp")
        r_sum = softmax_scope.alloc("sum")

        kb.asm("fma", r_idx, kb.sr_laneid, r_8, sm_A, pred=p_lt_8)
        kb.asm("set.imm", r_tmp, float('-inf'))
        kb.asm("set.imm", r_sum, 0.0)

        for i in range(8):
            kb.asm("sram.ld", r_val, r_idx, pred=p_lt_8)
            kb.asm("max", r_tmp, r_tmp, r_val, pred=p_lt_8)
            kb.asm("add", r_idx, r_idx, r_1, pred=p_lt_8)

        for i in range(8):
            kb.asm("sub", r_idx, r_idx, r_1, pred=p_lt_8)
            kb.asm("sram.ld", r_val, r_idx, pred=p_lt_8)
            kb.asm("sub", r_val, r_val, r_tmp, pred=p_lt_8)
            kb.asm("exp.trans", r_val, r_val, pred=p_lt_8)
            kb.asm("add", r_sum, r_sum, r_val, pred=p_lt_8)
            kb.asm("sram.st", r_idx, r_val, pred=p_lt_8)

        for i in range(8):
            kb.asm("sram.ld", r_val, r_idx, pred=p_lt_8)
            kb.asm("div", r_val, r_val, r_sum, pred=p_lt_8)
            kb.asm("sram.st", r_idx, r_val, pred=p_lt_8)
            kb.asm("add", r_idx, r_idx, r_1, pred=p_lt_8)


def baseline_attention_kernel(kb, r_base_Q, r_base_K, r_base_A, r_B, r_H):
    sm_Q1 = kb.sram.alloc("Q1", 128 * kb.num_warps)
    sm_Q2 = kb.sram.alloc("Q2", 128 * kb.num_warps)
    sm_K1 = kb.sram.alloc("K1", 128)
    sm_K2 = kb.sram.alloc("K2", 128)
    sm_A = kb.sram.alloc("A", 64 * kb.num_warps)

    #loading up K1 and K2 from dram
    load_8x32_into_sram(kb, r_base_K, sm_K1, sm_K2)

    with kb.reg_scope("compute_scope") as compute_scope:
        #constants
        r_256 = compute_scope.alloc("twofixtysix")
        r_128 = compute_scope.alloc("onetwentyeight")
        r_64 = compute_scope.alloc("sixtyfour")
        r_32 = compute_scope.alloc("thirtytwo")
        r_0 = compute_scope.alloc("zero")
        kb.asm("set.imm", r_256, 256.0)
        kb.asm("set.imm", r_128, 128.0)
        kb.asm("set.imm", r_64, 64.0)
        kb.asm("set.imm", r_32, 32.0)
        kb.asm("set.imm", r_0, 0.0)

        #partitioning shared memory across warps
        r_Q1 = compute_scope.alloc("Q1")
        r_Q2 = compute_scope.alloc("Q2")
        r_A = compute_scope.alloc("A")

        kb.asm("fma", r_Q1, kb.sr_warpid, r_128, sm_Q1)
        kb.asm("fma", r_Q2, kb.sr_warpid, r_128, sm_Q2)
        kb.asm("fma", r_A, kb.sr_warpid, r_64, sm_A)

        r_Q_address = compute_scope.alloc("Q_address")
        r_A_address = compute_scope.alloc("A_address")
        r_Q_end = compute_scope.alloc("Q_end")
        p_continue = compute_scope.alloc_pred("p_continue")

        r_A_t_addr = compute_scope.alloc("A_t_addr")
        r_src1_addr = compute_scope.alloc("src1_addr")
        r_src2_addr = compute_scope.alloc("src2_addr")
        r_tmp = compute_scope.alloc("tmp")

        kb.asm("fma", r_Q_address, kb.sr_warpid, r_256, r_base_Q)

        kb.asm("mul", r_Q_end, r_B, r_H)
        kb.asm("fma", r_Q_end, r_Q_end, r_256, r_base_Q)

        kb.asm("fma", r_A_address, kb.sr_warpid, r_64, r_base_A)
        kb.asm("add", r_src1_addr, r_A, kb.sr_laneid)
        kb.asm("add", r_src2_addr, r_src1_addr, r_32)

        kb.label("work_loop_start")
        kb.asm("cmp.lt", p_continue, r_Q_address, r_Q_end)
        kb.asm("jmp", "work_loop_end", pred=p_continue, pred_inv=True)

        #memset A to 0
        kb.asm("sram.st", r_src1_addr, r_0)
        kb.asm("sram.st", r_src2_addr, r_0)

        load_8x32_into_sram(kb, r_Q_address, r_Q1, r_Q2)
        compute_attention_score(kb, r_Q1, sm_K1, r_A)
        compute_attention_score(kb, r_Q2, sm_K2, r_A)
        scale_attention_score(kb, r_A, 0.17677669)
        compute_softmax(kb, r_A)

        kb.asm("add", r_A_t_addr, r_A_address, kb.sr_laneid)
        kb.asm("sram.ld", r_tmp, r_src1_addr)
        kb.asm("dram.st", r_A_t_addr, r_tmp)

        kb.asm("add", r_A_t_addr, r_A_t_addr, r_32)
        kb.asm("sram.ld", r_tmp, r_src2_addr)
        kb.asm("dram.st", r_A_t_addr, r_tmp)

        kb.asm("fma", r_Q_address, kb.sr_numwarps, r_256, r_Q_address)
        kb.asm("fma", r_A_address, kb.sr_numwarps, r_64, r_A_address)

        kb.asm("jmp", "work_loop_start")
        kb.label("work_loop_end")


def run_attention_kernel(Q, K, num_warps=4, device=None, tracer=True, debug=False):
    if len(Q.shape) != 4 or len(K.shape) != 2 or Q.shape[-1] != K.shape[-1]:
        raise ValueError("Invalid shape for Q or K")

    B, H, Nq, d = Q.shape
    Nk, _ = K.shape

    #following problem shapes exactly for now
    if Nq != 8 or Nk != 8:
        raise ValueError("Expected attention score matrix dimensions to be multiples of 8")
    if d != 32:
        raise ValueError("Expected token dimension to be multiple of 16")

    attention_size = B*H*Nq*Nk

    Q_arr = np.asarray(Q, dtype=np.float32)
    K_arr = np.asarray(K, dtype=np.float32)

    dev = device if device is not None else pGPU(num_warps_per_smsp=num_warps)

    alloc_Q = dev.dram_alloc("Q", Q_arr.size)
    alloc_K = dev.dram_alloc("K", K_arr.size)
    alloc_A = dev.dram_alloc("A", attention_size)

    dev.dram_copy(alloc_Q, Q_arr.ravel(), direction=CopyDirection.H2D)
    dev.dram_copy(alloc_K, K_arr.ravel(), direction=CopyDirection.H2D)

    compiled = build_kernel(baseline_attention_kernel, num_warps=num_warps, debug=debug)

    if tracer:
        tracer = KernelTracer("baseline_attention_trace")
        tracer.attach(dev, compiled)
        total_cycles = dev.launch(compiled, alloc_Q, alloc_K, alloc_A, float(B), float(H))
        tracer.export()
    else:
        total_cycles = dev.launch(compiled, alloc_Q, alloc_K, alloc_A, float(B), float(H))

    score = dev.dram_copy(alloc_A, direction=CopyDirection.D2H).reshape(B, H, Nq, Nk)
    return score, total_cycles

