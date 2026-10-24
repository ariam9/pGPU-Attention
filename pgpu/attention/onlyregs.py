import numpy as np

from pgpu.arch.gpu_device import pGPU, CopyDirection
from pgpu.sw.kernel import build_kernel
from pgpu.devtools.kernel_tracer import KernelTracer

def onlyregs_attention_kernel(kb, r_base_Q, r_base_K, r_base_A, r_B, r_H):
    #64 words of SRAM per warp to compute softmax reductions
    sm_scratch = kb.sram.alloc("scratch", 64 * kb.num_warps)

    with kb.reg_scope("main_scope") as scope:
        #scalar constants
        r_scale = scope.alloc("scale")
        r_1 = scope.alloc("one")
        r_neg_inf = scope.alloc("neg_inf")
        r_256 = scope.alloc("twofixtysix")
        r_64 = scope.alloc("sixtyfour")
        r_32 = scope.alloc("thirtytwo")
        r_8 = scope.alloc("eight")
        r_4 = scope.alloc("four")
        r_2 = scope.alloc("two")


        kb.asm("set.imm", r_scale, float(1.0 / np.sqrt(32)))
        kb.asm("set.imm", r_1, 1.0)
        kb.asm("set.imm", r_neg_inf, float('-inf'))
        kb.asm("set.imm", r_256, 256.0)
        kb.asm("set.imm", r_64, 64.0)
        kb.asm("set.imm", r_32, 32.0)
        kb.asm("set.imm", r_8, 8.0)
        kb.asm("set.imm", r_4, 4.0)
        kb.asm("set.imm", r_2, 2.0)


        #scratch registers
        r_tmp1 = scope.alloc("tmp1")
        r_tmp2 = scope.alloc("tmp2")
        r_tmp3 = scope.alloc("tmp3")
        p_tmp = scope.alloc_pred("p_tmp")


        #calculating offset of a thread/lane for loading from dram
        r_quad_off = scope.alloc("quad_off")
        r_row = scope.alloc("row")
        r_col = scope.alloc("col")


        #storing row in tmp1 and col in tmp2 to calculate offset, tmp3 is just tmp
        kb.asm("set.imm", r_row, 0.0)
        for thresh in [8.0, 16.0, 24.0]:
            kb.asm("set.imm", r_tmp3, thresh)
            kb.asm("cmp.lt", p_tmp, kb.sr_laneid, r_tmp3)
            kb.asm("add", r_row, r_row, r_1, pred=p_tmp, pred_inv=True)

        kb.asm("mul", r_tmp3, r_row, r_8)
        kb.asm("sub", r_col, kb.sr_laneid, r_tmp3)
        #quad_off = 32 * row + col -> this is offset for loading 8x32 matrices into registers
        kb.asm("fma", r_quad_off, r_row, r_32, r_col)


        #sram base addresses for this warp
        r_sm_base = scope.alloc("sm_base")
        kb.asm("fma", r_sm_base, kb.sr_warpid, r_64, sm_scratch)


        #registers for holding K, Q and A
        k_regs = [scope.alloc(f"k_{i}") for i in range(8)]
        q_regs = [scope.alloc(f"q_{i}") for i in range(8)]
        #mma result registers
        d_regs = [scope.alloc("d0"), scope.alloc("d1")]


        #dram addresses and loop control
        r_b = scope.alloc("b_idx")
        r_K_addr = scope.alloc("K_addr")
        r_Q_addr = scope.alloc("Q_addr")
        r_Q_batch_end = scope.alloc("Q_batch_end")
        r_A_addr = scope.alloc("A_addr")
        r_bh_offset = scope.alloc("bh_offset")


        #softmax predicates
        p_lt_4 = scope.alloc_pred("p_lt_4")
        p_lt_2 = scope.alloc_pred("p_lt_2")
        p_lt_1 = scope.alloc_pred("p_lt_1")

        kb.asm("cmp.lt", p_lt_4, r_col, r_4)
        kb.asm("cmp.lt", p_lt_2, r_col, r_2)
        kb.asm("cmp.lt", p_lt_1, r_col, r_1)


        #initialising batch with warp_id
        kb.asm("mov", r_b, kb.sr_warpid)


        #batch loop
        kb.label("batch_loop")
        kb.asm("cmp.lt", p_tmp, r_b, r_B)
        kb.asm("jmp", "kernel_end", pred=p_tmp, pred_inv=True)


        #loading K once for the entire batch
        kb.asm("fma", r_K_addr, r_b, r_256, r_base_K)
        kb.asm("add", r_K_addr, r_K_addr, r_quad_off)


        #processing left side 8x16 loads, then right side 8x16 loads
        #need to experiment with overlapping that's happening with later mma instructions
        offsets = [0.0, 128.0, 8.0, 136.0, 16.0, 144.0, 24.0, 152.0]
        for i, off in enumerate(offsets):
            if off == 0.0:
                kb.asm("dram.ld", k_regs[i], r_K_addr)
            else:
                kb.asm("set.imm", r_tmp1, off)
                kb.asm("add", r_tmp2, r_K_addr, r_tmp1)
                kb.asm("dram.ld", k_regs[i], r_tmp2)


        #calculating addresses for initialising head loop
        kb.asm("mul", r_bh_offset, r_b, r_H)
        kb.asm("fma", r_Q_addr, r_bh_offset, r_256, r_base_Q)
        kb.asm("fma", r_Q_batch_end, r_H, r_256, r_Q_addr)
        kb.asm("fma", r_A_addr, r_bh_offset, r_64, r_base_A)


        #head loop
        kb.label("head_loop")
        kb.asm("cmp.lt", p_tmp, r_Q_addr, r_Q_batch_end)
        kb.asm("jmp", "next_batch", pred=p_tmp, pred_inv=True)


        #loading Q for this head into registers(similar to loading K)
        offsets = [0.0, 128.0, 8.0, 136.0, 16.0, 144.0, 24.0, 152.0]
        kb.asm("add", r_tmp3, r_Q_addr, r_quad_off)
        for i, off in enumerate(offsets):
            if off == 0.0:
                kb.asm("dram.ld", q_regs[i], r_tmp3)
            else:
                kb.asm("set.imm", r_tmp1, off)
                kb.asm("add", r_tmp2, r_tmp3, r_tmp1)
                kb.asm("dram.ld", q_regs[i], r_tmp2)


        #GEMM instructions -> Q @ K.T
        #kb.rz -> default zero register
        kb.asm("avx.mma", d_regs, q_regs[0:4], k_regs[0:4], (kb.rz, kb.rz))
        kb.asm("avx.mma", d_regs, q_regs[4:8], k_regs[4:8], d_regs)

        #scaling(dividing by root d)
        kb.asm("mul", d_regs[0], d_regs[0], r_scale)
        kb.asm("mul", d_regs[1], d_regs[1], r_scale)


        #softmax
        #3 step reduction divided among groups of 8 threads to avoid bank conflicts
        kb.asm("add", r_tmp1, r_sm_base, kb.sr_laneid)
        kb.asm("sram.st", r_tmp1, d_regs[0])
        kb.asm("add", r_tmp2, r_tmp1, r_32)
        kb.asm("sram.st", r_tmp2, d_regs[1])

        r_offsets = {4: r_4, 2: r_2, 1: r_1}
        p_offsets = {4: p_lt_4, 2: p_lt_2, 1: p_lt_1}

        for idx in range(2):
            #max reduction
            for offset in [4, 2, 1]:
                reg_off = r_offsets[offset]
                pred_off = p_offsets[offset]
                kb.asm("sram.ld", r_tmp2, r_tmp1, pred=pred_off)
                kb.asm("add", r_tmp3, r_tmp1, reg_off, pred=pred_off)
                kb.asm("sram.ld", r_tmp3, r_tmp3, pred=pred_off)
                kb.asm("max", r_tmp3, r_tmp2, r_tmp3, pred=pred_off)
                kb.asm("sram.st", r_tmp1, r_tmp3, pred=pred_off)

            kb.asm("sub", r_tmp2, r_tmp1, r_col)
            kb.asm("sram.ld", r_tmp3, r_tmp2)

            #tmp3 now contains the max of the row
            kb.asm("sub", d_regs[idx], d_regs[idx], r_tmp3)
            kb.asm("exp.trans", d_regs[idx], d_regs[idx])
            kb.asm("sram.st", r_tmp1, d_regs[idx])

            #sum reduction
            for offset in [4, 2, 1]:
                reg_off = r_offsets[offset]
                pred_off = p_offsets[offset]
                kb.asm("sram.ld", r_tmp2, r_tmp1, pred=pred_off)
                kb.asm("add", r_tmp3, r_tmp1, reg_off, pred=pred_off)
                kb.asm("sram.ld", r_tmp3, r_tmp3, pred=pred_off)
                kb.asm("add", r_tmp3, r_tmp2, r_tmp3, pred=pred_off)
                kb.asm("sram.st", r_tmp1, r_tmp3, pred=pred_off)

            kb.asm("sub", r_tmp2, r_tmp1, r_col)
            kb.asm("sram.ld", r_tmp3, r_tmp2)

            #tmp3 now contains the sum of the row
            kb.asm("div", d_regs[idx], d_regs[idx], r_tmp3)

            #adding to tmp1 to contain address for next iteration
            kb.asm("add", r_tmp1, r_tmp1, r_32)


        #writing attention scores to dram(from d0 and d1 registers)
        kb.asm("add", r_tmp1, r_A_addr, kb.sr_laneid)
        kb.asm("dram.st", r_tmp1, d_regs[0])
        kb.asm("add", r_tmp1, r_tmp1, r_32)
        kb.asm("dram.st", r_tmp1, d_regs[1])


        #going to next head
        kb.asm("add", r_Q_addr, r_Q_addr, r_256)
        kb.asm("add", r_A_addr, r_A_addr, r_64)
        kb.asm("jmp", "head_loop")


        #going to next batch
        kb.label("next_batch")
        kb.asm("add", r_b, r_b, r_1)
        kb.asm("jmp", "batch_loop")


        kb.label("kernel_end")


def run_attention_kernel(Q, K, num_warps=4, device=None, debug=False, tracing=False):
    if len(Q.shape) != 4 or len(K.shape) != 3 or Q.shape[-1] != K.shape[-1] or Q.shape[0] != K.shape[0]:
        raise ValueError("Invalid shape for Q or K")

    B, H, Nq, d = Q.shape
    B, Nk, _ = K.shape

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

    compiled = build_kernel(onlyregs_attention_kernel, num_warps=num_warps, debug=debug)

    if tracing:
        tracer = KernelTracer("baseline_attention_trace")
        tracer.attach(dev, compiled)
        total_cycles = dev.launch(compiled, alloc_Q, alloc_K, alloc_A, float(B), float(H))
        tracer.export()
    else:
        total_cycles = dev.launch(compiled, alloc_Q, alloc_K, alloc_A, float(B), float(H))

    score = dev.dram_copy(alloc_A, direction=CopyDirection.D2H).reshape(B, H, Nq, Nk)
    return score, total_cycles

