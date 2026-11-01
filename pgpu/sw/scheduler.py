from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from pgpu.arch.isa import Instruction, OpCode, OpClass
from pgpu.arch.smsp import RZ_ID, SR_LANEID_ID, SR_WARPID_ID, SR_NUMWARPS_ID, PRED_BASE_ID, NUM_PRED_REGS

#registers that don't create data hazards
IGNORE_REG_IDS = {
    RZ_ID,
    SR_LANEID_ID,
    SR_WARPID_ID,
    SR_NUMWARPS_ID
}

#opcode latencies(in cycles)
#for modelling the DAG, to calculate the cycle when the result can be consumed by a dependent instruction
DEFAULT_OPCODE_LATENCY = {
    #VALU single cycle
    OpCode.ADD: 1,
    OpCode.SUB: 1,
    OpCode.MUL: 1,
    OpCode.FMA: 1,
    OpCode.MIN: 1,
    OpCode.MAX: 1,
    OpCode.SHL: 1,
    OpCode.SHR: 1,
    OpCode.AND: 1,
    OpCode.OR: 1,
    OpCode.XOR: 1,
    OpCode.NOT: 1,
    OpCode.CMP_LT: 1,
    OpCode.CMP_EQ: 1,
    OpCode.SELECT: 1,
    OpCode.MOV: 1,
    OpCode.BCAST: 1,
    OpCode.SET_IMM: 1,
    OpCode.JMP: 1,

    #VALU multi cycle
    OpCode.DIV: 4,

    #GEMMCORE
    OpCode.AVX_MMA: 8,
    OpCode.AVX_MMRED_MAX: 8,
    OpCode.AVX_MMRED_MIN: 8,

    #TRANS(need to look into accurate latencies for the fast versions)
    OpCode.EXP_TRANS: 24,
    OpCode.SIN_TRANS: 24,
    OpCode.COS_TRANS: 24,
    OpCode.TANH_TRANS: 24,
    OpCode.LOG_TRANS: 24,

    #assuming n = 64
    OpCode.AVX_EXP_FAST: 13,
    OpCode.AVX_SIN_FAST: 13,
    OpCode.AVX_COS_FAST: 13,
    OpCode.AVX_LOG_FAST: 13,
    OpCode.AVX_TANH_FAST: 24,

    #MEMORY
    #to be decided but some defaults for now
    OpCode.SRAM_LD: 14,
    OpCode.SRAM_ST: 14,
    OpCode.DRAM_LD: 90,
    OpCode.DRAM_ST: 90,

    #Sync
    #to be decided but some defaults for now
    OpCode.SYNC: 1,
    OpCode.SYNC_WARP: 1,
    OpCode.YIELD: 1
}


@dataclass(eq=False)
class Block:
    block_id: int
    terminators: List[Instruction] = field(default_factory=list)
    instructions: List[Instruction] = field(default_factory=list)
    label_names: List[str] = field(default_factory=list)
    new_start_pc: int = 0


def split_into_blocks(instructions, labels):
    if not instructions:
        return []

    pc_to_labels = {}
    for name, target_pc in labels.items():
        pc_to_labels.setdefault(target_pc, []).append(name)

    #terminators for a block
    block_terminators = {
        OpCode.JMP,
        OpCode.SYNC,
        OpCode.SYNC_WARP,
        OpCode.YIELD,
    }

    #leader PCs
    is_leader = [False] * (len(instructions) + 1)
    is_terminator = [False] * (len(instructions) + 1)
    is_leader[0] = True

    for instr in instructions:
        if instr.opcode == OpCode.JMP:
            #will fix label forwarding in the main kernel later
            #for now, label resolution happens before scheduling
            if not isinstance(instr.imm, int):
                raise ValueError("unresolved jmp operand")

            if 0 <= instr.imm <= len(instructions):
                is_leader[instr.imm] = True

    for pc in pc_to_labels:
        if 0 <= pc <= len(instructions):
            is_leader[pc] = True

    for pc, instr in enumerate(instructions):
        if instr.opcode in block_terminators:
            is_terminator[pc] = True
            is_leader[pc + 1] = True

    blocks = []
    current_block = None

    for pc, instr in enumerate(instructions):
        if is_leader[pc]:
            current_block = Block(
                block_id=len(blocks),
                label_names=pc_to_labels.get(pc, [])
            )
            blocks.append(current_block)

        if is_terminator[pc]:
            current_block.terminators.append(instr)
        else:
            current_block.instructions.append(instr)

    end_pc = len(instructions)
    if end_pc in pc_to_labels:
        empty_trailing_block = Block(
            block_id=len(blocks),
            label_names=pc_to_labels[end_pc]
        )
        blocks.append(empty_trailing_block)

    return blocks


@dataclass(eq=False)
class DAGNode:
    node_id: int
    instr: Instruction
    latency: int = 1
    preds: List[Tuple["DAGNode", int, str]] = field(default_factory=list)
    succs: List[Tuple["DAGNode", int, str]] = field(default_factory=list)
    height: int = 0


def build_dependency_dag(block, latencies=DEFAULT_OPCODE_LATENCY):
    nodes = [
        DAGNode(
            node_id=idx,
            instr=instr,
            latency=latencies[instr.opcode]
        )

        for idx, instr in enumerate(block.instructions)
    ]

    last_def = {}
    last_uses = {}

    last_sram_store = None
    last_sram_loads = []
    last_dram_store = None
    last_dram_loads = []

    def add_edge(src, dst, latency, dep_type):
        if src.node_id == dst.node_id:
            return

        for i, (succ, lat, dtype) in enumerate(src.succs):
            if succ.node_id == dst.node_id:
                if latency > lat:
                    src.succs[i] = (succ, latency, dep_type)

                    for j, (p, p_lat, p_dtype) in enumerate(dst.preds):
                        if p.node_id == src.node_id:
                            dst.preds[j] = (p, latency, dep_type)
                            break
                return

        src.succs.append((dst, latency, dep_type))
        dst.preds.append((src, latency, dep_type))

    for node in nodes:
        instr = node.instr

        #source registers read for an Instruction
        src_regs = set(instr.srcs) - IGNORE_REG_IDS
        if instr.pred is not None:
            src_regs.add(PRED_BASE_ID + instr.pred)

        #destination registers written to for an Instruction
        dest_regs = set(instr.dest) - IGNORE_REG_IDS

        #RAW
        for r in src_regs:
            if r in last_def:
                producer = last_def[r]
                add_edge(producer, node, producer.latency, "RAW")

        #WAR(antidependency, latency=1?)
        for r in dest_regs:
            if r in last_uses:
                for consumer in last_uses[r]:
                    add_edge(consumer, node, 1, "WAR")

        #WAW(latency=1?)
        for r in dest_regs:
            if r in last_def:
                prev_producer = last_def[r]
                add_edge(prev_producer, node, prev_producer.latency, "WAW")

        #SRAM hazards
        if instr.opcode == OpCode.SRAM_LD:
            if last_sram_store is not None:
                add_edge(last_sram_store, node, last_sram_store.latency, "MEM_RAW")
            last_sram_loads.append(node)
        elif instr.opcode == OpCode.SRAM_ST:
            if last_sram_store is not None:
                add_edge(last_sram_store, node, 1, "MEM_WAW")

            for ld in last_sram_loads:
                add_edge(ld, node, 1, "MEM_WAR")

            last_sram_store = node
            last_sram_loads = []

        #DRAM hazards
        if instr.opcode == OpCode.DRAM_LD:
            if last_dram_store is not None:
                add_edge(last_dram_store, node, last_dram_store.latency, "MEM_RAW")
            last_dram_loads.append(node)
        elif instr.opcode == OpCode.DRAM_ST:
            if last_dram_store is not None:
                add_edge(last_dram_store, node, 1, "MEM_WAW")

            for ld in last_dram_loads:
                add_edge(ld, node, 1, "MEM_WAR")

            last_dram_store = node
            last_dram_loads = []

        #update definitions and usages
        for r in dest_regs:
            last_def[r] = node
            last_uses[r] = []

        for r in src_regs:
            last_uses.setdefault(r, []).append(node)

    for n in reversed(nodes):
        #might change to only look for "RAW" succs
        #the others prevent illegal reordering, not part of latency hiding
        n.height = max((s.height + lat for s, lat, _ in n.succs), default=0)

    return nodes


def schedule_block(block, latencies=DEFAULT_OPCODE_LATENCY):
    if len(block.instructions) <= 1:
        return block.instructions + block.terminators

    terminators = block.terminators
    nodes = build_dependency_dag(block, latencies)

    in_degrees = {n: len(n.preds) for n in nodes}
    earliest_issue_cycle = {n: 0 for n in nodes}

    ready_list = [n for n in nodes if in_degrees[n] == 0]
    scheduled_nodes = []
    current_cycle = 0

    while len(scheduled_nodes) < len(nodes):
        if not ready_list:
            raise RuntimeError("DAG DEADLOCK!")

        available = [n for n in ready_list if earliest_issue_cycle[n] <= current_cycle]
        if not available:
            current_cycle = min(earliest_issue_cycle[n] for n in ready_list)
            available = [n for n in ready_list if earliest_issue_cycle[n] <= current_cycle]

        candidate = max(available, key=lambda n:(n.height, -n.node_id))
        ready_list.remove(candidate)
        scheduled_nodes.append(candidate)

        issue_time = current_cycle
        current_cycle += 1

        for succ, lat, _ in candidate.succs:
            earliest_issue_cycle[succ] = max(earliest_issue_cycle[succ], issue_time + lat)
            in_degrees[succ] -= 1
            if in_degrees[succ] == 0:
                ready_list.append(succ)

    scheduled_instr = [n.instr for n in scheduled_nodes]
    if terminators:
        scheduled_instr.extend(terminators)

    return scheduled_instr


def schedule_kernel(instructions, labels, trace_regions, latencies=DEFAULT_OPCODE_LATENCY):
    if not instructions:
        return [], dict(labels), dict(trace_regions or {})

    #recording original instructions for trace region bounds
    trace_region_instrs = {}
    if trace_regions:
        for name, (start_pc, end_pc) in trace_regions.items():
            if 0 <= start_pc < len(instructions) and 0 <= end_pc < len(instructions):
                trace_region_instrs[name] = (instructions[start_pc], instructions[end_pc])

    #block splitting
    blocks = split_into_blocks(instructions, labels)

    #scheduling blocks
    scheduled_instructions = []
    new_labels = {}

    current_pc = 0
    for block in blocks:
        block.new_start_pc = current_pc
        for lbl in block.label_names:
            new_labels[lbl] = block.new_start_pc

        sched_block = schedule_block(block, latencies)
        scheduled_instructions.extend(sched_block)
        current_pc += len(sched_block)

    #updating trace regions
    new_trace_regions = {}
    if trace_regions:
        instr_to_new_pc = {id(inst) : idx for idx, inst in enumerate(scheduled_instructions)}

        for name, (start_inst, end_inst) in trace_region_instrs.items():
            s_pc = instr_to_new_pc.get(id(start_inst), trace_regions[name][0])
            e_pc = instr_to_new_pc.get(id(end_inst), trace_regions[name][1])
            new_trace_regions[name] = (min(s_pc, e_pc), max(s_pc, e_pc))

    return scheduled_instructions, new_labels, new_trace_regions
