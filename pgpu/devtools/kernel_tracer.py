import json
from collections import defaultdict
from typing import Dict, List, Any, Optional

from pgpu.arch.gpu_device import pGPU
from pgpu.sw.kernel import CompiledKernel
from pgpu.arch.isa import Instruction, OpClass, OpCode

class KernelTracer:
    """
    Automated tracer that hooks into a pGPU device and generates Perfetto-compatible
    trace events and a standalone HTML ASM viewer.
    """
    def __init__(self, output_base: str = "trace", target_warp: int = 0):
        self.output_base = output_base
        self.target_warp = target_warp
        self.events: List[dict] = []
        self.trace_records: List[dict] = []
        self._active_regions: Dict[int, set] = defaultdict(set)
        self._trace_regions: Dict[str, tuple] = {}

        #(warp_id, res_name) -> this is a list of completion times
        self._res_slots: Dict[tuple, list] = {}

    def attach(self, dev: pGPU, kernel: CompiledKernel):
        """
        Attach the tracer to a pGPU device and extract trace_regions from the kernel.
        """
        self._trace_regions = getattr(kernel, "trace_regions", {})
        
        # Inject hook to all SMSPs
        for sm in dev.sms:
            for smsp in sm.smsps:
                smsp.trace_hook = self._hook

    def _allocate_pipe_tid(self, pid: int, base_tid: str, issue_cycle: int, completion_time: int) -> str:
        """
        Finding a sub pipe slot to prevent overlapping slices in perfetto
        """
        key = (pid, base_tid)
        slots = self._res_slots.setdefault(key, [])
        slot_idx = -1
        for idx, free_at in enumerate(slots):
            if free_at <= issue_cycle:
                slot_idx = idx
                slots[idx] = completion_time
                break
        if slot_idx == -1:
            slot_idx = len(slots)
            slots.append(completion_time)
        return base_tid if slot_idx == 0 else f"{base_tid} [Pipe {slot_idx}]"

    def _hook(self, warp_id: int, pc: int, clone: Instruction, issue_cycle: int):
        #is context switch?
        is_cs = (clone.op_class == OpClass.SRAM and not clone.srcs) or (pc == -1)

        #driver/context switch events go to a Driver process(with pid 100)
        if is_cs:
            pid = 100
            op_name = "sram.st" if clone.opcode == OpCode.SRAM_ST else "sram.ld"
            action = "Save" if clone.opcode == OpCode.SRAM_ST else "Restore"
            reg_id = clone.dest[0] if clone.dest else "?"
            base_tid = f"Warp {warp_id} ({action})"
            dur = max(clone.completion_time - issue_cycle, 1)
            tid_str = self._allocate_pipe_tid(pid, base_tid, issue_cycle, clone.completion_time)

            self.events.append({
                "name": f"{op_name} R{reg_id}",
                "cat": "driver_context_switch",
                "ph": "X",
                "pid": pid,
                "tid": tid_str,
                "ts": issue_cycle,
                "dur": dur,
                "args": {"warp": warp_id, "reg": reg_id}
            })
            return

        #normal kernel instructions
        pid = warp_id

        # 1. Update active regions (only if PC is valid for regions)
        for region_name, (start_pc, end_pc) in self._trace_regions.items():
            is_inside = (start_pc <= pc <= end_pc)
            was_inside = region_name in self._active_regions[warp_id]

            if is_inside and not was_inside:
                # Entered region
                self._active_regions[warp_id].add(region_name)
                self.events.append({
                    "name": region_name,
                    "cat": "warp_execution",
                    "ph": "B",
                    "pid": warp_id,
                    "tid": "Execution",
                    "ts": issue_cycle,
                })
            elif not is_inside and was_inside:
                # Exited region
                self._active_regions[warp_id].remove(region_name)
                self.events.append({
                    "name": region_name,
                    "cat": "warp_execution",
                    "ph": "E",
                    "pid": warp_id,
                    "tid": "Execution",
                    "ts": issue_cycle,
                })

        # 2. Log resource usage for each instruction within active regions
        for region_name in self._active_regions[warp_id]:
            dur = max(clone.completion_time - issue_cycle, 1)
            op_name = clone.opcode.name.lower().replace("_", ".")
            res_name = clone.op_class.name

            tid_str = self._allocate_pipe_tid(pid, res_name, issue_cycle, clone.completion_time)

            self.events.append({
                "name": op_name,
                "cat": f"resource_{res_name}",
                "ph": "X",
                "pid": warp_id,
                "tid": tid_str,
                "ts": issue_cycle,
                "dur": dur,
                "args": {"label": region_name, "pc": pc}
            })

        # 3. Collect HTML trace viewer records for the target warp
        if warp_id == self.target_warp:
            self.trace_records.append({
                "opcode":           clone.opcode.value,
                "op_class":         clone.op_class.value,
                "dest":             list(clone.dest),
                "srcs":             list(clone.srcs),
                "pred":             clone.pred,
                "pred_inv":         clone.pred_inv,
                "imm":              clone.imm if not isinstance(clone.imm, str) else None,
                "issue_cycle":      issue_cycle,
                "completion_cycle": clone.completion_time,
            })

    def export(self):
        """Export the collected trace events to Perfetto JSON and HTML viewer."""
        # Close any active regions at the end
        if self.events:
            last_ts = max((e.get("ts", 0) + e.get("dur", 0)) for e in self.events)
        else:
            last_ts = 0

        for warp_id, regions in self._active_regions.items():
            for region_name in list(regions):
                self.events.append({
                    "name": region_name,
                    "cat": "warp_execution",
                    "ph": "E",
                    "pid": warp_id,
                        "tid": "Execution",
                    "ts": last_ts,
                })
        

        #processing metadata labels
        #pids 0..3 -> warp ids
        #pid 100 -> driver context switch
        all_pids = set(self._active_regions.keys()).union(set(e.get("pid") for e in self.events if "pid" in e))
        for p in all_pids:
            name = "Driver context switch" if p == 100 else "Warp"
            self.events.insert(0, {
                "name": "process_name",
                "ph": "M",
                "pid": p,
                "args": {"name": name}
            })

        json_path = f"{self.output_base}.json"
        with open(json_path, "w") as f:
            json.dump({"traceEvents": self.events, "displayTimeUnit": "ns"}, f, indent=2)
        print(f"Kernel trace exported to {json_path} (Perfetto format)")

        html_path = f"{self.output_base}.html"
        from pgpu.devtools.objdump import export_html
        export_html(self.trace_records, warp_id=self.target_warp, output_path=html_path)
