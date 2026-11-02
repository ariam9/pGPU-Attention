import json
import argparse
from collections import defaultdict, Counter

def summarise_trace(file_path, top_bubbles=8):
    with open(args.file_path) as f:
        events = json.load(f)['traceEvents']

    by_proc = {}
    by_cat = {}
    total_dur = 0

    for e in events:
        if 'dur' in e:
            dur = e['dur']
            total_dur += dur
            pid = e.get('pid', 'unknown')
            cat = e.get('cat', 'other')
            by_proc[pid] = by_proc.get(pid, 0) + dur
            by_cat[cat] = by_cat.get(cat, 0) + dur

    instr_events = [
        e for e in events
        if e.get("ph") == "X" and e.get("cat", "").startswith("resource_")
    ]
    instr_events.sort(key=lambda x: x["ts"])

    total_instrs = len(instr_events)
    makespan = 0
    total_cycles = 0
    ipc = 0.0
    concurrency = 0.0
    total_bubble_cycles = 0
    bubble_pct = 0.0
    gaps = []
    gap_culprits = defaultdict(Counter)

    if instr_events:
        start_cyc = instr_events[0]["ts"]
        end_cyc = max(e["ts"] + e.get("dur", 1) for e in instr_events)
        makespan = max(end_cyc - start_cyc, 1)
        total_cycles = end_cyc
        ipc = total_instrs / makespan
        concurrency = total_dur / makespan

        for i in range(1, len(instr_events)):
            g = instr_events[i]["ts"] - instr_events[i-1]["ts"]
            gaps.append(g)
            if g > 1:
                prev_op = instr_events[i-1]["name"]
                curr_op = instr_events[i]["name"]
                gap_culprits[g][(prev_op, curr_op)] += 1

        total_bubble_cycles = sum(g - 1 for g in gaps if g > 1)
        bubble_pct = (total_bubble_cycles / makespan) * 100

    #pipeline overview
    print("\n" + "=" * 65)
    print(" " * 20 + "KERNEL PIPELINE OVERVIEW")
    print("=" * 65)
    print(f"  Wall-Clock Makespan       : {makespan:>14,d} cycles")
    print(f"  Total Cycles              : {total_cycles:>14,d} cycles")
    print(f"  Total Instructions Issued : {total_instrs:>14,d}")
    print(f"  Average IPC               : {ipc:>14.4f} instrs / cycle")
    print(f"  Cumulative Hardware Work  : {total_dur:>14,d} resource-cycles")
    print(f"  Hardware Concurrency      : {concurrency:>14.2f}x parallel units")
    print(f"  Total Idle Bubble Cycles  : {total_bubble_cycles:>14,d} ({bubble_pct:.2f}% of makespan)")

    #process breakdown
    print(f"\n{'PROCESS / PID':<22} | {'CUMULATIVE CYCLES':>18} | {'%WORK':>8}")
    print("-" * 54)
    for pid, dur in sorted(by_proc.items(), key=lambda x: x[1],
    reverse=True):
        name = f"Driver (PID {pid})" if pid == 100 else f"Warp {pid}"
        pct = (dur / total_dur * 100) if total_dur > 0 else 0
        print(f"{name:<22} | {dur:>18,d} | {pct:>7.2f}%")

    #resource breakdown
    print(f"\n{'CATEGORY / RESOURCE':<25} | {'CUMULATIVE CYCLES':>18} | {'%WORK':>8}")
    print("-" * 57)
    for cat, dur in sorted(by_cat.items(), key=lambda x: x[1], reverse=True):
        pct = (dur / total_dur * 100) if total_dur > 0 else 0
        print(f"{cat:<25} | {dur:>18,d} | {pct:>7.2f}%")

    #pipeline bubbles
    if gaps:
        print(f"\n{'PIPELINE BUBBLES':<45}")
        print(f"{'GAP (CYCLES)':<14} | {'COUNT':>10} | {'TOTAL DEAD CYCLES':>18} | {'PRIMARY CULPRIT':<25}")
        print("-" * 75)

        counter = Counter(gaps)

        if top_bubbles == -1:
            top_bubbles = len(counter)

        for gap, count in counter.most_common(top_bubbles):
            if gap > 1:
                dead_cyc = (gap - 1) * count
                top_culprit = gap_culprits[gap].most_common(1)[0][0]
                culprit_str = f"{top_culprit[0]} -> {top_culprit[1]}"
                print(f"{gap:<14} | {count:>10,d} | {dead_cyc:>18,d} | {culprit_str:<25}")
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="summary - summarise a kernel trace",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("file_path", nargs="?", help="Path to the trace file")
    parser.add_argument(
        "--bubbles",
        type=int,
        default=8,
        help="number of top pipeline bubble gaps to display(default: 8)",
    )
    args = parser.parse_args()

    summarise_trace(args.file_path, top_bubbles=args.bubbles)
