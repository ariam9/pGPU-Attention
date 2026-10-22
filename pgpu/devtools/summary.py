import json
import argparse
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="summary - summarise a kernel trace",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("file_path", nargs="?", help="Path to the trace file")
    args = parser.parse_args()

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

    print(f'\n{'PROCESS / PID':<20} | {'TOTAL CYCLES':>14} | {'% TIME':>8}')
    print('-' * 48)
    for pid, dur in sorted(by_proc.items(), key=lambda x: x[1], reverse=True):
        name = f'Driver (PID {pid})' if pid == 100 else f'Warp {pid}'
        print(f'{name:<20} | {dur:>14,d} | {dur/total_dur*100:>7.2f}%')

    print(f'\n{'CATEGORY':<25} | {'TOTAL CYCLES':>14} | {'% TIME':>8}')
    print('-' * 53)
    for cat, dur in sorted(by_cat.items(), key=lambda x: x[1], reverse=True):
        print(f'{cat:<25} | {dur:>14,d} | {dur/total_dur*100:>7.2f}%')
