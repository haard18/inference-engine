#!/usr/bin/env python3
"""Sample resident memory for one or more server process trees."""

from __future__ import annotations

import argparse
import json
import subprocess
import time


def processes() -> dict[int, tuple[int, int]]:
    output = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,rss="],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    rows: dict[int, tuple[int, int]] = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) != 3:
            raise ValueError("ps returned a row without PID, parent PID, and RSS")
        pid, parent, rss_kib = map(int, fields)
        if pid <= 0 or parent < 0 or rss_kib < 0:
            raise ValueError("ps returned an invalid process value")
        rows[pid] = (parent, rss_kib)
    return rows


def tree_rss(root: int, rows: dict[int, tuple[int, int]]) -> int | None:
    if root not in rows:
        return None
    children: dict[int, list[int]] = {}
    for pid, (parent, _) in rows.items():
        children.setdefault(parent, []).append(pid)
    total = 0
    pending = [root]
    while pending:
        pid = pending.pop()
        total += rows[pid][1]
        pending.extend(children.get(pid, []))
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--interval-ms", type=float, default=100)
    parser.add_argument("--root", action="append", required=True, metavar="NAME=PID")
    args = parser.parse_args()
    if not 0 < args.seconds <= 3600 or not 10 <= args.interval_ms <= 10_000:
        parser.error("seconds must be 0..3600 and interval-ms must be 10..10000")
    roots: dict[str, int] = {}
    for item in args.root:
        name, separator, pid_text = item.partition("=")
        if not separator or not name or not pid_text.isdecimal() or int(pid_text) <= 0:
            parser.error("each root must be NAME=positive-PID")
        if name in roots:
            parser.error("root names must be unique")
        roots[name] = int(pid_text)
    peaks = dict.fromkeys(roots, 0)
    seen = set()
    samples = 0
    started = time.monotonic()
    while True:
        rows = processes()
        for name, root in roots.items():
            rss = tree_rss(root, rows)
            if rss is not None:
                seen.add(name)
                peaks[name] = max(peaks[name], rss)
        samples += 1
        remaining = args.seconds - (time.monotonic() - started)
        if remaining <= 0:
            break
        time.sleep(min(args.interval_ms / 1000, remaining))
    if seen != set(roots):
        parser.error(f"root processes not seen: {', '.join(sorted(set(roots) - seen))}")
    print(
        json.dumps(
            {
                "peak_tree_rss_kib": peaks,
                "samples": samples,
                "elapsed_seconds": time.monotonic() - started,
                "method": "max sampled ps RSS for each root and its descendants",
            }
        )
    )


if __name__ == "__main__":
    main()
