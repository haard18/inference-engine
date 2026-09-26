#!/usr/bin/env python3
"""Measure repeated local model-worker crash recovery with a real GGUF model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def process_rows() -> dict[int, tuple[int, int]]:
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
            raise RuntimeError("ps returned an incomplete process row")
        pid, parent, rss_kib = map(int, fields)
        if pid <= 0 or parent < 0 or rss_kib < 0:
            raise RuntimeError("ps returned an invalid process row")
        rows[pid] = (parent, rss_kib)
    return rows


def direct_worker(server_pid: int, rows: dict[int, tuple[int, int]]) -> int | None:
    children = [pid for pid, (parent, _) in rows.items() if parent == server_pid]
    if len(children) > 1:
        raise RuntimeError("server has more than one direct child; refusing to kill a process")
    return children[0] if children else None


def tree_rss_kib(server_pid: int, rows: dict[int, tuple[int, int]]) -> int:
    if server_pid not in rows:
        raise RuntimeError("server process exited")
    total = 0
    pending = [server_pid]
    while pending:
        pid = pending.pop()
        total += rows[pid][1]
        pending.extend(child for child, (parent, _) in rows.items() if parent == pid)
    return total


def health_status(port: int) -> int | None:
    request = Request(f"http://127.0.0.1:{port}/health")
    try:
        with urlopen(request, timeout=1) as response:
            return response.status
    except HTTPError as error:
        return error.code
    except (URLError, TimeoutError):
        return None


def chat(port: int, key: str) -> None:
    body = json.dumps(
        {
            "model": "local-smollm2",
            "messages": [{"role": "user", "content": "Say hi"}],
            "max_completion_tokens": 2,
        }
    ).encode()
    request = Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )
    with urlopen(request, timeout=15) as response:
        if response.status != 200:
            raise RuntimeError(f"chat returned HTTP {response.status}")
        result = json.load(response)
    if not result.get("choices") or not result.get("usage"):
        raise RuntimeError("chat response has no completion or usage")


def unused_loopback_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def stop_group(server: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(server.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        server.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait(timeout=5)


def run(binary: Path, model: Path, cycles: int, metal: bool) -> dict[str, object]:
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError("server binary does not exist or is not executable")
    if not model.is_file():
        raise ValueError("GGUF model file does not exist")
    port = unused_loopback_port()
    key = secrets.token_hex(32)
    environment = os.environ.copy()
    environment["INFERENCE_API_KEY"] = key
    command = [str(binary.resolve())]
    if metal:
        command.append("--metal")
    command.extend([str(model.resolve()), str(port)])
    server = subprocess.Popen(
        command,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 30
        while health_status(port) != 200:
            if server.poll() is not None:
                raise RuntimeError("server exited before becoming ready")
            if time.monotonic() >= deadline:
                raise RuntimeError("server did not become ready within 30 seconds")
            time.sleep(0.05)
        chat(port, key)

        results: list[dict[str, object]] = []
        peak_rss = 0
        for index in range(cycles):
            rows = process_rows()
            worker = direct_worker(server.pid, rows)
            if worker is None:
                raise RuntimeError(f"cycle {index + 1}: ready server has no worker")
            peak_rss = max(peak_rss, tree_rss_kib(server.pid, rows))
            # Confirm parentage again immediately before injecting the failure.
            if direct_worker(server.pid, process_rows()) != worker:
                raise RuntimeError(f"cycle {index + 1}: worker changed before injection")
            started = time.monotonic()
            os.kill(worker, signal.SIGKILL)
            observed_unavailable = False
            replacement: int | None = None
            deadline = started + 15
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"cycle {index + 1}: server exited")
                status = health_status(port)
                observed_unavailable |= status == 503
                rows = process_rows()
                peak_rss = max(peak_rss, tree_rss_kib(server.pid, rows))
                replacement = direct_worker(server.pid, rows)
                if replacement is not None and replacement != worker and status == 200:
                    break
                time.sleep(0.025)
            else:
                raise RuntimeError(f"cycle {index + 1}: worker did not recover in 15 seconds")
            ready_ms = round((time.monotonic() - started) * 1000, 1)
            chat(port, key)
            ready_rss = tree_rss_kib(server.pid, process_rows())
            peak_rss = max(peak_rss, ready_rss)
            results.append(
                {
                    "cycle": index + 1,
                    "ready_after_ms": ready_ms,
                    "chat_completed_after_ms": round((time.monotonic() - started) * 1000, 1),
                    "observed_http_503": observed_unavailable,
                    "post_chat_tree_rss_kib": ready_rss,
                }
            )
        return {
            "model": model.name,
            "backend": "metal" if metal else "cpu",
            "cycles": results,
            "peak_sampled_tree_rss_kib": peak_rss,
            "method": "SIGKILL direct worker; poll health and child PID; complete a real chat",
        }
    finally:
        stop_group(server)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="path to the GGUF checkpoint")
    parser.add_argument("--binary", type=Path, default=Path("target/release/serve"))
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--metal", action="store_true", help="run the Metal worker")
    args = parser.parse_args()
    if not 1 <= args.cycles <= 100:
        parser.error("cycles must be 1 through 100")
    try:
        result = run(args.binary, args.model, args.cycles, args.metal)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"worker recovery failed: {error}\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
