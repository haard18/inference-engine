#!/usr/bin/env python3
"""Check local serving across repeated ordinary and streaming load waves."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def process_rows() -> dict[int, tuple[int, int]]:
    output = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,rss="],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    rows = {}
    for line in output.splitlines():
        pid, parent, rss_kib = map(int, line.split())
        rows[pid] = (parent, rss_kib)
    return rows


def process_tree(root: int, rows: dict[int, tuple[int, int]]) -> tuple[int, int]:
    if root not in rows:
        raise RuntimeError("server process disappeared")
    direct = [pid for pid, (parent, _) in rows.items() if parent == root]
    if len(direct) != 1:
        raise RuntimeError(f"expected one model worker, found {len(direct)}")
    pending = [root]
    rss_kib = 0
    while pending:
        pid = pending.pop()
        rss_kib += rows[pid][1]
        pending.extend(child for child, (parent, _) in rows.items() if parent == pid)
    return rss_kib, direct[0]


def health(port: int) -> int | None:
    try:
        with urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
            return response.status
    except HTTPError as error:
        return error.code
    except (URLError, TimeoutError, socket.timeout):
        return None


def unused_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def check_report(
    report: object, mode: str, requests: int, concurrency: int, max_tokens: int
) -> str:
    if not isinstance(report, dict):
        raise RuntimeError("benchmark did not return a JSON object")
    if (
        report.get("schema_version") != 1
        or report.get("mode") != mode
        or report.get("model") != "local-smollm2"
        or report.get("requested") != requests
        or report.get("concurrency") != concurrency
        or report.get("max_completion_tokens") != max_tokens
        or report.get("warmup_requests") != 2
    ):
        raise RuntimeError("benchmark model or load settings changed")
    if report.get("successful") != requests or report.get("http_statuses") != {"200": requests}:
        raise RuntimeError(f"{mode} wave did not complete every request")
    if report.get("transport_failures") or report.get("application_errors"):
        raise RuntimeError(f"{mode} wave reported transport or application failures")
    owners = report.get("requests_by_device")
    if (
        not isinstance(owners, dict)
        or len(owners) != 1
        or "unknown" in owners
        or next(iter(owners.values())) != requests
    ):
        raise RuntimeError("requests were not all served by the same local device")
    digests = report.get("completion_digests")
    if not isinstance(digests, dict) or len(digests) != 1:
        raise RuntimeError(f"{mode} wave produced differing text")
    digest, count = next(iter(digests.items()))
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or count != requests
    ):
        raise RuntimeError(f"{mode} wave has an invalid output digest")
    if mode == "stream" and report.get("responses_with_content") != requests:
        raise RuntimeError("stream wave had a response without visible content")
    return digest


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--server", type=Path, default=Path("target/release/serve"))
    parser.add_argument("--bench", type=Path, default=Path("target/release/pool-bench"))
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--waves", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--interval-ms", type=int, default=200)
    parser.add_argument("--metal", action="store_true")
    args = parser.parse_args()
    if not args.model.is_file() or not args.server.is_file() or not args.bench.is_file():
        parser.error("model, server binary, and benchmark binary must exist")
    if (
        not 1 <= args.requests <= 10_000
        or args.waves < 2
        or args.waves % 2
        or not 1 <= args.concurrency <= 4
        or not 1 <= args.max_tokens <= 256
        or not 50 <= args.interval_ms <= 10_000
    ):
        parser.error("use positive requests, even waves >= 2, concurrency 1..4, tokens 1..256, interval 50..10000 ms")

    port = unused_port()
    environment = os.environ.copy()
    environment["INFERENCE_API_KEY"] = secrets.token_hex(32)
    command = [str(args.server.resolve())]
    if args.metal:
        command.append("--metal")
    command.extend((str(args.model.resolve()), str(port)))
    with tempfile.TemporaryFile() as server_log:
        server = subprocess.Popen(
            command,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        stop = threading.Event()
        samples: list[tuple[float, int, int]] = []
        sampler_errors: list[str] = []

        def sample_memory() -> None:
            while not stop.is_set():
                try:
                    rss_kib, worker = process_tree(server.pid, process_rows())
                    samples.append((time.monotonic(), rss_kib, worker))
                except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                    sampler_errors.append(str(error))
                    stop.set()
                    return
                stop.wait(args.interval_ms / 1000)

        sampler: threading.Thread | None = None
        try:
            deadline = time.monotonic() + 30
            while health(port) != 200:
                if server.poll() is not None:
                    raise RuntimeError("server exited before readiness")
                if time.monotonic() >= deadline:
                    raise RuntimeError("server did not become ready within 30 seconds")
                time.sleep(0.05)
            baseline_rows = process_rows()
            baseline_rss, initial_worker = process_tree(server.pid, baseline_rows)
            baseline_server_rss = baseline_rows[server.pid][1]
            baseline_worker_rss = baseline_rows[initial_worker][1]
            sampler = threading.Thread(target=sample_memory, daemon=True)
            sampler.start()

            waves = []
            expected_digest = None
            for index in range(args.waves):
                mode = "ordinary" if index % 2 == 0 else "stream"
                bench_command = [
                    str(args.bench.resolve()),
                    f"127.0.0.1:{port}",
                    "local-smollm2",
                    str(args.requests),
                    str(args.concurrency),
                    str(args.max_tokens),
                ]
                if mode == "stream":
                    bench_command.append("--stream")
                started = time.monotonic()
                completed = subprocess.run(
                    bench_command,
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=max(120, args.requests * 5),
                )
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"benchmark exited with {completed.returncode}: "
                        f"{completed.stderr[-2000:]}"
                    )
                ended = time.monotonic()
                report = json.loads(completed.stdout)
                digest = check_report(
                    report, mode, args.requests, args.concurrency, args.max_tokens
                )
                if expected_digest is not None and digest != expected_digest:
                    raise RuntimeError("ordinary and streaming waves produced different text")
                expected_digest = digest
                if health(port) != 200:
                    raise RuntimeError(f"server became unhealthy after wave {index + 1}")
                end_rows = process_rows()
                end_rss, worker = process_tree(server.pid, end_rows)
                if worker != initial_worker:
                    raise RuntimeError(f"model worker restarted after wave {index + 1}")
                observed = [rss for when, rss, _ in samples if started <= when <= ended]
                waves.append(
                    {
                        "mode": mode,
                        "requested": args.requests,
                        "successful": report["successful"],
                        "requests_per_second": report["requests_per_second"],
                        "p50_ms": report["p50_ms"],
                        "p95_ms": report["p95_ms"],
                        "first_content_p50_ms": report["first_content_p50_ms"],
                        "sampled_peak_tree_rss_kib": max(observed, default=end_rss),
                        "post_wave_tree_rss_kib": end_rss,
                        "post_wave_server_rss_kib": end_rows[server.pid][1],
                        "post_wave_worker_rss_kib": end_rows[worker][1],
                        "memory_samples": len(observed),
                    }
                )
                print(
                    f"wave {index + 1}/{args.waves}: {mode} {args.requests}/{args.requests} complete",
                    file=sys.stderr,
                    flush=True,
                )
            stop.set()
            sampler.join(timeout=5)
            if sampler.is_alive() or sampler_errors:
                raise RuntimeError(f"memory sampling failed: {sampler_errors}")
            if any(worker != initial_worker for _, _, worker in samples):
                raise RuntimeError("model worker changed during a wave")
            print(
                json.dumps(
                    {
                        "schema_version": 1,
                        "model": str(args.model.resolve()),
                        "backend": "metal" if args.metal else "cpu",
                        "concurrency": args.concurrency,
                        "max_completion_tokens": args.max_tokens,
                        "output_digest": expected_digest,
                        "worker_stable": True,
                        "baseline_tree_rss_kib": baseline_rss,
                        "baseline_server_rss_kib": baseline_server_rss,
                        "baseline_worker_rss_kib": baseline_worker_rss,
                        "sampled_peak_tree_rss_kib": max((rss for _, rss, _ in samples), default=baseline_rss),
                        "waves": waves,
                    },
                    indent=2,
                )
            )
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, json.JSONDecodeError) as error:
            server_log.seek(0, os.SEEK_END)
            server_log.seek(max(0, server_log.tell() - 2000))
            detail = server_log.read().decode(errors="replace")
            parser.exit(1, f"local load check failed: {error}\n{detail}\n")
        finally:
            stop.set()
            if sampler is not None:
                sampler.join(timeout=5)
            stop_group(server)


if __name__ == "__main__":
    main()
