#!/usr/bin/env python3
"""Run one-host complete and split Metal serving checks with matching load."""

import argparse
import json
import os
import runpy
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, BinaryIO

REPO = Path(__file__).resolve().parents[1]
BIN = REPO / 'target/release'
HELP = runpy.run_path(str(REPO / 'scripts/local-soak.py'))


def run_checked(command: list[str], **kwargs: Any) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False, **kwargs)
    if result.returncode:
        raise RuntimeError(f'{command[0]} exited {result.returncode}: {result.stderr[-1000:]}')
    return result.stdout


def wait_health(port: int, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 35
    while HELP['health'](port) != 200:
        if process.poll() is not None:
            raise RuntimeError('server exited before it became ready')
        if time.monotonic() >= deadline:
            raise RuntimeError('server readiness timed out')
        time.sleep(0.05)


def wait_listener(port: int, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 35
    while True:
        if process.poll() is not None:
            raise RuntimeError('suffix exited before it began listening')
        try:
            with socket.create_connection(('127.0.0.1', port), timeout=0.25):
                return
        except (OSError, TimeoutError):
            if time.monotonic() >= deadline:
                raise RuntimeError('suffix listener timed out')
            time.sleep(0.05)


def start(
    command: list[str],
    env: dict[str, str],
    logs: list[BinaryIO],
    processes: list[subprocess.Popen[bytes]],
) -> subprocess.Popen[bytes]:
    log = tempfile.TemporaryFile()
    logs.append(log)
    process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    processes.append(process)
    return process


def run_trial(
    port: int,
    roots: dict[str, int],
    mode: str,
    requests: int,
    concurrency: int,
    tokens: int,
    env: dict[str, str],
) -> tuple[dict[str, Any], str, dict[str, int], dict[str, int], dict[str, set[int]]]:
    stop = threading.Event()
    peaks = {name: 0 for name in roots}
    counts = {name: 0 for name in roots}
    workers = {name: set() for name in roots}
    errors = []

    def sample() -> None:
        while not stop.is_set():
            try:
                rows = HELP['process_rows']()
                for name, pid in roots.items():
                    rss, worker = HELP['process_tree'](pid, rows)
                    peaks[name] = max(peaks[name], rss)
                    counts[name] += 1
                    workers[name].add(worker)
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                errors.append(str(error))
                stop.set()
                return
            stop.wait(0.1)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    command = [str(BIN / 'pool-bench'), f'127.0.0.1:{port}',
               'local-smollm2', str(requests), str(concurrency), str(tokens)]
    if mode == 'stream':
        command.append('--stream')
    try:
        report = json.loads(run_checked(command, env=env,
                                        timeout=max(120, requests * 10)))
    finally:
        stop.set()
        thread.join(timeout=5)
    if thread.is_alive() or errors:
        raise RuntimeError(f'memory sampling failed: {errors}')
    digest = HELP['check_report'](report, mode, requests, concurrency, tokens)
    if any(count < 2 for count in counts.values()):
        raise RuntimeError('too few memory samples')
    if any(len(ids) != 1 for ids in workers.values()):
        raise RuntimeError(f'a model worker changed during a trial: {workers}')
    return report, digest, peaks, counts, workers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('--requests', type=int, default=20)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--concurrency', type=int, default=2)
    parser.add_argument('--max-tokens', type=int, default=16)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != 'darwin':
        parser.error('Metal split load checks require macOS')
    if (
        not args.model.is_file()
        or not 1 <= args.requests <= 10_000
        or not 1 <= args.trials <= 20
        or not 1 <= args.concurrency <= 4
        or not 1 <= args.max_tokens <= 256
    ):
        parser.error('model must exist; requests 1..10000, trials 1..20, concurrency 1..4, tokens 1..256')
    env = os.environ.copy()
    env['INFERENCE_API_KEY'] = secrets.token_hex(32)
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    expected_digest = {}
    expected_workers = {}

    for scenario in ('whole', 'split'):
        processes = []
        logs = []
        try:
            if scenario == 'whole':
                port = HELP['unused_port']()
                server = start([str(BIN / 'serve'), '--metal',
                                str(args.model.resolve()), str(port)], env, logs, processes)
                wait_health(port, server)
                roots = {'whole': server.pid}
            else:
                with tempfile.TemporaryDirectory() as temporary:
                    state = Path(temporary)
                    prefix_state = state / 'prefix'
                    suffix_state = state / 'suffix'
                    for directory in (prefix_state, suffix_state):
                        run_checked([str(BIN / 'device'), 'init', str(directory)])
                    prefix_offer = state / 'prefix-offer.json'
                    suffix_offer = state / 'suffix-offer.json'
                    prefix_offer.write_text(run_checked([str(BIN / 'device'), 'offer', str(prefix_state)]))
                    suffix_offer.write_text(run_checked([str(BIN / 'device'), 'offer', str(suffix_state)]))
                    prefix_data = json.loads(prefix_offer.read_text())
                    suffix_data = json.loads(suffix_offer.read_text())
                    suffix_port = HELP['unused_port']()
                    port = HELP['unused_port']()
                    while port == suffix_port:
                        port = HELP['unused_port']()
                    run_checked([str(BIN / 'device'), 'trust', str(suffix_state),
                                 str(prefix_offer), f'127.0.0.1:{port}', prefix_data['fingerprint']])
                    run_checked([str(BIN / 'device'), 'trust', str(prefix_state),
                                 str(suffix_offer), f'127.0.0.1:{suffix_port}', suffix_data['fingerprint']])
                    suffix = start([str(BIN / 'serve'), '--metal', '--stage-suffix',
                                    str(suffix_state), f'127.0.0.1:{suffix_port}',
                                    str(args.model.resolve()), '15', '30'], env, logs, processes)
                    wait_listener(suffix_port, suffix)
                    prefix = start([str(BIN / 'serve'), '--metal', '--split-prefix',
                                    str(prefix_state), suffix_data['device_id'],
                                    str(args.model.resolve()), '15', str(port)], env, logs, processes)
                    wait_health(port, prefix)
                    if suffix.poll() is not None:
                        raise RuntimeError('suffix exited before split readiness')
                    roots = {'prefix': prefix.pid, 'suffix': suffix.pid}
                    measure(scenario, port, roots, args, env, reports, expected_digest, expected_workers)
                    continue
            measure(scenario, port, roots, args, env, reports, expected_digest, expected_workers)
        except Exception:
            for log in logs:
                log.seek(0, os.SEEK_END)
                log.seek(max(0, log.tell() - 1500))
                print(log.read().decode(errors='replace'), file=sys.stderr)
            raise
        finally:
            for process in reversed(processes):
                HELP['stop_group'](process)
            for log in logs:
                log.close()

    print(json.dumps({'schema_version': 1, 'physical_devices': 1,
                      'backend': 'metal', 'reports': reports}, indent=2))


def measure(
    scenario: str,
    port: int,
    roots: dict[str, int],
    args: argparse.Namespace,
    env: dict[str, str],
    reports: list[dict[str, Any]],
    expected_digest: dict[str, str],
    expected_workers: dict[str, int],
) -> None:
    for trial in range(args.trials):
        for mode in ('ordinary', 'stream'):
            report, digest, peaks, counts, workers = run_trial(
                port, roots, mode, args.requests, args.concurrency,
                args.max_tokens, env)
            for name, ids in workers.items():
                key = f'{scenario}/{name}'
                prior_worker = expected_workers.setdefault(key, next(iter(ids)))
                if next(iter(ids)) != prior_worker:
                    raise RuntimeError(f'{name} worker restarted between trials')
            if HELP['health'](port) != 200:
                raise RuntimeError(f'{scenario} became unhealthy after a trial')
            prior = expected_digest.setdefault('all', digest)
            if prior != digest:
                raise RuntimeError(f'{scenario}/{mode} generated different text')
            name = f'{scenario}-{mode}-{trial+1}.json'
            (args.output / name).write_text(json.dumps(report, indent=2) + '\n')
            reports.append({'scenario': scenario, 'mode': mode, 'trial': trial + 1,
                            'requests_per_second': report['requests_per_second'],
                            'p50_ms': report['p50_ms'], 'p95_ms': report['p95_ms'],
                            'first_content_p50_ms': report['first_content_p50_ms'],
                            'peak_tree_rss_kib': peaks, 'memory_samples': counts})
            print(f'{scenario} {mode} {trial+1}/{args.trials}: '
                  f'{report["successful"]}/{args.requests} complete',
                  file=sys.stderr, flush=True)


if __name__ == '__main__':
    main()
