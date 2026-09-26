# 2026-09-26 - Pool measurement review

## Delivered

- A repeatable loopback benchmark client for the authenticated chat API.
- A live-model pilot with one and two co-located workers at three concurrency levels.
- A concrete model-splitting plan tied to the current executor, loader, and cache boundaries.

## Evidence and limits

All 72 measured requests across the six runs returned HTTP 200. At concurrency two, routing divided requests evenly and measured 1.98 times the one-worker request rate. The small single-run sample on one physical Mac cannot establish two-Mac throughput, network latency, reliability, or model-splitting benefit. The benchmark reports post-run resident memory, not peak memory. Physical LAN trials and model splitting remain open.

Strict Clippy, formatting, and the full release suite with the real checkpoints passed. The suite ran 47 tests, including the existing peer-loss and conversation migration cases.
