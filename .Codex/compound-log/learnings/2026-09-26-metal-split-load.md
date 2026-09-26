# 2026-09-26 - One-host Metal split load

The split Metal chat test proved parity for a few requests, but it did not show behavior under concurrent load. A repeatable `scripts/split-soak.py` check now starts one complete Metal server, then a mutually approved prefix and suffix pair on loopback. It runs the same benchmark body, model, concurrency, and completion limit in both scenarios. It validates every response status and digest, samples each server process tree, and checks that stage child process IDs remain stable across trials. It uses temporary device identities and removes the services at the end.

On one Apple Silicon Mac with SmolLM2-135M Q4_K_M, three trials per scenario and mode sent 20 measured requests each, after two warm-ups per trial, with concurrency two and a 16-token limit. All 240 measured requests succeeded. Twelve full benchmark reports passed the existing report validator and had one common output digest. No model worker restarted.

| Scenario | Mode | Median requests/s | Median response p50 | Median response p95 | Median first visible content |
| --- | --- | ---: | ---: | ---: | ---: |
| Complete Metal | Ordinary | 4.231 | 472 ms | 476 ms | — |
| Split Metal | Ordinary | 3.911 | 511 ms | 514 ms | — |
| Complete Metal | Stream | 4.232 | 472 ms | 475 ms | 372 ms |
| Split Metal | Stream | 3.548 | 542 ms | 780 ms | 402 ms |

The split ordinary median rate was about 7.6% lower. Split streaming varied more: its three rates were 3.918, 3.548, and 3.455 requests/s. The physical host ran only one scenario at a time, but all stages shared that host's CPU and GPU. These are local loopback observations, not predictions for a network of Macs.

At 100 ms nominal sampling intervals, the highest complete-server process-tree RSS was 275,968 KiB. The highest sampled prefix and suffix process-tree RSS values were 153,152 and 155,168 KiB. Later samples were lower even though the same stage workers remained alive. The available evidence does not establish why those resident-memory values changed; the report treats them as sampled maxima, not stable working-set sizes or guaranteed peaks.

The check does not cover a physical LAN, mixed CPU/GPU devices, or long-duration operation. A second Mac is not set up yet, so two-device throughput, transfer latency, and device-loss behavior under load remain open.
