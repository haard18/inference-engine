# 2026-09-26 - Longer Metal split serving check

After adding request-deadline leases to active split sessions, rebuild the release server and run the existing one-host load check again. The run uses SmolLM2-135M Q4_K_M, two warm-ups per trial, 30 measured requests per trial, concurrency two, and a 16-token limit. It runs six ordinary and six streaming trials for each of complete Metal serving and a 15/15 Metal split through mutually approved loopback TLS. The scenarios run one after the other on one physical Mac.

All 720 measured requests completed with HTTP 200. All 24 full benchmark reports passed the report validator and shared the completed-text digest `4c65e5358e5654545275285959535d068e006fd0ff7ffb00a0f809902036fe0b`. The load script saw the same model-worker process IDs across all trials; it found no worker restart.

| Scenario | Mode | Median requests/s | Median response p50 | Median response p95 | Median first visible content |
| --- | --- | ---: | ---: | ---: | ---: |
| Complete Metal | Ordinary | 4.177 | 478 ms | 487 ms | — |
| Split Metal | Ordinary | 3.860 | 518 ms | 521 ms | — |
| Complete Metal | Stream | 4.183 | 478 ms | 484 ms | 376 ms |
| Split Metal | Stream | 3.863 | 518 ms | 520 ms | 404 ms |

The split ordinary median rate was about 7.6% lower in this run. The highest sampled process-tree resident memory was 275,984 KiB for complete serving, 152,784 KiB for the prefix, and 155,216 KiB for the suffix. Resident-memory samples later fell even though worker IDs stayed unchanged. The 100 ms sampling interval can miss short peaks, and resident memory does not measure all GPU allocation. These values do not prove a stable working-set size.

This is stronger one-host evidence that the new lease protocol does not break ordinary or streaming split serving under repeated requests. It still uses one prompt and one physical Mac. LAN transfer, two-device capacity, and device-loss behavior under live load remain open until the second Mac is available.
