# 2026-09-26 - Co-located pool measurement

**Context:** Check whether capacity-aware whole-request routing uses a second worker under concurrent live-model load before moving into model splitting.

**Method:** One Apple Silicon Mac ran either one server or two mutually trusted server processes, each with the same SmolLM2-135M Q4_K_M GGUF model on CPU. A loopback client warmed the endpoint with two requests, then sent 12 identical chat requests for at most 16 completion tokens at concurrency one, two, or four. These were single pilot runs, not repeated statistical trials. The paired servers ran on the same physical host, so the result does not measure LAN transfer or independent-device capacity. The client recorded full response latency, successful completion-token rate, HTTP status, transport failures, and the conversation ID's owner.

| Workers | Concurrency | Requests/s | Completion tokens/s | p50 ms | p95 ms | Requests by worker |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| One | 1 | 0.611 | 9.776 | 1,630 | 1,680 | 12 local |
| One | 2 | 0.613 | 9.802 | 3,259 | 3,289 | 12 local |
| One | 4 | 0.611 | 9.773 | 6,543 | 6,572 | 12 local |
| Two | 1 | 0.597 | 9.550 | 1,634 | 2,088 | 12 gateway |
| Two | 2 | 1.210 | 19.356 | 1,648 | 1,669 | 6 gateway, 6 peer |
| Two | 4 | 1.209 | 19.343 | 3,292 | 4,989 | 6 gateway, 6 peer |

Every measured request returned HTTP 200, with no transport failures. At concurrency two, the paired rate was 1.98 times the one-worker rate in this pilot. At concurrency one, the gateway used only its local worker. At concurrency four, each worker queued requests and latency grew. The post-run resident memory sample at paired concurrency two was 129.5 MiB for the gateway process tree and 125.0 MiB for the peer process tree; this is duplicated model memory, as expected for whole-request routing. A cold paired memory sample was inconsistent and is excluded.

**Learning:** Pooling complete requests increases concurrent throughput when each worker can hold the model, but it does not reduce the memory required per worker. A real two-Mac LAN run and repeated trials are needed before making a deployment performance claim.

**Pattern:** Measure throughput, latency, routing distribution, error counts, and memory together. Use the same workload for one and two workers, and identify the physical placement of workers in the result.

**Anti-pattern:** Treating two processes on one host as proof of two-device speedup, or calling a per-worker full-model copy a model split.
