# 2026-09-26 - One-host serving measurement

The benchmark now has an optional streaming mode. It parses server-sent events as bytes arrive, including events split across network frames. It records first visible content, full response time, HTTP status, and application stream errors. A stream counts as successful only after a finish reason and `[DONE]`. Ordinary mode still uses the API's exact completion-token count for throughput. First visible content is a client-observed time; it can lag the first generated token when text decoding buffers bytes.

The new `scripts/process-rss.py` samples `ps` every configured interval and sums each named server process with its descendants. This captures both gateway and model child without reading process command lines or secrets. It reports the highest sampled value, not a guaranteed instantaneous peak. The script runs with the system Python 3 interpreter on this Mac.

One-host SmolLM2-135M Q4_K_M pilot, 12 requests per run, 16 requested completion tokens, two warm-ups:

| Mode | Concurrency | Requests/s | Median full response | Median first visible content | Completion tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Complete worker, ordinary | 1 | 0.615 | 1,627 ms | — | 9.84 |
| Complete worker, streaming | 1 | 0.613 | 1,630 ms | 950 ms | — |
| Split 15/15, ordinary | 1 | 0.606 | 1,653 ms | — | 9.69 |
| Split 15/15, streaming | 1 | 0.605 | 1,652 ms | 962 ms | — |
| Complete worker, ordinary | 2 | 0.614 | 3,259 ms | — | 9.82 |
| Split 15/15, ordinary | 2 | 0.603 | 3,317 ms | — | 9.64 |
| Whole-request pool, ordinary | 2 | 1.214 | 1,644 ms | — | 19.42 |
| Whole-request pool, ordinary, second run | 2 | 1.039 | 1,655 ms | — | 16.62 |

Every pilot request returned HTTP 200 and completed. The first pool run placed six requests on each worker; the second placed seven and five. The separate pool streaming run served 1.045 requests/s and had 958 ms median first visible content, but its 95th percentile rose to 2,587 ms. These short runs are exploratory; scheduling, cache warmth, and host contention can affect them.

During concurrency-two load, process-tree samples with 100 ms sleeps between `ps` calls found maxima of 159,232 KiB (155.5 MiB) for the complete worker; 101,328 KiB (99.0 MiB) for the split prefix and 84,400 KiB (82.4 MiB) for the split suffix; and 160,960 and 160,592 KiB (about 157 MiB each) for two pooled complete workers. `ps` execution time made actual spacing longer than 100 ms. The split stages held less resident memory per simulated device than the complete worker. Their combined memory on one host was higher because tied embeddings and process overhead are duplicated. A real LAN adds network latency and may change every timing result. Two physical Macs and longer repeated trials remain necessary for a deployment claim.
