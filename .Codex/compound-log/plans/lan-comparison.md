# Reviewable two-Mac comparison

## Need

The remaining model-splitting and device-pooling checks require repeated measurements on two physical Macs. The current benchmark reports rate and latency, but it could not show whether all workers generated the same text. Its latency percentiles also included failed responses, which could make an outage look fast.

## Approach

Record a digest of completed text in both ordinary and streaming benchmark modes. Compute latency percentiles from successful requests. Save structured metadata for matching trial settings. Add a report tool that requires repeated complete, pooled, and split trials, checks every request succeeded, checks generated-text parity, and confirms both pooled devices handled requests at concurrency two or higher. Keep process-tree RSS samples and server startup records beside the reports for the physical run.

## Acceptance

- Ordinary and fragmented streaming responses with the same text have the same digest.
- Comparison rejects failed requests, changed model or load settings, changed generated text, and a pool that uses only one device at concurrency two.
- A real-model loopback smoke run produces one shared digest for ordinary and streaming requests.
- Physical two-Mac trials capture complete, pooled, and split throughput, latency, first visible content, device memory, and link-loss behavior.

## Progress

- [x] Add completed-text digests and successful-request latency to the benchmark.
- [x] Add a comparison tool with repeated-trial and parity checks.
- [x] Run a six-request ordinary and streaming real-model loopback smoke test; all requests completed and both modes produced the same digest.
- [x] Verify peer loss and return in a real-model paired serving test while the local worker is down. The test checks unavailable responses during loss and confirms the restarted peer owns the recovered response.
- [x] Write a physical two-Mac runbook with separate pool and split pairing states, matching model checksums, report collection, per-host memory samples, and separate loss trials.
- [ ] Run repeated sustained trials and loss/recovery checks on two physical Macs. The second Mac's reachable SSH address and model path are not available yet.
