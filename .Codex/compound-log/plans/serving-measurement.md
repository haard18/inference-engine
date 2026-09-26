# Serving measurement

## Need

The existing benchmark measures completed ordinary chat requests and completion-token throughput. Split serving adds a network step per prompt and generated token, so the remaining comparison also needs client-visible first output timing and a way to distinguish a completed stream from an HTTP 200 stream that ends with an error.

## Approach

Extend `pool-bench` with an optional streaming mode. Keep the same fixed prompt, warm-up count, request count, concurrency, and loopback endpoint limit. Parse SSE events incrementally with bounded memory. Measure the time from connection start to the first visible content delta, total request latency, successful completions, HTTP statuses, and stream error codes. Report token throughput only for ordinary responses, where the API returns exact completion-token usage. Use the same command on a complete worker, a whole-request pool, and a split prefix to make results comparable.

For resident memory, measure the complete worker and both split stage process trees separately while load runs. Do not combine co-located measurements into a claim about two physical machines. The physical comparison requires the second Mac's reachable address and matching model file.

## Acceptance

- A fragmented SSE stream yields one first-content time and counts as successful only after a finish reason and `[DONE]`.
- An SSE error following HTTP 200 counts as an application failure, not a successful request.
- The ordinary benchmark output remains comparable to previous runs.
- One-host complete, pooled, and split baselines are recorded with limits; the two-Mac result stays open until measured on two devices.

## Progress

- [x] Extend `pool-bench` with bounded SSE parsing, first visible content timing, and separate stream error accounting. A fragmented stream test covers success, an application error, and an incomplete stream.
- [x] Add a process-tree RSS sampler that runs on the local Python 3 runtime and reports sampled maxima.
- [x] Run one-host Q4_K_M pilots for a complete worker, a 15/15 split over loopback TLS, and two complete workers with whole-request routing. Record ordinary and streaming results and concurrent RSS samples in the learning log.
- [ ] Repeat with sustained load on two physical Macs, including link loss and recovery, and compare per-device memory, first visible content, completion rate, and latency variation across trials.
