# Reuse the split activation connection

## Need

The suffix receives a request for each generated token. The approved peer client currently opens a new TCP, mutual TLS, and HTTP connection for each request. That adds repeated setup to the serial token path and may dominate latency on a real LAN. The [prompt-batching learning](../learnings/2026-09-26-split-prefill-batches.md) removed avoidable prompt requests; the [capacity-readiness learning](../learnings/2026-09-26-split-capacity-readiness.md) requires health checks to remain independent of an active step.

## Approach

- Keep one authenticated HTTP/1 connection per approved suffix client for activation calls. Share it among clones of that client and serialize activation calls until the bounded response body is consumed.
- Keep capacity probes on independent connections so an active token step cannot block readiness checks behind a client-side lock.
- Reconnect before a new activation if the previous connection has closed. If an activation send or response fails, discard the connection and report the failure without replaying that activation.

## Acceptance

- A real stage test proves multiple activations use one underlying TLS connection and retain output parity with the complete model.
- Closing the connection or dropping the suffix preserves existing failure and recovery behavior; no activation is silently replayed.
- The split and peer regression tests, release suite, strict Clippy, formatting, and build pass.

## Progress

- [x] Keep an authenticated HTTP/1 sender for activation requests and share it among peer-client clones. Read each bounded response before releasing the sender lock. Reconnect if the sender is already closed; never replay an activation after an uncertain send failure.
- [x] Leave capacity snapshots on separate connections. A transparent proxy in the real-model stage test counted one TLS connection for multiple valid and rejected activation calls while score parity held.
- [x] Run a one-Mac 1.7B Metal smoke check: 10 complete and 10 split requests in each of ordinary and streaming modes, all 40 completed with matching text and stable workers. Split loopback throughput was 1.774 ordinary and 1.787 streaming requests/s in this single trial; no speed gain is established.
- [x] Update the stream-loss test to shut down active server connections. Aborting only the accept task leaves a reused connection alive, so it did not simulate physical device loss. The targeted loss-and-recovery test passed after the correction.
- [x] Pass all 74 release tests with real-model cases included, strict Clippy, formatting, whitespace checks, and the release binary build.
