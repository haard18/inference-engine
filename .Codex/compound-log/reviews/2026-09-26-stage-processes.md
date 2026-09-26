# 2026-09-26 - Stage process review

## Delivered

- A private stage-worker mode in the existing server executable.
- Separate prefix and suffix processes that load only their own model weights.
- A bounded protocol for token inputs, raw activation frames, raw score outputs, and explicit session close.
- At most eight live request sessions per worker, 128 MiB of allocated stage cache, and five-minute idle cleanup.

## Validation

The real Q4_K_M test ran both stage workers as child processes, passed four activations through the parent, and matched complete-model scores exactly. It checked model digests and layer ranges, rejected an activation with the wrong request ID, refused a ninth session, and admitted it after a close. A single idle resident-memory sample measured 119.5 MiB for a full worker and 65.5 MiB for each stage worker. Strict Clippy, formatting, and all 52 release tests passed.

## Remaining work

The parent process in this test is a local harness. Production serving still needs to place stages on approved devices, carry activations through mutual TLS, enforce the original request deadline across both workers, and fail clearly when a stage or the link is lost. Memory under load and two-Mac throughput remain unmeasured.
