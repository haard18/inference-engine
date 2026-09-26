# 2026-09-26 - Capacity routing review

## Delivered

- A mutual TLS peer client for model capacity snapshots and ordinary or streamed chat forwarding.
- A coordinator that compares local backlog with compatible peers and routes each whole request to one worker.
- Short peer failure cooldown and local fallback before a response begins.
- An explicit error event when a peer stream ends before its completion marker.

## Validation

- A capacity test checked free queue slots and unavailable state.
- Real-model tests checked encrypted capacity lookup, ordinary and streamed forwarding, two-device routing while the local worker was busy, and local and peer loss.
- A truncated-response test checked that ordinary responses fail before forwarding and streams report an error before closing.
- All 44 release tests passed, including real-model CPU, Metal, and two-device checks. Strict Clippy, formatting, and diff checks passed.

## Scope

The coordinator uses a current snapshot, but admission remains race-prone by design; the destination worker enforces its queue bound. A peer stream that fails after bytes reach the caller reports an error and cannot be replayed safely. Conversation reuse, model splitting, and multi-device performance measurements remain open.
