# Capacity probes during split inference

## Need

The suffix holds its worker lock for each token step. Its capacity route waited for the same lock, while the prefix allowed only 750 ms for a probe. A legitimate slow step could therefore make the prefix mark a healthy suffix unavailable after two probes.

## Approach

Let the capacity route read worker state without waiting for an active step. Keep a readiness flag current when a child exits, restarts, completes work, or is dropped after a failed or canceled request. Keep queue availability separate from worker readiness.

## Acceptance

- A capacity request completes promptly while an activation is blocked in its child process, reports the worker ready, and reports no free queue slot.
- Canceling that activation discards its child, and a later activation succeeds with a replacement.
- The release suite, warnings-denied clippy check, and release build pass.

## Progress

- [x] Add a nonblocking capacity snapshot and cancellation-safe worker lease.
- [x] Extend the slow-step test to check the capacity response and subsequent replacement.
- [x] Run all 59 release tests, warnings-denied clippy, the release build, and a whitespace check.
