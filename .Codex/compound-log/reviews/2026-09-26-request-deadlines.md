# 2026-09-26 - Request deadline review

## Delivered

- A configurable shared deadline for queue wait and model execution; the CLI uses 120 seconds.
- HTTP 504 and streaming timeout behavior.
- Cancellation of expired queued work and bounded streaming forwarding.
- Continued service after a real-model execution error.
- Health and admission failure while an active calculation is overdue.

## Validation

- A model-free test holds requests in the queue until they expire, checks both response forms, and verifies a later request succeeds.
- A real SmolLM2 Q4_K_M test checks that a failed job does not stop later chat completions.
- A worker-status test checks HTTP 503 while overdue and readiness recovery after status clears.
- Full release suite, strict Clippy, and formatting checks.

## Scope

The in-process model calculation cannot be interrupted while it is inside a single forward operation. A permanently stuck calculation still needs process isolation and restart to restore capacity.
