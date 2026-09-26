# CPU matrix-row execution plan

## Purpose

Improve CPU token latency for compact GGUF models without changing the model calculation or stored weight format.

## Approach

Run independent output rows of sufficiently large matrix-vector operations in parallel. Keep small matrices serial to avoid thread scheduling overhead. Share one Rayon worker pool across calls and requests. Leave each row's arithmetic order unchanged so the independent NumPy and Metal comparisons remain useful.

## Acceptance

- A test crosses the parallel threshold and matches the serial row calculation.
- The real Q4_K_M model retains first-token score and generated-text parity with the current engine.
- A five-run CPU probe with the same model, prompt, and token count has lower median fresh-process latency than the recorded 0.571 s baseline. Record the measured result and conditions.
- Release tests, Clippy, formatting, and the real-model checks pass.

If latency does not improve, remove the parallel path and record the measurement.
