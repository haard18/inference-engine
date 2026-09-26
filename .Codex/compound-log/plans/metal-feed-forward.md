# Metal feed-forward command

## Need

The Metal path returned gate and up projections to the CPU, applied the feed-forward activation there, then sent the activated vector to a second Metal matrix command. This caused one extra CPU/GPU completion wait per decoder layer and moved two intermediate vectors across the boundary.

## Approach

Dispatch gate and up projections, the SiLU gate activation, and the down projection in one Metal command. Keep the normalized input and final down-projection output as the current CPU boundaries. Check matrix dimensions and buffer limits before encoding. Preserve the session's existing position commit behavior.

## Acceptance

- Tiny f32/f16 and real bf16, Q8_0, and Q4_K_M CPU/Metal score comparisons pass at multiple positions.
- The complete release suite, warnings-denied Clippy check, release build, and format check pass.
- A short local probe records timing and resident memory without claiming a general speed gain.

## Progress

- [x] Add the GPU activation kernel and combine all three feed-forward projections in one command.
- [x] Verify tiny and real-model parity through the full release suite.
- [x] Record short local timing and resident-memory samples.
