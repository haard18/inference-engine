# One Metal command per decoder layer

## Need

The Metal decoder finished attention, attention-output projection, and feed-forward work in three separate commands. Each command waited for completion so the CPU could normalize or add a residual vector. These waits occurred for every layer and token.

## Approach

Upload each layer's normalization weights once. Add GPU kernels for RMS scale, normalization application, and residual addition. Dispatch those kernels with query/key/value projection, rotary positions, attention, and feed-forward work in one ordered command buffer per layer. Read back only the completed hidden vector. Preserve the session's separate committed position so a failed token or rewind can overwrite uncommitted key/value slots.

## Acceptance

- Tiny f32/f16 and real bf16, Q8_0, and Q4_K_M scores stay within their established CPU/Metal tolerances across multiple token positions.
- Key/value growth and conversation rewind still work.
- All release tests, warnings-denied Clippy, format check, and release build pass.
- A short local probe records timing and resident memory without a broad performance claim.

## Progress

- [x] Combine a decoder layer into one command and keep normalization weights in reusable GPU buffers.
- [x] Verify real-model and long-context cache parity in the release suite.
- [x] Record short local timing and resident-memory samples.
