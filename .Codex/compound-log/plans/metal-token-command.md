# One Metal command per token

## Need

The decoder now ran each layer in one command, but it still waited and read a hidden vector after every layer. Final normalization and output projection made another CPU/GPU boundary. A 30-layer model therefore had many per-token waits even though each layer's operations were already on the GPU.

## Approach

Encode all decoder layers and the final score projection into one ordered Metal command. Pass each layer's output buffer directly to the next layer. Upload final normalization weights once. Retain temporary buffers through command completion. Keep the session position outside the command as the commit marker; a failed command must leave the position unchanged so a retry can overwrite its uncommitted cache slots.

## Acceptance

- Tiny f32/f16 and real bf16, Q8_0, and Q4_K_M scores stay within the established CPU/Metal tolerances at multiple positions.
- Cache growth and conversation rewind still work.
- The full release suite, warnings-denied Clippy, format check, and release build pass.
- A short local probe records timing and resident memory without a broad speed claim.

## Progress

- [x] Keep hidden buffers on the GPU across all layers and final projection.
- [x] Retain all command inputs and intermediates until completion; commit the token position only after finite scores return.
- [x] Pass the 59-test release suite and record a short local measurement.
