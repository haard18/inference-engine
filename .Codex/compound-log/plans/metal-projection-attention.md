# Metal projection-to-attention boundary

## Need

Metal currently returns query, key, and value projections to the CPU, waits for the result, applies rotary positions, then writes key and value back into Metal buffers for attention. This is one avoidable per-layer CPU/GPU handoff.

## Approach

Build the rotary sine/cosine pairs once for each token position on the CPU. Dispatch the three existing Metal matrix projections, rotate query and key, write the new key/value slot, and run attention in one Metal command buffer. Read only the final attended vector back for the remaining CPU residual and feed-forward path. The committed position remains outside GPU buffers so failed steps and rewinds can overwrite an uncommitted slot.

## Acceptance

- Tiny f32/f16 and real bf16, Q8_0, and Q4_K_M scores stay within the established CPU/Metal tolerance over multiple token positions.
- Cache growth and conversation rewind still work; a failed step does not commit a position.
- Full release tests, Clippy, and release build pass.
- Measure local timing and memory on the same probe used for the previous Metal layer. Make no speed claim from a short run alone.

## Progress

- [x] Dispatch query/key/value projection, rotary rotation, key/value cache write, and attention in one Metal command buffer.
- [x] Compute the small rotary sine/cosine table once per token and preserve both supported rotary layouts.
- [x] Match tiny and real-model CPU scores; the full release suite covers cache growth and Metal conversation reuse.
- [x] Compare short warm local CPU and Metal runs and record their limits.
