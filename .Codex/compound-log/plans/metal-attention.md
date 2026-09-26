# Metal attention and key/value state

## Need

The current Metal path uploads matrix weights but sends every query, key, and value back to CPU attention. Its key/value cache also stays in CPU vectors. That creates a GPU/CPU boundary at every decoder layer and duplicates memory work.

## Approach

Give each Metal generation session key/value buffers that grow with the context. Store each new key/value vector at its token position. Run scaled dot-product scores and softmax-weighted value reduction in Metal kernels, then pass the attended vector to the existing matrix path. Keep the cache position as the commit marker: a failed token can overwrite its uncommitted GPU slot on retry, and rewind changes only the committed position. Keep CPU attention unchanged as the reference.

## Acceptance

- Tiny f32 and f16 models match CPU scores and token choices across multiple positions.
- Real bf16, Q8_0, and Q4_K_M models match CPU scores within the established Metal tolerance across multiple positions.
- Rewind and reuse preserve the same next-token result without reading stale cache slots.
- Metal cache memory is counted in the serving session limit, and capacity grows with used context rather than allocating the full context at startup.
- Record a local performance and memory comparison. Do not claim improvement unless measured.

## Progress

- [x] Add two Metal attention kernels and grow per-session key/value buffers with context length.
- [x] Keep the committed position outside the Metal buffers so failed steps and prompt rewinds can overwrite uncommitted slots.
- [x] Count persistent Metal cache buffers toward the serving checkpoint limit.
- [x] Check tiny-model cache growth, real Q8_0 and Q4_K_M parity, and Metal conversation reuse.
- [x] Record a short CPU/Metal timing and process-memory comparison.
