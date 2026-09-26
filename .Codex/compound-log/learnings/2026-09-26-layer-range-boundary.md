# 2026-09-26 - Decoder layer-range boundary

**Context:** Prepare the decoder for one model's layers to run on different devices.

**Learning:** The hidden vector is the natural transfer boundary between contiguous decoder layer ranges. Each range uses the same token position but touches only the key/value state for its own layers. The current full-model path can call the same layer-slice executor as a two-range path.

**Pattern:** Compute new key/value entries as pending data and commit them only after final output projection succeeds. This keeps a failed token step from advancing only part of the cache. Validate hidden width, finite values, and cache positions at the range boundary.

**Anti-pattern:** Treating local layer-range parity as proof of distributed model splitting. The parity test still holds all weights and caches in one process; selective loading, independent stage ownership, transport, and failure handling remain necessary.
