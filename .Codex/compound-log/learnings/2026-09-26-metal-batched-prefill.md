# 2026-09-26 - Batch known prompt tokens through each Metal layer

**Context:** A 42-token 1.7B Q4_K_M prompt took about 0.84–0.89 seconds to evaluate after unused score projections were removed. The complete Metal worker ran one decoder command per prompt token.

**Learning:** Encode up to eight known tokens together through each decoder layer. Normalize and project token-major buffers, write all keys and values for that batch, then apply causal attention so token `i` sees only earlier positions and positions through `i`. The cache grows before the GPU command starts and copies only committed positions. The session advances its position after the command succeeds. A matrix kernel with one thread per row and an eight-value accumulator was correct but made the 42-token prompt slower. A two-dimensional row-and-token grid used GPU cache locality for shared weights and measured 341–389 ms for the same prompt in five local probes. Three ten-request warm-server trials measured median ordinary throughput of 2.238 requests/s, compared with 1.068 before batching, and the output digest stayed the same. The split-stage path still uses one-token execution.

**Pattern:** Batch at the decoder-layer boundary, not by submitting several complete token commands together. For quantized matrix work on this Mac, give row/token pairs separate GPU threads and verify both numerical parity and serving throughput. Reserve key/value capacity before the batch so growth cannot copy unfinished GPU writes.

**Anti-pattern:** Assigning one GPU thread a row and a runtime-sized array of eight accumulators. It reduced parallel work and made even a one-token batch several times slower in local probes.
