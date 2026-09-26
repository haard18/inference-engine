# 2026-09-26 - Selective model-stage loading

**Context:** The layer-range executor still held a complete model in one process. A split model must let each worker load only its assigned weights and maintain only its assigned attention state.

**Learning:** The GGUF loader can select the prefix or suffix layer tensors by index. The prefix needs token embeddings. The suffix needs final normalization and output projection; SmolLM2-135M ties output projection to token embeddings, so it needs that matrix too. For Q4_K_M, the full model stores 99,230,976 bytes of weight values. The 15-layer prefix stores 59,346,432 bytes and the 15-layer suffix stores 59,348,736 bytes. Each stage is about 60% of the full model by stored weight bytes; their sum exceeds the full model because the tied matrix is present in both.

**Pattern:** Give each stage its own weight object and key/value cache. Hash the exact GGUF file and require complementary ranges with the same digest before exchanging hidden activations. Read only selected tensor payloads; stream the digest calculation without loading the whole file into memory. Reject wrong activation widths, non-finite values, and calls made to the wrong stage role before advancing its cache.

**Anti-pattern:** Interpreting stored weight bytes as process resident memory, or treating local two-stage parity as a distributed serving result. Network transfer, stage placement, request admission, and failure handling remain open.
