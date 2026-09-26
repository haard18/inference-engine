# 2026-09-26 - Complete-worker cache admission review

The complete-model child now limits one active request's key/value cache to 512 MiB by default. `INFERENCE_WORKER_CACHE_MIB` accepts 16 to 8,192 MiB. The child advertises a context that fits its growable cache, and both the API and child reject longer requests before token work. A replacement child must report the same model digest and context limit before it serves requests. Split stages use the same budget calculation as before.

A real 135M Q4_K_M worker with a 16 MiB budget advertised 256 positions, rejected a request for 257, then completed a short request. The real 1.7B Q4_K_M worker advertised 1,024 positions at the 512 MiB default and 8,192 positions with a 4,096 MiB budget. All 74 release tests passed. After grouping the worker restart contract, the 26 library, four isolated-serving, and five paired-serving tests passed again. Strict Clippy passed.

The new budget covers one active request's key/value cache and score buffer. Saved conversation checkpoints have a separate 128 MiB cap. Model weights and temporary Metal allocations are outside this budget, so these values do not establish a complete process-memory bound. Physical two-Mac serving and loss measurements remain open.
