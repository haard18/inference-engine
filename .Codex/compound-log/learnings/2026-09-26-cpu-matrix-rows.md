# 2026-09-26 - Parallel CPU output rows

**Context:** The quantized CPU decoder processed every matrix output row serially. The Q4_K_M eight-token probe took about 0.57 s per fresh process on the reference Mac.

**Learning:** Output rows are independent. A shared Rayon pool can distribute large matrix-vector operations without changing the arithmetic order within each row. The engine now uses parallel rows at 262,144 matrix elements and at least 64 outputs; smaller matrices remain serial. Two alternating five-run comparisons against the previous release binary measured medians of 0.569 versus 0.311 s and 0.579 versus 0.326 s. The completion digest was identical in every run. With `RAYON_NUM_THREADS=4`, the updated binary measured 0.313 s median; with one thread, it measured 0.595 s. The default worker count is selected by Rayon.

**Verification:** The large-matrix test crossed the parallel threshold and matched a serial calculation exactly. Release tests, Clippy, real Safetensors and GGUF reference tests, and CPU–Metal GGUF parity passed. A loopback API check completed 12 of 12 Q4_K_M requests at concurrency one and 16 requested output tokens. Two such runs measured 1.718 and 1.715 requests/s. One post-run process-tree snapshot was 161,424 KiB. The API run lacks a paired old-server binary, and the memory snapshot is not a peak value.

**Pattern:** Parallelize independent rows only when the matrix is large enough to cover scheduling cost. Compare the previous and new binary in alternating order with identical inputs, and check the output digest alongside latency. Leave thread count adjustable through Rayon's standard environment setting for devices with different CPU capacity.

**Open risk:** Multiple complete workers on one device each have a Rayon pool and can compete for CPU cores. The threshold and pool size are not tuned across device types or concurrent serving loads. A physical two-Mac serving comparison remains open.
