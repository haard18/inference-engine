# 2026-09-26 - Per-tensor Safetensors loading

**Context:** Compact bf16 matrix storage reduced retained model size, but `fs::read` still buffered the full checkpoint during startup.

**Learning:** The Safetensors crate exposes a validated `Metadata` type. Reading only the bounded header and comparing its declared data length with the file length lets the loader seek to each tensor and read one payload at a time without mapping a mutable external file. The crate's metadata validator checks tensor dimensions, sizes, and contiguous offsets.

**Evidence:** The real SmolLM2-135M numerical comparison still passed. On one Apple Silicon Mac, peak resident memory for the same one-token probe fell from 543,440,896 to 342,310,912 bytes. Earlier f32-expanded loading used 812,384,256 bytes. A malformed-header test rejects overlapping ranges, incomplete data, and a header beyond the format's size limit.

**Pattern:** Use the format library's validated metadata, then copy one required tensor into owned storage. Keep external file mutations as recoverable read or validation errors rather than introducing a memory-mapping safety assumption.
