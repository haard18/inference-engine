# 2026-09-26 - Compact matrix weights

**Context:** The first loader expanded every bf16 matrix to f32. This doubled the retained matrix payload and raised peak process memory for SmolLM2-135M.

**Learning:** Matrix weights can remain as f16 or bf16 bits while CPU operations convert each value to f32 for accumulation. The real checkpoint still matched the independent NumPy scores within `1e-3`. On one Apple Silicon Mac, peak resident memory for a one-token probe fell from 812,384,256 to 543,440,896 bytes. The loader still reads the whole checkpoint file during startup, so peak memory is higher than the retained weight payload.

**Pattern:** Keep source precision explicit in storage and separate it from arithmetic precision. Report both numerical parity and observed memory usage before claiming an improvement.
