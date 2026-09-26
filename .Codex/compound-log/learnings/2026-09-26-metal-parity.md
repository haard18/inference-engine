# 2026-09-26 - Metal matrix execution

**Context:** The CPU engine already runs bf16 Safetensors, Q8_0 GGUF, and mixed Q4_K_M GGUF. The first Metal path sends their matrix operations to the Mac GPU while the model's attention and KV cache remain on the CPU.

**Learning:** A prepared Metal runtime can upload model weights once and share them across generation sessions. Q/K/V projections and gate/up projections use one GPU command each because they share an input vector. The command queue is protected by a lock for concurrent sessions. f32, f16, bf16, Q8_0, Q5_0, and Q4_K matrix formats each have a shader path.

**Evidence:** The tiny f32 and f16 models match CPU scores. On an Apple M5 Mac, full SmolLM2 bf16, Q8_0, and Q4_K_M score vectors all differed from CPU by less than `1e-3`, and greedy token choice matched. Two threads made sessions from the same prepared runtime and received the same scores. An eight-token Q4_K_M prompt produced the same text on CPU and Metal.

**Limit:** A single local eight-token Q4_K_M run took 1.20 seconds with Metal and 0.78 seconds on the CPU. Peak resident memory was about 235 MB with Metal and 122 MB on the CPU. The Metal path copies weights to GPU buffers while retaining the CPU model and waits for each matrix group to finish before CPU work continues. The measurement does not establish general speed or memory behavior. GPU-resident attention and KV state, fewer transfers, and scheduling remain necessary.

**Dependency note:** `metal` 0.33.0 pulled in `block` 0.1.6, which the current Rust compiler reports as future-incompatible. The build and tests pass today. Revisit the dependency before a production release.
