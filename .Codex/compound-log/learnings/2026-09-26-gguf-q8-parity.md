# 2026-09-26 - GGUF Q8_0 parity

**Context:** The first quantized checkpoint is the ggml-org SmolLM2-135M Q8_0 GGUF file. It is based on the same model revision as the Safetensors checkpoint.

**Learning:** GGUF Llama query and key weights use a row permutation for interleaved rotary positions. Applying the Safetensors split-half rotary calculation to these weights would be wrong. The engine now chooses the rotary layout from its model configuration. Q8_0 stores a 16-bit scale and 32 signed values in each 34-byte block; the CPU matrix operation retains those blocks.

**Evidence:** The Rust GGUF scores match an independent NumPy Q8_0 implementation within `1e-3` for the first twelve scores after token IDs `[1, 2, 3]`, and both choose token `30`. The Q8_0 checkpoint changes several scores by more than one point relative to the bf16 checkpoint, so direct Q8_0 reference scores are the correct parity target. The two checkpoint formats generated the same first eight token IDs for one text prompt. The GGUF checkpoint SHA-256 matched its published value, `813966e2b5e5aba261f6f198849a13b815db3c77bf4e389e72aa1e0f35a0d455`.

**Memory:** On one Apple Silicon Mac, the same one-token text prompt used 166,035,456 bytes peak resident memory with Q8_0 GGUF and 358,400,000 bytes with bf16 Safetensors. This single local measurement does not establish throughput or output quality for other prompts.
