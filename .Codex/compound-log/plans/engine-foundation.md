# Engine foundation plan

## Goal

Build the model execution core before device pooling. The core must produce correct inference results on one device and remain callable without a server.

## Decided architecture

- Rust library with a small command-line test program.
- Direct Llama-style executor with reusable tensor operations.
- CPU path first, then Metal. Validate 32-bit results before 16-bit and quantized execution.
- Safetensors plus model configuration first, then GGUF.
- A project-owned tokenizer, checked against an existing implementation for the first model.
- SmolLM2-135M as the first real checkpoint.
- A per-request generation session owns its KV cache and emits tokens incrementally.

## Foundation acceptance criteria

- [x] A tiny decoder model computes next-token scores without a server.
- [x] Fixed input and weights match an independent numerical reference within a documented tolerance.
- [x] Invalid tensor shapes, model values, and token IDs return explicit errors.
- [x] The CPU result gives a baseline for later model loading and Metal work.

## Real-model layer

- [x] Read SmolLM2-135M weights and configuration from Safetensors and JSON.
- [x] Validate tensor names, shapes, and numeric types.
- [x] Run a text prompt through an existing tokenizer, then a project-owned tokenizer.
- [x] Compare real-model next-token scores and token choice with an independent NumPy reference.

## Compact-weight layer

- [x] Retain f16 and bf16 matrix weights without expanding them to f32 at load time.
- [x] Preserve real-model numerical parity and measure the memory change.
- [x] Reduce load-time file buffering while preserving Safetensors validation.
- [x] Add Q8_0 block execution and load the supported GGUF model layout.
- [x] Read the supported SmolLM byte-level BPE tokenizer from GGUF metadata without a separate JSON file.
- [x] Run the SmolLM2 Q4_K_M GGUF checkpoint with Q5_0, Q8_0, and Q4_K matrices.
- [x] Run f32, f16, bf16, Q8_0, Q5_0, and Q4_K matrix operations on Metal and check real-model parity.
- [x] Make uploaded Metal weights reusable across generation sessions.
- [ ] Keep attention and KV state on the GPU and remove per-operation CPU/GPU synchronization.

Metal attention and per-session key/value buffers now run through GPU commands. Query/key/value projection, rotary positions, and attention share one command, so those three projection vectors no longer return to CPU. Normalization, activation, residual, and output boundaries still synchronize with CPU. The remaining checkbox tracks removal of those boundaries.

## Risks

- Floating-point operation order can cause small differences. Record tolerances and compare intermediate outputs when needed.
- A real checkpoint can use weight layouts or model options absent from the tiny fixture. Validate each option before expanding the executor.
