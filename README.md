# Inference Engine

A Rust inference engine for language models on user-owned devices. The engine loads weights, runs model calculations, and generates tokens on one device. A server and a device pool will use the engine as their foundation.

## Architecture

- **Engine library:** Tensor storage, CPU operations, a direct Llama-style model executor, a per-request KV cache, and token selection.
- **Model input:** Safetensors with separate configuration and GGUF with Q8_0 matrix weights. A project-owned byte-level BPE tokenizer handles the first real model; the GGUF command currently reads the matching tokenizer JSON separately.
- **Device backends:** A 32-bit CPU path checks numerical correctness. CPU execution keeps f16, bf16, or Q8_0 matrix weights compact while accumulating in f32. Further quantized formats and Metal are additional layers.
- **Serving:** A per-device server exposes an authenticated, streaming chat API. It limits queues and memory use and reports overload clearly.
- **Device pool:** An owner pairs devices explicitly. A coordinator routes whole requests to capable devices and handles device loss. Reusable conversation state and model splitting are additional layers.

The first real checkpoint target is SmolLM2-135M. We will carry one model family through the engine and serving layers before adding another family. llama.cpp is a reference and benchmark, not the implementation blueprint. We will measure performance and reliability before making comparative claims.

## Current milestone

The Rust library runs a Llama-style decoder on the CPU. It owns the tensor calculations, grouped-query attention, rotary positions, RMS normalization, feed-forward layers, per-request KV cache, greedy token selection, and a byte-level BPE tokenizer for the supported SmolLM2 layout. It loads Llama-style configuration and Safetensors weights in f32, f16, or bf16 format. It also loads the supported Llama-style GGUF layout with Q8_0 matrices and interleaved rotary positions. Matrix weights retain their source precision in memory, while activations and accumulations use f32. The command-line probes accept token IDs or text.

The small fixed-weight fixture, the real SmolLM2-135M Safetensors checkpoint, and its Q8_0 GGUF variant each have an independent NumPy reference. The real-model tests compare next-token scores within `1e-3` and check the selected token. The checkpoint tests are opt-in because they need the model files.

Run the checks after installing Rust:

```sh
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

Download `config.json`, `model.safetensors`, and `tokenizer.json` from the [SmolLM2-135M checkpoint](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/tree/main) into one directory. Then run the model comparison and a text prompt:

```sh
SMOLLM2_DIR=/path/to/SmolLM2-135M cargo test --release --test real_model -- --ignored
SMOLLM2_DIR=/path/to/SmolLM2-135M cargo test --test tokenizer -- --ignored
cargo run --release --bin text-probe -- \
  /path/to/SmolLM2-135M/config.json \
  /path/to/SmolLM2-135M/model.safetensors \
  /path/to/SmolLM2-135M/tokenizer.json \
  "The capital of France is" 8
```

For GGUF, also download `SmolLM2-135M-Q8_0.gguf` from the [ggml-org SmolLM2-135M-GGUF repository](https://huggingface.co/ggml-org/SmolLM2-135M-GGUF/tree/main) into the same directory. The GGUF model and the Safetensors model were converted from the same source revision. Run:

```sh
SMOLLM2_DIR=/path/to/SmolLM2-135M cargo test --release --test gguf -- --ignored
cargo run --release --bin gguf-probe -- \
  /path/to/SmolLM2-135M/SmolLM2-135M-Q8_0.gguf \
  /path/to/SmolLM2-135M/tokenizer.json \
  "The capital of France is" 8
```

The current decoder is a correctness baseline, not a production serving runtime. It uses f32 CPU calculations, greedy token selection, and one supported model and tokenizer layout. The loader validates the Safetensors header and reads tensor payloads one at a time. On one Apple Silicon Mac, the same one-token text prompt used 166 MB peak resident memory with Q8_0 GGUF and 358 MB with the bf16 Safetensors checkpoint. These are local measurements; quantization can change model scores and output quality. The next engine layers add more quantized formats, GGUF-native tokenization, Metal, serving, and device pooling.
