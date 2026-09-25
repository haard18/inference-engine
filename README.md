# Inference Engine

A Rust inference engine for language models on user-owned devices. The engine loads weights, runs model calculations, and generates tokens on one device. A server and a device pool will use the engine as their foundation.

## Architecture

- **Engine library:** Tensor storage, CPU operations, a direct Llama-style model executor, a per-request KV cache, and token selection.
- **Model input:** Safetensors and its separate configuration first. GGUF and quantized weights follow as additional input and execution layers. A project-owned byte-level BPE tokenizer handles the first real model.
- **Device backends:** A 32-bit CPU path checks numerical correctness. CPU execution can keep f16 or bf16 matrix weights compact while accumulating in f32. Metal and quantized execution are additional layers.
- **Serving:** A per-device server exposes an authenticated, streaming chat API. It limits queues and memory use and reports overload clearly.
- **Device pool:** An owner pairs devices explicitly. A coordinator routes whole requests to capable devices and handles device loss. Reusable conversation state and model splitting are additional layers.

The first real checkpoint target is SmolLM2-135M. We will carry one model family through the engine and serving layers before adding another family. llama.cpp is a reference and benchmark, not the implementation blueprint. We will measure performance and reliability before making comparative claims.

## Current milestone

The Rust library runs a Llama-style decoder on the CPU. It owns the tensor calculations, grouped-query attention, rotary positions, RMS normalization, feed-forward layers, per-request KV cache, greedy token selection, and a byte-level BPE tokenizer for the supported SmolLM2 layout. It loads Llama-style configuration and Safetensors weights in f32, f16, or bf16 format. Matrix weights retain their source precision in memory, while activations and accumulations use f32. The command-line probes accept token IDs or text.

The small fixed-weight fixture and the real SmolLM2-135M checkpoint each have an independent NumPy reference. The real-model test compares next-token scores within `1e-3` and checks the selected token. The checkpoint test is opt-in because it needs the model files.

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

The current decoder is a correctness baseline, not a production serving runtime. It uses f32 CPU calculations, greedy token selection, and one supported model and tokenizer layout. The loader validates the Safetensors header and reads tensor payloads one at a time. On one Apple Silicon Mac, a one-token SmolLM2 probe used 342 MB peak resident memory with compact bf16 weights and per-tensor loading. Earlier versions used 543 MB with the full checkpoint buffered and compact weights, and 812 MB with weights expanded to f32. These are local measurements, not device-wide guarantees. The next engine layers add quantized execution, GGUF, and Metal. Serving and device pooling build on this library.
