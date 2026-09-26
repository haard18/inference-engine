# Inference Engine

A Rust inference engine for language models on user-owned devices. The engine loads weights, runs model calculations, and generates tokens on one device. A server and a device pool will use the engine as their foundation.

## Architecture

- **Engine library:** Tensor storage, CPU and Metal matrix operations, a direct Llama-style model executor, a per-request KV cache, and token selection.
- **Model input:** Safetensors with separate configuration and GGUF with Q8_0 or mixed Q4_K_M matrix weights. A project-owned byte-level BPE tokenizer reads the supported SmolLM2 layout from JSON or GGUF metadata.
- **Device backends:** A 32-bit CPU path checks numerical correctness. CPU execution keeps f16, bf16, Q8_0, Q5_0, and Q4_K matrix weights compact while accumulating in f32. On macOS, a reusable Metal runtime uploads matrix weights once and runs matrix operations on the GPU; attention and the KV cache still run on the CPU.
- **Serving:** A per-device server exposes an authenticated, streaming chat API. It limits queues and memory use and reports overload clearly.
- **Device pool:** An owner pairs devices explicitly. A coordinator routes whole requests to capable devices and handles device loss. Reusable conversation state and model splitting are additional layers.

The first real checkpoint target is SmolLM2-135M. We will carry one model family through the engine and serving layers before adding another family. llama.cpp is a reference and benchmark, not the implementation blueprint. We will measure performance and reliability before making comparative claims.

## Current milestone

The Rust library runs a Llama-style decoder on the CPU, with an optional Metal path for matrix operations on Macs. It owns the tensor calculations, grouped-query attention, rotary positions, RMS normalization, feed-forward layers, per-request KV cache, greedy token selection, and a byte-level BPE tokenizer for the supported SmolLM2 layout. It loads Llama-style configuration and Safetensors weights in f32, f16, or bf16 format. It also loads the supported Llama-style GGUF layout with Q8_0 or mixed Q4_K_M matrices, interleaved rotary positions, and tokenizer metadata. Matrix weights retain their source precision in CPU memory, while activations and accumulations use f32. The command-line probes accept token IDs or text. A local server now offers an authenticated subset of the OpenAI chat completion API with ordinary and streaming responses.

The small fixed-weight fixture, the real SmolLM2-135M Safetensors checkpoint, and its Q8_0 and Q4_K_M GGUF variants each have an independent NumPy reference. The real-model tests compare next-token scores within `1e-3` and check the selected token. The checkpoint tests are opt-in because they need the model files.

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

For GGUF, also download `SmolLM2-135M-Q8_0.gguf` and `SmolLM2-135M-Q4_K_M.gguf` from the [ggml-org SmolLM2-135M-GGUF repository](https://huggingface.co/ggml-org/SmolLM2-135M-GGUF/tree/main) into the same directory. The GGUF models and the Safetensors model were converted from the same source revision. Run:

```sh
SMOLLM2_DIR=/path/to/SmolLM2-135M cargo test --release --test gguf -- --ignored
SMOLLM2_DIR=/path/to/SmolLM2-135M cargo test --release --test tokenizer -- --ignored
cargo run --release --bin gguf-probe -- \
  /path/to/SmolLM2-135M/SmolLM2-135M-Q4_K_M.gguf \
  "The capital of France is" 8
# On macOS, add --metal before the model path to use the GPU matrix path.
```

To run the server on the same device, set a secret with at least 32 visible characters and start it with a GGUF model:

```sh
export INFERENCE_API_KEY="$(openssl rand -hex 32)"
cargo run --release --bin serve -- /path/to/SmolLM2-135M-Q4_K_M.gguf 8080
# On macOS, add --metal before the model path to use the GPU matrix path.
```

The server listens on `127.0.0.1` only. Send a chat request from another terminal:

```sh
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer $INFERENCE_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-smollm2","messages":[{"role":"user","content":"Say hello"}],"max_completion_tokens":32,"stream":true}'
```

The API supports string chat messages, one choice, and greedy decoding. It rejects other settings explicitly. One worker runs model calculations, up to four requests wait in the queue, and a full queue returns HTTP 429. The server limits request bodies to 64 KiB and completion length to 256 tokens. It requires a Bearer token for model and chat routes. The health route is public. This is a local serving layer; paired-device access, transport encryption, recovery after device loss, and model splitting are still planned layers.

The current decoder is a correctness baseline. It uses f32 activations, greedy token selection, and one supported model and tokenizer layout. The loader validates the Safetensors header and reads tensor payloads one at a time. On one Apple Silicon Mac, the same one-token text prompt used 120 MB peak resident memory with Q4_K_M GGUF, 162 MB with Q8_0 GGUF, and 358 MB with the bf16 Safetensors checkpoint. The Metal runtime currently keeps CPU weights as well as uploaded GPU weights. One local eight-token Q4_K_M run took 1.20 seconds and 235 MB peak resident memory with Metal, compared with 0.78 seconds and 122 MB on the CPU. These are single local measurements; quantization can change model scores and output quality. GPU-resident attention and lower synchronization cost remain open layers.
