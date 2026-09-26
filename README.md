# Inference Engine

A Rust inference engine for language models on user-owned devices. The engine loads weights, runs model calculations, and generates tokens on one device. A server and a device pool will use the engine as their foundation.

## Architecture

- **Engine library:** Tensor storage, CPU and Metal matrix operations, a direct Llama-style model executor, a per-request KV cache, and token selection.
- **Model input:** Safetensors with separate configuration and GGUF with Q8_0 or mixed Q4_K_M matrix weights. A project-owned byte-level BPE tokenizer reads the supported SmolLM2 layout from JSON or GGUF metadata.
- **Device backends:** A 32-bit CPU path checks numerical correctness. CPU execution keeps f16, bf16, Q8_0, Q5_0, and Q4_K matrix weights compact while accumulating in f32. On macOS, a reusable Metal runtime uploads matrix weights once and runs matrix operations on the GPU; attention and the KV cache still run on the CPU.
- **Serving:** A per-device server exposes an authenticated, streaming chat API. It limits queues and memory use and reports overload clearly. The server keeps inference in a child process that it can replace after a timeout or process failure.
- **Device pool:** Each device has a stable certificate. An owner approves another device by checking its certificate fingerprint. Mutual TLS allows only approved certificates to use a peer API on the local worker. A coordinator checks each worker's available queue space and routes complete requests to a suitable peer when that peer has less work or the local worker is unavailable. Conversation IDs keep follow-ups on the device that owns their cached prompt. Model splitting is the next layer under development.

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

The response includes an `x-inference-conversation-id` header. Send that header on a follow-up request and include the full message history in `messages`. The worker reuses a cached prompt only when the new tokenized prompt starts with the cached prompt exactly. Changed history, an expired or evicted cache entry, or a worker restart causes a full recomputation. Ordinary responses report reused tokens in `usage.prompt_tokens_details.cached_tokens`. Each worker retains at most eight conversation checkpoints and 128 MiB of checkpoint data; entries are removed after five minutes when the cache is next accessed.

The API supports string chat messages, one choice, and greedy decoding. It rejects other settings explicitly. One child process runs model calculations, up to four requests wait in the queue, and a full queue returns HTTP 429. Accepted requests have a 120-second deadline. An ordinary request returns HTTP 504 if it expires; a stream reports the timeout and closes. The server kills and replaces a child process that hangs or exits. While the worker is unavailable, health and new chat requests return HTTP 503. The server limits request bodies to 64 KiB and completion length to 256 tokens. It requires a Bearer token for model and chat routes; the child process does not receive that secret. The health route is public.

## Pairing devices

On each device, create a private state directory and show its public pairing offer:

```sh
cargo run --release --bin device -- init /path/to/device-state
cargo run --release --bin device -- offer /path/to/device-state > device-offer.json
```

Copy one device's offer to the other. Check the **full certificate fingerprint** with its owner through a separate trusted channel, such as a direct call or an in-person comparison. Then approve that device and its reachable IP address and port:

```sh
cargo run --release --bin device -- trust /path/to/device-state \
  /path/to/other-device-offer.json 192.168.1.20:8443 VERIFIED_FULL_FINGERPRINT
cargo run --release --bin device -- peers /path/to/device-state
```

Repeat in the other direction so both devices trust each other. `device remove STATE_DIR DEVICE_ID` revokes a stored peer. Restart a running peer listener to apply that change. Private keys stay in the state directory; the offer contains only the public certificate. The state directory is restricted to its owner on Unix systems.

Start a peer listener only after at least one peer is approved:

```sh
cargo run --release --bin serve -- \
  --peer /path/to/device-state 0.0.0.0:8443 \
  /path/to/SmolLM2-135M-Q4_K_M.gguf 8080
# On macOS, put --metal before --peer to use the GPU matrix path.
```

The local API remains on `127.0.0.1:8080` and still requires the Bearer key. The peer API is on the explicitly chosen address and port. It requires a mutually approved device certificate instead of the Bearer key; requests from unapproved devices cannot reach the API. Both APIs share one bounded worker and its queue. A device with approved peers checks their model, readiness, and free queue slots when its own worker is busy or unavailable. It forwards a whole request to a compatible peer with less work. A failed peer request can fall back to the local worker before any response reaches the caller and within the original request deadline. Once streaming begins, the coordinator does not replay the request. If the peer stream breaks, it sends a `peer_stream_lost` event and closes the stream. Peer failures get a short cooldown so the server does not retry a lost device on every request. Restart the server after changing peer trust.

End-to-end tests cover an approved request, an unapproved rejection, a streaming peer response, automatic routing to an idle second device, local worker loss, peer loss, and truncated peer responses. A conversation ID pins follow-ups to its owning device while it is available. If that device is lost, the coordinator recomputes from the supplied full history on another device and returns a new ID. Model splitting and multi-device performance work remain open layers.

To measure the local API with one or more approved workers, run the benchmark against its loopback address. The tool sends a fixed chat prompt, warms the endpoint twice, then reports successful requests, HTTP statuses, completion-token rate, latency percentiles, and the worker that handled each request. `REQUESTS` is the total request count; `CONCURRENCY` is the maximum number in flight.

```sh
INFERENCE_API_KEY="$INFERENCE_API_KEY" cargo run --release --bin pool-bench -- \
  127.0.0.1:8080 local-smollm2 12 2 16
```

In one pilot on a single Apple Silicon Mac, two separate Q4_K_M worker processes served 12 requests at concurrency two in 9.92 seconds, versus 19.59 seconds for one worker. The measured rates were 1.210 and 0.613 successful requests per second; each paired worker handled six requests. All 24 requests returned HTTP 200. This is a co-located throughput check, not a result for two physical Macs. The sample size is small, and a physical network run with repeated trials remains open.

The current decoder is a correctness baseline. It uses f32 activations, greedy token selection, and one supported model and tokenizer layout. The loader validates the Safetensors header and reads tensor payloads one at a time. On one Apple Silicon Mac, the same one-token text prompt used 120 MB peak resident memory with Q4_K_M GGUF, 162 MB with Q8_0 GGUF, and 358 MB with the bf16 Safetensors checkpoint. The Metal runtime currently keeps CPU weights as well as uploaded GPU weights. One local eight-token Q4_K_M run took 1.20 seconds and 235 MB peak resident memory with Metal, compared with 0.78 seconds and 122 MB on the CPU. These are single local measurements; quantization can change model scores and output quality. GPU-resident attention and lower synchronization cost remain open layers.
