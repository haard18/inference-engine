# Inference Engine

A Rust inference engine for language models on user-owned devices. The engine loads weights, runs model calculations, and generates tokens on one device. A server and a device pool will use the engine as their foundation.

## Architecture

- **Engine library:** Tensor storage, CPU and Metal matrix operations, a direct Llama-style model executor, a per-request KV cache, and token selection. A GGUF stage loader can now read a prefix or suffix of the decoder layers into separate CPU stage sessions.
- **Model input:** Safetensors with separate configuration and GGUF with Q8_0 or mixed Q4_K_M matrix weights. A project-owned byte-level BPE tokenizer reads the supported SmolLM2 layout from JSON or GGUF metadata.
- **Device backends:** A 32-bit CPU path checks numerical correctness. CPU execution keeps f16, bf16, Q8_0, Q5_0, and Q4_K matrix weights compact while accumulating in f32. On macOS, a reusable Metal runtime uploads matrix weights once and runs matrix operations and attention on the GPU. Each Metal session keeps its key/value history in growing Metal buffers.
- **Serving:** A per-device server exposes an authenticated, streaming chat API. It limits queues and memory use and reports overload clearly. The server keeps inference in a child process that it can replace after a timeout or process failure.
- **Device pool:** Each device has a stable certificate. An owner approves another device by checking its certificate fingerprint. Mutual TLS allows only approved certificates to use a peer API on the local worker. A coordinator checks each worker's available queue space and routes complete requests to a suitable peer when that peer has less work or the local worker is unavailable. Conversation IDs keep follow-ups on the device that owns their cached prompt. A separate split mode serves one chat request through a local prefix and an approved remote suffix.

The first real checkpoint target is SmolLM2-135M. We will carry one model family through the engine and serving layers before adding another family. llama.cpp is a reference and benchmark, not the implementation blueprint. We will measure performance and reliability before making comparative claims.

## Current milestone

The Rust library runs a Llama-style decoder on the CPU, with an optional Metal path for matrix operations and attention on Macs. It owns the tensor calculations, grouped-query attention, rotary positions, RMS normalization, feed-forward layers, per-request KV cache, greedy token selection, and a byte-level BPE tokenizer for the supported SmolLM2 layout. It loads Llama-style configuration and Safetensors weights in f32, f16, or bf16 format. It also loads the supported Llama-style GGUF layout with Q8_0 or mixed Q4_K_M matrices, interleaved rotary positions, and tokenizer metadata. Matrix weights retain their source precision in CPU memory, while activations and accumulations use f32. The command-line probes accept token IDs or text. A local server now offers an authenticated subset of the OpenAI chat completion API with ordinary and streaming responses.

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

To repeat the local worker-crash check after building the release binary, run `python3 scripts/worker-recovery.py /path/to/model.gguf --cycles 5`. Add `--metal` to exercise the Metal worker. The tool starts and stops its own loopback server, kills its child in each cycle, and reports readiness time, completed-chat time, and sampled process-tree memory. On one Mac with SmolLM2-135M Q4_K_M, five CPU crashes recovered readiness in 673–714 ms and three Metal crashes in 711–728 ms. Each cycle returned HTTP 503 during restart, then completed a real chat. These short idle-crash runs do not measure recovery under concurrent load.

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

End-to-end tests cover an approved request, an unapproved rejection, a streaming peer response, automatic routing to an idle second device, local worker loss, peer loss, and truncated peer responses. A conversation ID pins follow-ups to its owning device while it is available. If that device is lost, the coordinator recomputes from the supplied full history on another device and returns a new ID. Split serving works in a one-host test; performance on two physical devices remains unmeasured.

To measure the local API with one or more approved workers, run the benchmark against its loopback address. The tool sends a fixed chat prompt, warms the endpoint twice, then reports successful requests, HTTP statuses, completion-token rate, latency percentiles, and the worker that handled each request. `REQUESTS` is the total request count; `CONCURRENCY` is the maximum number in flight.

```sh
INFERENCE_API_KEY="$INFERENCE_API_KEY" cargo run --release --bin pool-bench -- \
  127.0.0.1:8080 local-smollm2 12 2 16
```

Append `--stream` to measure the time until the first nonempty text delta reaches the client. This is **first visible content**, which can occur later than the model's first generated token when decoding buffers incomplete text. Streaming output has no exact completion-token count, so the benchmark reports token throughput only in ordinary mode. It counts a stream as successful only when it sees both a finish reason and `[DONE]`; a stream error after HTTP 200 is counted separately. To sample resident memory during a run, use `python3 scripts/process-rss.py --seconds 25 --root gateway=PID --root worker=PID` in another terminal. The script reports the highest sampled RSS for each root process and its descendants; short spikes between samples can be missed.

The benchmark also reports a SHA-256 digest of each completed response's text. Ordinary and streaming responses for the same prompt should have the same digest. Its latency percentiles use completed requests only; status and error counts report failures separately. For a two-Mac comparison, save at least three benchmark JSON reports for each of complete serving, whole-request pooling, and split serving. Use the same model, request count, concurrency, and completion limit in every run. Then run `python3 scripts/lan-compare.py --whole whole-1.json whole-2.json whole-3.json --pool pool-1.json pool-2.json pool-3.json --split split-1.json split-2.json split-3.json`. The comparison rejects failed runs, differing output text, mismatched load settings, and a concurrency-two pool that used fewer than two devices. It summarizes median rates and latencies. Run a separate comparison for streaming mode, and record process-tree memory on each physical Mac while each scenario runs. The report checks benchmark data; keep the server startup records to prove which machines and serving modes produced it.

In one pilot on a single Apple Silicon Mac, two separate Q4_K_M worker processes served 12 requests at concurrency two in 9.92 seconds, versus 19.59 seconds for one worker. The measured rates were 1.210 and 0.613 successful requests per second; each paired worker handled six requests. All 24 requests returned HTTP 200. This is a co-located throughput check, not a result for two physical Macs. The sample size is small, and a physical network run with repeated trials remains open.

The current decoder is a correctness baseline. It uses f32 activations, greedy token selection, and one supported model and tokenizer layout. The loader validates the Safetensors header and reads tensor payloads one at a time. On one Apple Silicon Mac, the same one-token text prompt used 120 MB peak resident memory with Q4_K_M GGUF, 162 MB with Q8_0 GGUF, and 358 MB with the bf16 Safetensors checkpoint. The Metal runtime still keeps CPU weights as well as uploaded GPU weights. It now keeps the hidden vector on the GPU across all decoder layers and final projection, with one command completion and score readback per token. Three short local eight-token Q4_K_M probes took 0.28–0.29 seconds with Metal; two warm CPU runs took about 0.58 seconds. Separate maximum resident-memory samples were 233.4 MB and 122.0 MB. These observations do not establish serving throughput or a stable speed ratio. Quantization can change model scores and output quality. The initial embedding row and rotary table still enter from the CPU, and token selection still reads scores on the CPU.

A local split-stage test loads layers 0–14 and 15–29 of the Q4_K_M model separately. The complete model stores 99,230,976 bytes of weight values; the prefix stores 59,346,432 and the suffix stores 59,348,736. Both stages need the tied token embedding matrix, which limits memory savings. The two stage sessions produced the same scores as the complete model across four token positions. These counts cover stored weight values, not total process memory. Execution across two physical devices still needs validation.

The stage boundary uses a versioned binary activation frame. For this model, it carries 2,304 bytes of f32 hidden values plus a 64-byte header with the model digest, request ID, and token position. The suffix checks those fields, the frame length, and finite values before it runs. Local and separate-process parity tests both pass encoded frames between stages.

A separate process test now starts one worker for each 15-layer stage and passes activation frames between them through a parent process. Both workers retain their assigned weights and at most eight request sessions, with a 128 MiB total key/value cache limit and five-minute idle cleanup. The split scores matched the complete model at four token positions. At startup on one Apple Silicon Mac, a single resident-memory snapshot showed 119.5 MiB for the complete model worker and 65.5 MiB for each stage worker. This is a single idle snapshot, not peak memory under load.

An approved device can now serve a suffix stage over mutual TLS. After pairing both devices, start the suffix endpoint on its device with the same state directory used for pairing:

```sh
cargo run --release --bin serve -- \
  --stage-suffix /path/to/device-state 0.0.0.0:8444 \
  /path/to/SmolLM2-135M-Q4_K_M.gguf 15 30
```

The endpoint starts a partial-weight child process. It reports stage capacity and accepts bounded activation frames only from approved certificates. A real-model test sent four activations over mutual TLS and got the same scores as the complete model. It also rejected an unapproved certificate and a wrong request ID. Each remote step has a remaining-deadline header; a timed-out or canceled step discards its child process before another request can use it.

On the prefix device, use the approved suffix device ID and the same GGUF model file. The saved address for that suffix device must point to its stage service port. Set a local API key with at least 32 visible ASCII characters:

```sh
INFERENCE_API_KEY='replace-with-a-long-random-secret' \
  cargo run --release --bin serve -- \
  --split-prefix /path/to/device-state SUFFIX_DEVICE_ID \
  /path/to/SmolLM2-135M-Q4_K_M.gguf 15 8080
```

The loopback API then accepts the same `/v1/chat/completions` requests as a complete worker, including streaming. It checks model identity and stage capacity before each request, uses one request deadline for every token step, and ends generation if either stage fails. Suffix overload produces HTTP 429 for an ordinary response or a `queue_full` stream error. It can keep a prompt checkpoint on both stages for a returned conversation ID. A follow-up request must supply the full message history; only an exact token prefix can reuse the checkpoint. Both stage processes must confirm the saved position before reuse. A missing or restarted stage causes full prompt evaluation. The gateway keeps at most eight checkpoints for five minutes, and stage processes keep at most eight bounded sessions each. Active stage sessions cannot be evicted for a new one; an older checkpoint can be. This mode currently uses CPU stage execution. An unreachable suffix may retain state until its idle timeout. One-host tests confirm response parity, conversation reuse, checkpoint recovery, and suffix loss. Tests on two physical Macs and sustained performance measurements remain open.

In a one-host pilot with the Q4_K_M model, 12 ordinary requests and concurrency one gave 0.615 requests/s for the complete worker and 0.606 requests/s for a 15/15 split over loopback TLS. Median complete response times were 1,627 ms and 1,653 ms. Separate streaming runs measured median first visible content at 950 ms and 962 ms. With concurrency two, the complete worker and split served 0.614 and 0.603 requests/s; two complete workers with whole-request routing served 1.214 requests/s in one run and 1.039 in a second run. All these requests completed successfully. The short sample and uneven 7/5 routing in the second pool run prevent a stable throughput claim.

The split prefix probes its suffix once a second. Two failed probes make `/health` and new chat requests return HTTP 503; one successful probe restores admission. A full suffix queue does not count as an unhealthy device. The suffix answers capacity probes while a token step is running, so a busy worker does not delay the probe behind model computation. A stream already in progress still reports an inference error if the suffix disappears after output begins.

During the concurrency-two runs, `ps` sampling with a 100 ms sleep between samples found peak process-tree RSS of 155.5 MiB for the complete worker, 99.0 MiB for the split prefix device, 82.4 MiB for the split suffix device, and about 157 MiB for each complete worker in the pool. The actual sample spacing included `ps` execution time. These are sampled process-tree values on one Mac, including each server and its child process. They do not establish peak memory on two machines or performance over a real network. Repeatable two-Mac comparison remains the open acceptance check.
