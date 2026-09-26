# Per-device serving plan

## Goal

Turn the engine library into a reliable local serving process. Preserve the engine library as an independent layer. Keep the full multi-device inference goal active after this layer.

## Completed local layer

- [x] Load one supported GGUF model and tokenizer into a persistent worker.
- [x] Offer a loopback HTTP server with Bearer authentication for model and chat routes.
- [x] Support ordinary and streamed chat completions for string messages and greedy decoding.
- [x] Reject unsupported request options and oversized bodies with clear errors.
- [x] Bound the waiting queue and per-response channel; report a full queue with HTTP 429.
- [x] Stop producing output when a client disconnects and limit time spent waiting for a slow client.
- [x] Test the service against a real Q4_K_M model and smoke-test the CLI over HTTP.
- [x] Apply a deadline to queued and running requests, report HTTP 504 or a streaming timeout, and abandon expired queue entries.
- [x] Continue serving after an ordinary model execution error; catch per-request panics and rebuild Metal state if needed.
- [x] Report an overdue active worker as unavailable through health and new chat requests.
- [x] Keep the model in a child process and replace it after a hung or failed calculation.
- [x] Advertise a cache-safe context for the complete worker, reject oversized requests before model work, and require a replacement child to preserve that context contract.

## Integrated layers

- [x] Measure throughput, latency, and sampled memory under concurrent local load; verify that disconnected streams do not block later requests. A fresh two-wave 135M Q4_K_M Metal run completed 40/40 requests at concurrency two with one stable worker. The real-model disconnect test passed.
- [x] Measure restart latency and sampled process-tree memory under repeated worker failures on one host, for both CPU and Metal workers. Physical-device load trials remain open.
- [x] Add stable device identities, explicit owner-approved pairing, and verified mutual TLS configurations.
- [x] Connect the mutual TLS transport to a peer listener before listening beyond loopback.
- [x] Route whole requests across paired devices with health and loss handling; a real-model test passed for local and peer loss followed by recovery on one host.
- [x] Reuse exact prompt prefixes with device-owned conversation IDs and bounded eviction; CPU and Metal real-model conversation tests passed.
- [x] Run model splitting through approved peer services; CPU, Metal, and mixed-backend local tests match complete serving.
- [x] Keep attention and key/value state on the GPU and run known prompt tokens in bounded Metal batches.
- [ ] Measure sustained whole-request routing, split serving, and failure recovery on two physical Macs.

The Metal decoder keeps its intermediate hidden state and key/value history on the GPU through each token. Known prompt tokens can run in bounded batches. The CPU supplies token embeddings and rotary values and receives final scores for token choice. Physical two-Mac validation remains the open deployment check.

## Limits

This API is an intentional subset of chat completions. It supports one model, one choice, greedy decoding, and text messages. It does not claim full OpenAI API compatibility. The local listener is loopback-only; an optional second listener accepts mutually approved peers over TLS. Pool routing and split serving pass one-host checks; deployment over a physical LAN is unmeasured.
