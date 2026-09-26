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

## Next layers

- [ ] Measure throughput, latency, memory, and behavior under concurrent and disconnected clients.
- [ ] Add request deadlines and worker recovery so a hung or failed calculation does not stall the serving process.
- [ ] Add explicit owner-controlled pairing and encrypted transport before listening beyond loopback.
- [ ] Route whole requests across paired devices with health and loss handling.
- [ ] Reuse conversation state safely across requests and devices.
- [ ] Explore model splitting across devices after whole-request routing works.
- [ ] Improve the Metal path by keeping attention and KV state on the GPU and reducing synchronization.

## Limits

This API is an intentional subset of chat completions. It supports one model, one choice, greedy decoding, and text messages. It does not claim full OpenAI API compatibility. The CLI listens only on loopback because the current transport has no TLS or device pairing.
