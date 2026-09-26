# Model splitting plan

## Goal

Run one model across two approved owner-owned devices when it does not fit on one device, without requiring both devices to load the full model. Keep whole-request routing for models that fit independently on each worker.

## Existing boundary

`Model::forward_token` currently embeds one token, executes every decoder layer, projects next-token scores, and commits all key/value cache entries. `GenerationSession` owns one cache for all layers. The GGUF loader reads every layer into one `ModelWeights`. These three boundaries must change together so a stage owns only its assigned weights and cache.

## Layers and acceptance checks

1. Extract a layer-range executor and split the token step into embedding, layers, and output projection. Prove that a local two-stage run gives the same scores, selected tokens, and cache positions as the whole-model path for a fixed fixture and the real Q4_K_M model.
2. Add a GGUF stage loader that reads only the needed layer range and its endpoint weights. The prefix stage needs token embeddings; the suffix stage needs final norm and output projection. For tied embeddings, the suffix also needs the embedding matrix for output projection. Check actual stored-weight bytes for each stage and reject incompatible stage ranges or model identities.
3. Add separate, bounded key/value state per stage and a versioned activation message containing model identity, request identity, token position, and `hidden_size` f32 values. Validate lengths and finite values before executing a stage. SmolLM2-135M has 576 hidden values, so a raw f32 activation is 2,304 bytes per token at one boundary, before protocol overhead.
4. Run each stage on its assigned device through the existing approved peer channel. Keep per-request stage placement fixed; stop generation clearly if either stage or the link fails after output begins. Admit only when both stages have capacity. Keep the full request deadline and bounded memory rules.
5. Check real-model parity, stage memory, output rate, first-token latency, and injected peer loss. Compare the split with one-device execution and whole-request pooling on two physical Macs. Model splitting succeeds when both workers hold less than the full model's weights where the format permits and the combined run produces the same greedy tokens within the established score tolerance.

## Risks to measure

- Tied embeddings limit memory savings because the prefix and suffix both need that matrix.
- A network round trip per token may make splitting slower than a single capable worker. Prefill batching can reduce message overhead later, but correctness comes first.
- A failed stage cannot be silently retried after tokens have streamed to a caller.

## Progress

- [x] Extract a layer-slice executor and isolate embedding and output projection. A local 15/15 layer boundary produced exactly the same next-token scores as the full path on the real Q4_K_M model. A two-layer fixture also checked multiple token positions and confirmed that a failed step does not commit any key/value state.
- [x] Load and own only each stage's GGUF weights and key/value cache. A local real-model test checks score parity and stored-weight reduction for each stage. Cross-device operation and process resident-memory measurements remain open.
- [x] Define a bounded, versioned binary activation frame and validate model, request, position, width, and finite values before suffix execution. The real-model stage parity test now passes encoded activations through this boundary.
- [x] Run prefix and suffix stages in independent bounded worker processes and check real-model parity through their binary activation exchange on one host. A single idle resident-memory snapshot confirmed that each stage process used less memory than the full worker.
- [x] Carry a suffix activation over mutually approved TLS to a partial-weight child process. Verify exact real-model scores, certificate rejection, request identity, and peer loss. Bound the peer queue and per-step deadline; discard a canceled child before reuse.
- [x] Orchestrate a full chat request across a local prefix and remote suffix. The prefix API checks suffix identity and capacity, then runs greedy generation across both stages with the original request deadline. Ordinary and streaming real-model responses match a complete worker. A test drops the suffix after streamed text and verifies a terminal error. Both stage sessions close or rewind after a request; the prefix child is replaced after an interrupted step. Split conversation IDs reuse exact prompt checkpoints when both stages still hold them, and recompute from full history after a cache miss or suffix restart.
- [x] Record a one-host pilot for complete, split, and whole-request pool serving. The split was close to the complete worker on this short prompt, with lower sampled resident memory on each stage device. A bounded streaming benchmark now measures first visible content and detects stream errors. These results do not substitute for a real LAN comparison.
- [x] Probe the suffix during split serving so the prefix health route and new-request admission reflect suffix loss and recovery. Two failed probes mark the suffix unavailable; a successful probe restores readiness. The existing request still reports a terminal error if a stream loses its suffix after output begins.
- [x] Batch consecutive prompt activations over one peer request, up to 16 frames and 4 MiB. A real-model peer test checks final-score parity and rejects an out-of-order batch without keeping partial state. A longer split chat still matches the complete model. Generated tokens continue to use one request per token.
- [ ] Measure split serving on two physical Macs under sustained load and compare it with complete-model routing and one-device execution.
