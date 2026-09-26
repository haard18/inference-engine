# Split conversation reuse

## Need

The complete-model worker keeps a bounded prompt checkpoint. Split serving currently accepts full history for each request but evaluates that history again on both stages. Reuse must be safe when either stage restarts or evicts its cache.

## Approach

Keep the public conversation ID owned by the prefix device. The prefix supervisor tracks at most eight checkpoints keyed by that ID: exact prompt token IDs, one private stage UUID, the prompt-end scores, and last use time. The two stage processes retain key/value state under the same private UUID. After generation, rewind both stage caches to the prompt-end position and save the gateway entry only if both stages acknowledge it. Before reuse, probe both stages for that UUID and exact position. If either no longer has the state, evaluate the full supplied prompt with a new UUID. Never infer cache validity from the ID alone.

Reuse only when the new tokenized prompt starts with the exact saved token sequence. Keep generated tokens out of the checkpoint, because sampled output can tokenize differently when later supplied as message text. The existing chat API reports the number of reused prompt tokens. A failure during generation discards both stage sessions. Eviction and five-minute idle expiry bound retained state; a stage may evict independently, so the next probe must tolerate a miss.

## Acceptance

- Repeated and extended full-history requests with the same conversation ID match a complete-model response and report exact cached token counts.
- A changed prefix, missing stage state, or stage restart causes a full prompt run with zero cached tokens.
- Two stages agree on the prompt position before reuse; a failed checkpoint is never published as reusable.
- State and protocol inputs are bounded; stage cache limits continue to apply.
- Ordinary and streaming responses keep their existing deadline and failure behavior.

## Result

- [x] Repeated, extended, and changed prompts report exact reused token counts in a real-model test.
- [x] A suffix process restart invalidates its checkpoint; the next request recomputes the full prompt.
- [x] Both stages acknowledge rewind before the gateway publishes a checkpoint.
- [x] Gateway entries are bounded to eight and 16 MiB with five-minute expiry. Each stage retains its existing eight-session and 128 MiB limits; it evicts only idle checkpoints when admitting a new session.
- [x] Ordinary response parity, streaming parity, reuse after a streamed request, and stage-loss behavior continue to pass.
