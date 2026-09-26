# Split conversation reuse review

Correctness: The gateway verifies exact prompt tokens and both stage positions. It saves a checkpoint only after both rewind acknowledgments. A restart or eviction produces a full prompt run. Real-model tests cover repeated, extended, changed, and restarted-suffix cases.

Capacity: Gateway entries have count, byte, and age limits. Stage caches keep their existing count, byte, and age limits. New sessions can evict saved checkpoints but cannot evict active sessions. The suffix makes its final bounded admission decision on each activation.

Overload: A full suffix queue is reported as HTTP 429 in ordinary chat or a `queue_full` event in streaming chat. This keeps capacity rejection distinct from stage execution failure.

Recovery: A preflight rejection leaves the prefix child and its other checkpoints in place. A request that has entered the prefix pipe still discards that child on failure, which prevents reuse of uncertain local state. This is conservative and can lose unrelated cached prompts after a remote failure; the next request recomputes from full history.

Security: Probe and rewind use the existing approved peer TLS service and unguessable stage UUIDs. The endpoints validate request IDs, deadlines, and checkpoint positions. No API key or model weights are sent to the remote stage.

Remaining measurement: Two physical Macs, sustained throughput, first-token latency, resident memory under load, and repeated link-loss recovery are still unverified.
