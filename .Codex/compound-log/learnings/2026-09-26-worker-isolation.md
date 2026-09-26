# 2026-09-26 - Worker process isolation

The local server now starts the same executable in a private worker mode. The child loads and retains the GGUF model, then reads one request at a time from standard input and writes bounded generation events to standard output. The HTTP process owns authentication, tokenization, queueing, deadlines, and responses. It does not load the model weights. The child does not inherit the API key.

The supervisor retains the child between requests. If a request reaches its deadline, a client leaves during execution, the pipe breaks, or the child exits, the supervisor kills and reaps the child and starts another. It marks readiness unavailable during replacement. A failed model request with a valid protocol response does not require a restart.

The child protocol is private to the executable. A startup message gives the parent the model context length; request lines contain token IDs and generation length; response lines contain text, finish, or error events. The parent treats invalid messages as a broken worker and restarts it. This process boundary permits a hard stop of an unresponsive model calculation, which Rust threads alone cannot provide.

A real SmolLM2 Q4_K_M test sends both ordinary and streaming chat requests through separate CPU and Metal processes. A fake worker test blocks after accepting a request, verifies a deadline response, and verifies later requests succeed after replacement. It also verifies recovery after a child exits while idle. An in-flight request can fail when its child exits; later requests use the replacement rather than replaying possibly partial output.

Current limits: startup can take up to 30 seconds; repeated startup failures retry once per second and keep readiness unavailable. The worker accepts one request at a time. Restart latency and memory use under repeated failures still need measurement. Device pairing and cross-device routing remain separate layers.
