# 2026-09-26 - Metal split serving through child processes

The previous layer proved that two in-process Metal stages matched a complete Metal model. The split serving path still spawned CPU stage children, so the GPU stages were not reachable through the chat API.

Stage workers now accept a backend choice. A Metal worker prepares one stage runtime and uses it for each bounded request session. The prefix and approved suffix services pass their selected backend to child startup and child replacement. The `serve` command accepts `--metal` before `--split-prefix` or `--stage-suffix` on macOS. CPU remains the default, and each device chooses its own backend.

The real Q4_K_M SmolLM2 Metal split test ran both stages as child processes through an approved loopback TLS connection. Ordinary chat choices and token usage matched a complete Metal worker. Streaming returned the same text and a completion marker. Repeating a conversation with its ID reused the full prompt checkpoint. The existing CPU split test, including suffix loss and recovery, passed again. `cargo clippy --all-targets -- -D warnings`, `cargo build`, and `cargo test` passed.

This validates process and protocol integration on one Mac. It does not establish split throughput or memory under sustained load. A second Mac is not set up, so cross-device latency, failure behavior, and performance remain unverified.
