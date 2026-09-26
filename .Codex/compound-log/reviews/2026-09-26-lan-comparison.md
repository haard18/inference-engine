# LAN comparison review

The stream parser updates the digest only from visible content deltas and returns a digest only after a finish reason and `[DONE]`. Failed or truncated streams cannot enter the completed-output digest map. Ordinary responses require a string completion and exact completion-token count before they count as successful. Latency percentiles exclude failed samples, while failures remain visible in separate counts.

The comparison script rejects incomplete or inconsistent reports before calculating cross-mode rates. Three Python tests cover a valid three-scenario comparison, output divergence, request failure, and a one-device pool. A real-model loopback run checked digest equality between ordinary and streaming modes and parsed the emitted report schema. All 59 Rust release tests passed with real-model tests included. Clippy passed with warnings denied, the release build succeeded, and format and whitespace checks passed. Cargo still reports the existing future-compatibility warning for `block` 0.1.6.

The tool cannot itself prove that a named scenario ran on two physical Macs. Startup records, the second Mac's process-tree memory sample, and injected link-loss observations remain required evidence.
