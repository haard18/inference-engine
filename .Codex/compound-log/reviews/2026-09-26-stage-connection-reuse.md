# 2026-09-26 - Split activation connection review

The approved peer client now keeps one mutual TLS and HTTP/1 connection for sequential suffix activations. It holds the sender lock through the bounded response body, checks for a closed connection before sending, and discards it after a send or body failure. It does not replay an activation with an unknown commit outcome. Capacity snapshots and whole-request forwarding keep their independent connections.

The real 135M stage parity test used a transparent TCP proxy to count connections. Multiple rejected and valid activations used one TLS connection and returned the same scores as a complete model. After suffix loss, the client reported an activation failure; after suffix restart, a new request reconnected and again matched complete-model scores. The streamed-loss regression initially failed because aborting the server's accept task left established connections alive. It now shuts down active connections and passed its targeted rerun. The real 1.7B one-Mac Metal smoke run completed 40 ordinary and streaming requests across complete and split serving with matching text and stable workers. In that single loopback trial, split throughput was 1.774 ordinary and 1.787 streaming requests/s, close to the prior run; no speed gain is established.

All 74 release tests passed after correcting the shutdown test. Strict Clippy, formatting, whitespace checks, and the release binary build passed. Cargo still reports its existing future-compatibility warning for `block` 0.1.6.

This change adds no new listener or trust path. It retains the approved peer certificate check on connection creation. A real two-Mac trial is still needed to measure network latency, repeated link loss, and any throughput benefit.
