# Device pooling plan

## Goal

Let an owner join two or more of their devices into one reliable inference service. Route whole requests to devices with enough capacity. Continue toward reusable conversation state and model splitting after the basic pool works.

## Layers

- [x] Give each device a stable public certificate and private key in a restricted local state directory.
- [x] Require an explicit fingerprint comparison before one device stores another as trusted.
- [x] Configure mutual TLS from the approved certificate list and test a live approved connection and rejected connections.
- [x] Serve peer traffic on a separately configured encrypted listener. Keep the local bearer-authenticated listener on loopback.
- [x] Define an authenticated peer capacity snapshot with model inventory, readiness, queue space, and whole-request forwarding.
- [x] Route complete requests to local or peer workers with bounded admission and clear overload responses.
- [x] Handle peer loss before and during responses, retry only before output escapes, and treat capacity snapshots as hints with destination-side admission.
- [x] Match the loaded GGUF file digest before routing or preserving a worker restart, so one model name cannot silently mix different weights.
- [x] Measure a co-located two-worker pool with a live model; exercise peer loss in end-to-end tests.
- [ ] Measure two physical Macs over the LAN with repeated throughput, latency, memory, and loss trials.
- [x] Reuse exact prompt prefixes across requests with device-owned conversation IDs and bounded checkpoint eviction.
- [x] Split model layers across approved peer services, transfer hidden activations, and verify output parity and partial-weight memory use on one Mac.
- [ ] Verify split output parity, per-device memory, and reliability on two physical devices.

## Current boundary

The peer listener uses mutually approved certificates and exposes the same bounded local model worker over an encrypted API. The local listener remains on loopback. A coordinator compares local backlog with fresh peer snapshots and forwards a whole request when a peer has the same model file digest and less work. Conversation IDs preserve device ownership for cached prompt prefixes. If an owner is lost, the full prompt can be rebuilt on another device with the same model file and receives a new ID. The destination worker enforces queue admission after a potentially stale snapshot. A co-located live-model pilot shows that two workers can share concurrent requests. Split execution, including Metal and mixed backends, passes one-host tests through approved peer services. Physical LAN deployment and measurements remain open. Peer trust changes take effect after a restart.
