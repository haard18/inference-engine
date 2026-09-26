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
- [ ] Measure two-device behavior with live models and failure injection.
- [x] Reuse exact prompt prefixes across requests with device-owned conversation IDs and bounded checkpoint eviction.
- [ ] Explore model splitting after whole-request routing is correct and measured.

## Current boundary

The peer listener uses mutually approved certificates and exposes the same bounded local model worker over an encrypted API. The local listener remains on loopback. A coordinator compares local backlog with fresh peer snapshots and forwards a whole request when a compatible peer has less work. Conversation IDs preserve device ownership for cached prompt prefixes. If an owner is lost, the full prompt can be rebuilt on another device and receives a new ID. The destination worker enforces queue admission after a potentially stale snapshot. Multi-device performance measurements and model splitting remain open. Peer trust changes take effect after a restart.
