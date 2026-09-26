# Device pooling plan

## Goal

Let an owner join two or more of their devices into one reliable inference service. Route whole requests to devices with enough capacity. Continue toward reusable conversation state and model splitting after the basic pool works.

## Layers

- [x] Give each device a stable public certificate and private key in a restricted local state directory.
- [x] Require an explicit fingerprint comparison before one device stores another as trusted.
- [x] Configure mutual TLS from the approved certificate list and test a live approved connection and rejected connections.
- [x] Serve peer traffic on a separately configured encrypted listener. Keep the local bearer-authenticated listener on loopback.
- [ ] Define a small authenticated peer protocol for model inventory, capacity, readiness, and request forwarding.
- [ ] Route complete requests to local or peer workers with bounded admission and clear overload responses.
- [ ] Handle peer loss, retries only when no output has escaped, and stale readiness.
- [ ] Measure two-device behavior with live models and failure injection.
- [ ] Reuse conversation state across requests with explicit ownership and eviction.
- [ ] Explore model splitting after whole-request routing is correct and measured.

## Current boundary

The peer listener uses mutually approved certificates and exposes the same bounded local model worker over an encrypted API. The local listener remains on loopback. A coordinator that chooses and forwards requests to remote workers is the next layer. Peer removal takes effect when the listener restarts because the approved certificate list is loaded at startup.
