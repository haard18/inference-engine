# 2026-09-26 - Peer listener review

## Delivered

- An optional mutual TLS listener for explicitly approved devices, separate from the loopback Bearer API.
- One shared model worker, queue, and health state behind both entry points.
- A command-line option for the owner to bind the peer listener to a chosen address and port.

## Validation

- A real-model network test gets model information and a chat response from an approved device without a Bearer key.
- The same test rejects an unapproved device and checks that the local model route still requires a Bearer key.
- A second real-model test launches the command-line server and reaches its encrypted peer listener.
- All 39 release tests passed, including CPU and Metal model checks. Strict Clippy passed.

## Scope

The listener loads peer trust at startup. Trust changes need a restart. This is a remote worker endpoint; coordinator routing, device-loss handling, conversation state reuse, and model splitting are still open.
