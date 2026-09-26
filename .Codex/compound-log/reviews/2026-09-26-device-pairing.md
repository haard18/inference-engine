# 2026-09-26 - Device pairing review

## Delivered

- Stable local device identities and public offers, with private key file permissions on Unix.
- A command-line flow to create an identity, inspect an offer, approve a fingerprint and address, list peers, and revoke a peer.
- Client and server mutual TLS configurations rooted in explicitly approved certificates.

## Validation

- Identity persistence, tamper detection, exact fingerprint checks, revocation, and the command-line flow have automated tests.
- A live TLS socket test exchanged bytes between approved devices and rejected an unapproved client and a wrongly identified server.
- All 37 release tests passed, including real-model CPU and Metal tests. Strict Clippy, formatting, and diff checks passed.

## Scope

The local inference server does not yet expose the peer listener. The trust list is loaded when a TLS configuration is built. Pool routing, device-loss handling, conversation reuse, and model splitting remain open.
