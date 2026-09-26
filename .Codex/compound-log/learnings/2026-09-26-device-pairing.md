# 2026-09-26 - Device identity and pairing

Each device now creates one P-256 key pair and self-signed certificate with a name derived from a stable device ID. The private key and certificate are stored in an owner-only state directory on Unix. The public pairing offer contains the certificate, device ID, and SHA-256 certificate fingerprint. Loading an identity checks its content digests, certificate signature, name, validity, and matching private key.

The owner approves a peer by supplying its offer, IP address and port, and the full fingerprint obtained through a separate trusted channel. The trust store verifies the exact fingerprint and rejects duplicate devices. Removing a peer updates the saved trust list.

The mutual TLS configuration uses only approved peer certificates as trust anchors. A live socket test showed two mutually approved devices exchanging data. A client with an unapproved certificate and a client trusting the wrong server both failed. This establishes the connection boundary for later peer HTTP routes. Revoking a peer in the saved list does not change an already running TLS configuration; a peer listener must reload trust or restart to apply revocation.

The current server still binds to loopback. A peer listener and inference routing remain the next layers. Concurrent edits to the same trust file from separate processes also need serialization before pairing commands are used concurrently.
