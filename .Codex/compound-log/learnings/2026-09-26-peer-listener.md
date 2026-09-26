# 2026-09-26 - Encrypted peer listener

The server now has two separate entry points to one bounded model worker. The loopback entry point still requires a Bearer key. The optional peer entry point has the same model and chat routes, but its route container is private to a server object that always wraps it in mutual TLS. A peer must present a certificate from the approved list. The peer entry point does not require the loopback Bearer key.

The peer listener is bound only when the owner passes an explicit state directory and IP address and port. It refuses to start with an empty trust list. A running listener retains the trust list it loaded at startup, so the owner must restart it after adding or removing a peer. Both entry points share one queue, worker process, and readiness state. A peer request therefore cannot bypass the local capacity limit.

The first live network test stalled because a pre-bound standard TCP listener was handed to Tokio in blocking mode. The peer server now changes that socket to nonblocking mode before the async server accepts connections. A real-model integration test sends model and chat HTTP requests over mutual TLS, verifies the local route still rejects a missing Bearer key, and verifies an unapproved peer cannot receive a successful response. A second test launches the command-line server with a real model and checks the peer listener.

This creates a secure remote worker endpoint. It does not yet choose a device, forward requests, recover from peer loss, or reuse conversation state.
