# 2026-09-26 - Peer return after loss

The existing paired serving test stopped a local worker, routed through an approved peer, and checked that stopping the peer caused HTTP 503. It did not check whether the same peer could serve again after returning.

The test now restarts that peer with the same device identity and network address. It retries through the original coordinator until the five-second failure cooldown expires, then checks HTTP 200, generated text, and the restarted peer's device ID in the conversation header. The real SmolLM2 Q4_K_M model test passed.

This checks recovery of one peer on a single host. It does not measure throughput, memory, network behavior, or recovery on two physical Macs. The second Mac is not currently visible through SSH service discovery on this network, so those physical checks remain open.
