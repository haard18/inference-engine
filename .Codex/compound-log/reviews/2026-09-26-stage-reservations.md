# 2026-09-26 - Reserve split-stage cache before work

## Change

Each split request now reserves cache capacity on the approved suffix and local prefix before either stage runs a prompt token. The reservation covers prompt tokens plus the requested maximum completion. The stage worker charges the cache capacity that its growable buffers may allocate, including power-of-two growth, against the shared cache budget. It rejects an over-budget reservation with an overload response. A token or activation without a reservation is rejected.

A close or expired active lease releases the reservation. A completed conversation rewinds both stages to the prompt checkpoint and is charged for the cache it actually retains. Saved checkpoints remain evictable when a new active request needs room. If admission fails before model execution, the supervisor closes any reservation that succeeded. A transport or server error during reservation now updates the peer failure cooldown; an ordinary overload does not.

## Evidence

- The real SmolLM2-1.7B-Instruct Q4_K_M suffix on one Mac admitted a 400-position request under its 128 MiB default budget, rejected a second concurrent 400-position reservation, then admitted that request after the first closed.
- Stage worker tests exercise the reservation requirement, per-session limit, session-count limit, and lease cleanup. Split serving tests cover context admission, ordinary and streaming generation, conversation reuse, and mixed CPU/Metal stages.
- A one-host repeated-serving check completed 40 requests: complete and split Metal serving, ordinary and streaming, 10 requests in each case, concurrency two, 16 generated tokens. The responses had one text digest and stable worker IDs. The loopback throughput was about 1.8 requests/s in each case; this does not predict two-Mac performance.
- All 73 release tests passed with the real-model tests included. Strict Clippy, formatting, and whitespace checks passed. Cargo reports the existing future-compatibility warning for the `block` 0.1.6 dependency.

## Remaining work

The second Mac is not set up, so physical two-device throughput, latency, memory, and link-loss recovery remain unmeasured. The broader engine goal remains active.
