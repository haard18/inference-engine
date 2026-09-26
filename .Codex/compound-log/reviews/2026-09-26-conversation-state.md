# 2026-09-26 - Conversation state review

## Delivered

- Opaque device-owned conversation IDs in request and response headers.
- Bounded child-process attention checkpoints with exact token-prefix validation and least-recently-used eviction.
- Follow-up routing to the cache owner and full-prompt migration when that owner is lost.
- Reused token counts in ordinary response usage.

## Validation

- Real-model requests verified repeated, extended, changed, and evicted prompt behavior.
- A real-model Metal request verified checkpoint reuse on the GPU matrix path.
- A two-device test verified sticky peer ownership and fallback to a new local owner after peer loss.
- All 47 release tests passed, including real-model CPU, Metal, and paired-device checks. Strict Clippy, formatting, and diff checks passed.

## Scope

Clients must send full history. Cache expiry is applied on access; the memory cap applies at insertion. A process restart clears all checkpoints. Conversation IDs are capabilities within the owner-controlled trust domain, not separate user accounts. Model splitting and multi-device performance measurements remain open.
