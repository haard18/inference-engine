# Complete-worker cache admission

## Need

The split stages advertise a context limit derived from their key/value cache budget. The complete-model child still advertises the model's full context, so one request can grow its attention cache well beyond a small device's intended serving budget.

## Approach

- Share the cache-budget and context-limit calculation with the split stages.
- Give the complete worker a configurable per-request key/value cache budget with a documented default. Advertise the resulting context limit to the HTTP service.
- Reject an over-limit child request before taking a saved checkpoint or running any tokens. Preserve the same limit across child restarts.

## Acceptance

- A real GGUF worker advertises a smaller context under a small cache budget, rejects a longer prompt before work, then serves a short request.
- Existing split context and serving behavior stays correct.
- Release tests, strict Clippy, formatting, and build pass.

The budget limits one active complete-model session. The existing saved-checkpoint cache has its own 128 MiB cap; model weights and temporary Metal buffers are outside this key/value budget.

## Progress

- [x] Share checked cache-budget parsing and growth-aware context calculation with split stages.
- [x] Advertise a configurable complete-worker limit, reject over-limit child requests before model work, and require replacement workers to keep the original limit.
- [x] Exercise the real 135M worker at 16 MiB: it advertised 256 positions, rejected a 257-position request, then completed a short request. The real 1.7B worker advertised 1,024 positions at the 512 MiB default and 8,192 at 4,096 MiB.
