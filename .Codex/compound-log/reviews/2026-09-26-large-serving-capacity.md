# 2026-09-26 - Larger-model serving and cache admission

## Delivered

- Fixed the local load tool's health poll so a socket read timeout during worker startup can be retried until the readiness deadline.
- Ran complete and 12/12 split Metal serving on the official SmolLM2-1.7B-Instruct Q4_K_M file through separate worker processes on one Mac.
- Made each split worker advertise an effective context limit derived from its key/value cache budget. The prefix serves the smaller limit of the two stages, and each stage enforces its own limit before a token or activation batch.
- Added `INFERENCE_STAGE_CACHE_MIB` to set the per-stage cache budget from 16 to 8,192 MiB, with a 128 MiB default. Evict saved checkpoints before an active request fails for aggregate cache pressure.

## Evidence

- Three 20-request trials per scenario and mode completed all 240 requests at concurrency two and 16 generated tokens. Ordinary and streaming responses shared one text digest, and worker IDs did not change.
- Median requests/s: complete ordinary 1.829, split ordinary 1.776, complete streaming 1.827, split streaming 1.778. Median response p50: 1,094 ms complete ordinary and 1,126 ms split ordinary. Median first visible content: 973 ms complete and 997 ms split.
- The 1.7B 12-layer stage advertises 512 positions under the default 128 MiB cache budget and 8,192 positions under a 2,048 MiB budget. Both values were read from the actual Metal stage worker startup message. An opt-in real-model split test rejects a request beyond the default stage limit before generation and completes a short request afterward.
- The release test suite, strict Clippy, and formatting check passed. The real-model split admission test and the 135M Metal split serving regression passed.

## Limits and next action

- These were two worker processes on one physical Mac. Two-Mac LAN behavior remains unmeasured because the second Mac is not set up.
- Process resident-memory samples varied markedly between trials and do not include a dependable GPU allocation total. The rate and latency measurements are useful for this one-host run only.
- The budget is shared across all stage sessions. Under concurrent long contexts, active sessions can still exhaust the aggregate budget after admission. A reservation protocol or earlier aggregate-capacity check is a remaining reliability layer.
