# 2026-09-26 - Activation frame review

## Delivered

- A bounded version-one binary frame for one layer-boundary hidden vector.
- Decoder validation of model digest, request UUID, token position, hidden width, exact frame length, and finite f32 values.
- A suffix-session entry point that validates an encoded frame before model execution.

## Validation

A small fixture rejects corrupted headers, wrong request or position, truncated payloads, and non-finite values. The real Q4_K_M split-stage test encoded and decoded every hidden vector across four token positions and still matched full-model scores exactly. Invalid request IDs did not advance the suffix cache.

## Remaining work

The frame still crosses only an in-process boundary during the parity test. Independent stage processes, mutually approved peer transport, bounded per-request session storage, loss handling, and network performance measurement remain open.
