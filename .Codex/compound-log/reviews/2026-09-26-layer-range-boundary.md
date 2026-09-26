# 2026-09-26 - Layer-range boundary review

## Delivered

- A reusable decoder layer-slice executor plus explicit embedding and output projection steps.
- A two-range local execution path that preserves the full path's deferred cache commit.
- Fixed-fixture and real Q4_K_M tests for full versus split-range score parity.

## Validation

The fixed fixture matched the full path at four token positions and kept its cache unchanged after an injected backend failure. The real Q4_K_M model matched scores exactly across a 15/15 layer boundary at four token positions. Strict Clippy, formatting, and the full release suite with real models passed; 49 tests ran.

## Remaining work

This is an arithmetic boundary within one complete model. Separate stage-owned weights and caches, selective GGUF loading, activation transport, two-device operation, and memory measurements remain open.
