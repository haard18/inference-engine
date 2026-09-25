# 2026-09-26 - Per-tensor loading review

## Delivered

- A bounded Safetensors header reader that validates metadata and full file coverage.
- Seek-based reads of individual tensor payloads into owned matrix or vector storage.
- Explicit errors for invalid header length, invalid tensor ranges, missing tensors, and truncated payloads.

## Validation

- Malformed archive tests for overlapping ranges, incomplete data, and oversized headers.
- Full release suite with the real SmolLM2-135M checkpoint and tokenizer comparison.
- Peak memory measurement on the same one-token probe used for the previous baseline.
- Format and lint checks.

## Next layer

Quantized CPU execution and GGUF model input, followed by Metal and serving work. The current loader still has a temporary copy of each individual tensor during conversion.
