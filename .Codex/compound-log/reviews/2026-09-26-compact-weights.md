# 2026-09-26 - Compact weights review

## Delivered

- Matrices that retain f16 or bf16 weights and compute f32 outputs.
- A loader that preserves 16-bit matrix weights from Safetensors.
- A stored-weight byte count for future device capacity reporting.

## Validation

- F16 and bf16 matrix operations and non-finite value rejection.
- Real SmolLM2-135M numerical comparison with the independent reference.
- One-token peak memory measurement before and after the change.
- Full release test suite, format, and lint checks.

## Remaining concern

Startup still reads the entire Safetensors file into a temporary buffer. Later work should reduce load-time memory without weakening file validation. Quantized weights, faster kernels, and Metal remain separate execution layers.
