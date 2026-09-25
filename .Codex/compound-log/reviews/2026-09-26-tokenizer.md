# 2026-09-26 - Tokenizer review

## Delivered

- A Rust byte-level BPE tokenizer for the SmolLM2 tokenizer layout.
- The text probe uses the project-owned tokenizer; the external tokenizer is test-only.
- Explicit rejection of unsupported tokenizer configuration options.

## Validation

- Exact token-ID and decode parity against the reference tokenizer on fixed and mixed prompts.
- Real-model text probe, unit tests, format, and lint checks.

## Scope

This tokenizer supports the observed SmolLM2 layout. Other tokenizer normalizers, pre-tokenizers, decoders, and BPE options need separate implementations and validation before support is claimed.
