# 2026-09-26 - GGUF tokenizer review

## Delivered

- A tokenizer loader for the supported SmolLM2 GGUF metadata.
- The GGUF text probe now needs only the GGUF model, prompt, and generation count.
- Shared byte-level BPE construction for JSON and GGUF sources.

## Validation

- Tokenization parity against JSON and the reference library on real checkpoint prompts.
- End-to-end text generation using only the GGUF checkpoint.
- Rejection checks for malformed vocabulary entries.
- Full release test suite, formatting, and lint checks.

## Scope

The GGUF tokenizer loader accepts the `gpt2` model with `smollm` preprocessing and the options verified in the SmolLM2 checkpoint. It rejects other token types and tokenizer layouts until they have their own parity tests.
