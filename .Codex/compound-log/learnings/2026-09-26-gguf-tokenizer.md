# 2026-09-26 - Tokenization from GGUF

**Context:** The Q8_0 GGUF model initially needed a separate `tokenizer.json` file to turn text into token IDs. The supported SmolLM2 GGUF checkpoint includes the vocabulary, ordered merge rules, and token types in metadata.

**Learning:** The checkpoint's `tokenizer.ggml.tokens` entries have exactly the same IDs as the JSON vocabulary, and `tokenizer.ggml.merges` has the same order as the JSON merge rules. Token type `3` identifies the 17 special tokens. The `gpt2` model with the `smollm` pre-tokenizer and disabled prefix-space/BOS options matches the project's existing byte-level BPE implementation. Other GGUF tokenizer layouts need their own validation and implementation.

**Evidence:** The GGUF tokenizer matched the JSON tokenizer and the reference `tokenizers` library on fixed prompts and 100 deterministic mixed-text prompts. The GGUF command generated text from a single model file. The loader rejects inconsistent token types, duplicate token strings, and unsupported tokenizer settings.
