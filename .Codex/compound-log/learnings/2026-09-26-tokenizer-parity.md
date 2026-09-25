# 2026-09-26 - Tokenizer parity

**Context:** The first text probe used the `tokenizers` crate. The project now owns its SmolLM2 byte-level BPE implementation.

**Learning:** Tokenization depends on more than BPE merges. SmolLM2 isolates digits, applies a byte-level pattern, maps bytes into Unicode symbols, and recognizes special markers before merging. A tokenizer can produce fluent-looking text while still sending different token IDs to the model.

**Pattern:** Keep the reference tokenizer as a test-only dependency. Compare exact IDs and decoded text across spaces, Unicode, digits, special markers, and mixed inputs. Reject tokenizer JSON options that the local implementation does not support.
