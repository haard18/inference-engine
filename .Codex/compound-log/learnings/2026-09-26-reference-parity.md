# 2026-09-26 - Real checkpoint parity

**Context:** The CPU executor first passed a fixed-weight fixture, then loaded SmolLM2-135M.

**Learning:** A small fixture proves the mechanics, but a real checkpoint also tests tensor names, layouts, grouped-query attention, rotary positions, weight types, and layer count. The first twelve next-token scores agreed with an independent NumPy implementation within `1e-3`, and both implementations selected token `30` for token IDs `[1, 2, 3]`.

**Pattern:** Keep the independent calculation outside the Rust library. Compare numerical scores before comparing generated text, because decoding and tokenization can hide model errors.

**Constraint:** The current loader expands bf16 weights to f32 and holds the full checkpoint in memory. This is useful for correctness and has a higher memory cost than the source checkpoint.
