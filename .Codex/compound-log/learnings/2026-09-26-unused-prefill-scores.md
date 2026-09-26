# 2026-09-26 - Do not project unused prompt-token scores

**Context:** The complete Metal session projected the full vocabulary after every prompt token, although only the final prompt token's scores are used to select the first generated token.

**Learning:** The session can run intermediate prompt tokens through the decoder without final normalization and vocabulary projection. The final prompt token still computes scores. On the 1.7B Q4_K_M model, real-model output and a conversation-reuse test remained correct. Three new ten-request warm-server trials measured median ordinary and streaming throughput of 1.068 requests/s, compared with 0.987 and 0.985 before the change. The trials were separate and short, so this is an observed gain for this prompt, not a stable general percentage. Most prompt cost remains: a local probe measured 878–980 ms for prompt evaluation and 141–143 ms for seven generation steps before the change.

**Pattern:** When a prompt supplies known tokens, compute next-token scores only after the last supplied token. Keep the cache position as the commit marker for each completed Metal step.

**Anti-pattern:** Doing a full vocabulary projection for every known prompt token and expecting a large serving gain from removing only that projection. Matrix work across decoder layers still repeats per token.
