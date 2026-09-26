# 2026-09-26 - Match the rendered prompt in warm serving comparisons

**Context:** The first warm-server comparison sent the same chat messages to this engine and llama.cpp, but the model's GGUF chat template inserted a default system message only in llama.cpp. The requests therefore used 21 and 42 prompt tokens.

**Learning:** An explicit system message from that template made both servers report 42 prompt tokens and return the same seven-token answer. In three ten-request trials at concurrency one, median ordinary throughput was 9.467 requests/s for llama.cpp Metal and 0.987 requests/s for this engine's Metal path. Median streaming throughput was 9.425 and 0.985 requests/s. Both used the same 1.7B Q4_K_M file, eight requested completion tokens, two warmups per trial, and no cross-request prompt reuse. The results are a short, one-Mac baseline, not a general speed ranking. They do not isolate the time spent in prompt evaluation.

**Pattern:** Record prompt-token counts and completed-text digests before comparing rates. Send an explicit system message when a reference template would otherwise add one. Save a digest of the system message in every benchmark report so trials with different message settings cannot be combined.

**Anti-pattern:** Treating identical chat message JSON as proof that two engines processed identical prompt tokens.
