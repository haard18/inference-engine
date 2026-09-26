# 2026-09-26 - Check cached conversations against full history

**Context:** Earlier real-model tests checked one repeated prompt, one extension, and one eviction. A serving hub also needs correct results as histories grow across several turns and a streamed response.

**Learning:** For a cached request, use a fresh full-history request as the correctness reference. Check the exact prior prompt length in `cached_tokens` when the history extends the saved token prefix. Then change the prefix or evict its entry and check that recomputation still matches the fresh result. A successful HTTP status alone does not prove that reused attention state produced the right answer.

**Evidence:** The Q4_K_M integration test ran six distinct conversations on each CPU and Metal worker. Each conversation grew through three ordinary follow-ups, one streamed request, and another ordinary follow-up. Cached choices matched fresh full-history choices. Every extended ordinary request reported the prior prompt length as cached tokens. The stream included a finish event and `[DONE]`, produced the same text as a fresh ordinary request, and left a reusable checkpoint for the next turn. Changed prompts returned zero cached tokens and matched fresh output on both backends. The CPU test also confirmed eviction with zero cached tokens and matching fresh output. Both ignored real-model tests passed.

**Pattern:** Carry the full message history in every follow-up, even when sending a conversation ID. Exact token-prefix comparison permits reuse; a changed prefix falls back to full evaluation. Use several independent conversations in the test so the bounded cache repeatedly admits and evicts sessions.

**Limit:** The test calls the API router with a real isolated child process. It does not add physical network delay, concurrent clients, or sustained memory sampling. Those are separate checks in the goal.
