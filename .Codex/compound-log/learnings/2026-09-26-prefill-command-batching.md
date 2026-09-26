# 2026-09-26 - Command batching without matrix reuse did not help

**Context:** The Metal decoder used one command per prompt token. A trial encoded four known prompt tokens in one command, retained their temporary buffers until completion, and preallocated the key/value cache so growth could not copy unfinished GPU writes.

**Learning:** A small fixture and a real 135M Q4_K_M checkpoint produced the same scores as one-token execution, including across cache growth. On the 1.7B Q4_K_M 42-token prompt, five fresh-process trials measured a median prompt phase of about 1,069 ms with the trial batch path. Five earlier trials after removing unused score projections measured about 841 ms. These separate short runs also showed generation-time variation, so they do not isolate the exact cause of the difference. The batch path was removed because it had no demonstrated benefit. It still dispatched a separate matrix-vector operation for every token and retained many more temporary buffers in each command.

**Pattern:** Require a measured gain before keeping a larger Metal command. A useful prompt batch must reuse matrix weights across known tokens and preserve causal attention and the cache boundary.

**Anti-pattern:** Assuming fewer command submissions are enough to accelerate prompt evaluation when the same quantized weights are still read once per token.
