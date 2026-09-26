# Batched Metal prompt evaluation

## Need

The corrected warm-server check serves the 1.7B Q4_K_M model at about one request per second, compared with about 9.5 requests per second for the pinned llama.cpp Metal server on the same short prompt. The current `GenerationSession::prefill` runs one complete Metal decoder command and final score projection per prompt token. The measurement does not prove how much time each part consumes, so first measure prompt and generation phases separately.

## Approach

1. Add a local phase-timing probe around real-model prompt evaluation and token generation. Keep timing out of the public chat response.
2. Avoid final score projection for intermediate prompt tokens, since their scores are never selected. Verify parity and measure the gain before adding a larger GPU path.
3. Build a bounded batched Metal prompt path. Process multiple known prompt tokens through each layer together, reuse matrix weights across those tokens, and apply causal attention to the growing key/value cache. Keep the existing one-token path for generation and as a fallback.
4. Check numerical scores and selected text against the existing path for f16 and supported GGUF quantization. Check cache growth, resumed conversations, and context limits.
5. Repeat the same warm-server comparison and record throughput, latency, prompt-token count, text digest, and memory. Keep the two-Mac acceptance run separate.

## Acceptance

- Full and split sessions return the same selected tokens and next-token scores within the established Metal tolerance after prompt batches.
- Failed or oversized batches do not advance the session position or expose partial cache state.
- The real 1.7B Q4_K_M warm-server benchmark is repeatable and improves on the recorded 0.987 ordinary requests/s baseline without changing output.
- Release tests, warnings-denied Clippy, formatting, and build checks pass.

## Progress

- [x] Record a matching-prompt warm-server baseline.
- [x] Measure prompt evaluation separately from generation for the 42-token 1.7B Q4_K_M prompt.
- [x] Avoid unused prompt-token score projections and verify real-model output and conversation reuse.
- [x] Test a four-token Metal command batch without shared matrix work. It kept real-model scores but made the 42-token prompt slower in five local runs, so the implementation was removed.
- [x] Add bounded eight-token batched prompt evaluation to complete Metal sessions. Check tiny, 135M Q4_K_M, and 1.7B Q4_K_M score parity against one-token execution, including cache growth.
- [x] Repeat the warm-server comparison after the projection change; ordinary throughput rose from 0.987 to 1.068 requests/s in separate short runs.
- [x] Repeat the warm-server comparison after batched prompt evaluation; median ordinary and streaming rates rose to 2.238 and 2.245 requests/s with the same digest.
- [x] Run four local Metal serving waves; all 80 requests completed with one stable worker.
- [ ] Extend the batch path to split Metal stages and verify parity and failure behavior.
- [ ] Measure sustained two-Mac serving and loss recovery when the second Mac is available.
