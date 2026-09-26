# Batched Metal prompt evaluation review

## Correctness

The new complete-model Metal prefill path runs at most eight known tokens through each layer. The attention score and reduction kernels use `first_position + token + 1` as each token's causal limit. Key/value capacity is grown before the command starts and copies only positions that the session has committed. A batch updates the session position only after Metal reports a completed command and the returned scores are finite. Tests compare batched and one-token scores on tiny f32, 135M Q4_K_M, and 1.7B Q4_K_M models. The tiny test reaches the configured context limit. A split-serving test confirms that the unchanged stage path still matches the faster complete path.

## Quality and security

The one-token path remains for generated tokens and split stages. The batch path uses the same uploaded model weights and isolated per-request cache; it does not add a network endpoint or change authentication. The scalar and batched matrix shaders contain parallel format-decoding branches, which are a maintenance risk: changes to Q8_0, Q5_0, Q4_K, or Q6_K decoding must update both and rerun the seven-format matrix test. The batch size is bounded at eight in Rust and in the shader.

## Evidence and limits

The release suite, warnings-denied Clippy, formatting, real-model parity, cached-conversation, and split-serving checks passed. Three ten-request warm-server trials per response mode used the same 42-token prompt and answer digest as the earlier baseline. Median ordinary throughput rose from 1.068 to 2.238 requests/s; median streaming throughput rose from 1.068 to 2.245 requests/s. Four local Metal serving waves completed 80/80 requests with one stable worker and a sampled process-tree peak of 268,720 KiB. These short one-Mac runs do not establish long-term memory stability, concurrent 1.7B throughput, two-Mac behavior, or parity for every supported model. Split stages still evaluate prompt tokens one at a time.
