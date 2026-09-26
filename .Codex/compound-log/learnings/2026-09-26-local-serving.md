# 2026-09-26 - Local serving

The engine can now serve chat requests on one device without moving model calculations into HTTP handlers. One dedicated worker owns inference and receives jobs through a bounded channel. The server checks a Bearer token, validates supported request settings, and reports overload before accepting more work.

Two token handling details matter for correctness. User content is encoded as plain text, so a literal special-token spelling in a message cannot become a chat control token. The streaming decoder holds incomplete UTF-8 bytes until later tokens finish the character. When generation ends, the server returns any remaining bytes as replacement text rather than silently dropping them.

The generation session now selects a token separately from advancing the model state. This avoids an extra model calculation after the final requested token or a stop token. It also permits returning the final token when the model context is full.

The service test used the real SmolLM2 Q4_K_M model for authenticated requests, ordinary and streamed responses, unsupported settings, and queue saturation. A CLI request over HTTP produced SSE chunks and a final `[DONE]` event. The worker is still a single execution lane. Request deadlines, recovery, encrypted transport, and pooling are separate layers to implement and measure.
