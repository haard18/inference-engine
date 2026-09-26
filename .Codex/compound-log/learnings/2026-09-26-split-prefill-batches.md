# 2026-09-26 - Bounded split prompt batches

The split prefix previously sent one peer request for every prompt token, even though the suffix's score is only needed after the last prompt token. This adds one network round trip per prompt token before any answer can be generated.

The prefix now collects consecutive activation frames in batches. Each batch contains at most 16 frames and 4 MiB. The suffix validates the count and exact body size, executes frames in order against one request ID, and returns only the final scores. The worker protocol defaults to one frame when an older caller omits the count. After prompt evaluation, generated tokens still cross the stage boundary one at a time so streaming and failure handling remain unchanged. A failed batch discards its partial suffix session; the prefix supervisor discards its child after an incomplete local step.

The real-model peer test sent four frames in one approved TLS request and matched the complete model's final scores. It sent an out-of-order two-frame batch, got a rejection, and found no session at the suffix. The split chat test used a prompt longer than one batch and matched complete-model text and token counts. The existing split streaming, conversation, suffix-loss, and restart checks also passed.

This removes avoidable peer requests during prompt evaluation. It does not yet measure speed over a physical LAN. The second Mac is not set up, and split stages still execute on CPU.
