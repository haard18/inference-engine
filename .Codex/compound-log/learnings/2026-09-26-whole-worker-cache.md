## 2026-09-26 - Complete-worker cache admission

**Context**: The complete-model worker still offered the GGUF model's full context even when a single request's key/value cache could exceed the memory intended for serving on a small device.

**Learning**: The same growth-aware context formula applies to complete and partial model workers. A complete worker processes one request at a time, so it can enforce a per-request cache limit before it takes a conversation checkpoint or runs the prompt. Its restart contract must include that advertised limit as well as the model digest.

**Pattern**: Calculate cache bytes from layer count and key/value width, deduct score-buffer space, round the allowed context down to the cache growth boundary, and pass the result through the child startup response to API admission. Check the limit again in the child before execution.

**Anti-pattern**: Do not treat the model format's maximum context as a device memory promise. A key/value budget also does not cover model weights, temporary Metal buffers, or separately retained conversation checkpoints.
