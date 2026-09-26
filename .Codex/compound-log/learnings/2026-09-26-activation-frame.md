# 2026-09-26 - Activation frame boundary

**Context:** Separate model stages must exchange one hidden vector per token without mistaking an old request, a different model, or a malformed payload for the current step.

**Learning:** A fixed 64-byte header can carry a magic value, protocol version, exact GGUF digest, request UUID, token position, and hidden width. SmolLM2-135M has 576 f32 hidden values, so one encoded activation is 2,368 bytes. The decoder checks identity, position, width, length, and the 4 MiB bound before allocating the hidden vector. It rejects non-finite values before executing the suffix.

**Pattern:** Bind decoding to the suffix stage and its current cache position, and require the request ID from the surrounding session. Validate all header fields before copying hidden values. A failed validation leaves the suffix cache unchanged.

**Anti-pattern:** Treating a serialized vector as a trusted activation because it arrived from another process. The transport must still use approved peer authentication, and the stage worker must bind each UUID to one live request with bounded state.
