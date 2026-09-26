# 2026-09-26 - Separate model-stage processes

**Context:** The activation frame and partial model weights had only been exercised in one process. A real split needs independently loaded stages with bounded request state.

**Learning:** The existing server executable can run a private stage-worker mode. Each child loads one GGUF prefix or suffix and announces its model digest, layer range, hidden width, and stored-weight bytes. A parent can send a token to the prefix, relay its raw activation frame to the suffix, and read raw next-token scores. For four real Q4_K_M token positions, this two-process path exactly matched the complete model's scores. An invalid request ID produced a failure response without advancing the suffix. Eight live request sessions were admitted; a ninth was refused until one closed.

**Memory evidence:** In one idle snapshot on one Apple Silicon Mac, process resident memory was 119.5 MiB for the full worker and 65.5 MiB for each 15-layer stage worker. All three processes were alive during the sample. The stage workers each stored about 59.3 million weight bytes. These resident-memory numbers are one startup observation, not peak or sustained-load measurements.

**Pattern:** Keep model weights in the stage child, bind each request UUID to its own stage cache, limit live session count and total cache allocation, remove idle sessions, and drop a session after a failed stage step. A close command releases state when generation ends.

**Anti-pattern:** Treating a private parent-to-child pipe as the final device-pool transport. Approved peer authentication, remote admission, deadlines, and link-loss behavior still need to wrap the stage protocol.
