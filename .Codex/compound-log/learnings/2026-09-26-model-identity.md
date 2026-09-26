# 2026-09-26 - Exact model identity in a pool

Whole-request routing originally compared the model ID string. Two devices could advertise the same name while loading different GGUF files. A conversation could then move between different weights, so the owner could receive different greedy answers from requests that appeared to use one model.

The child worker now reports a SHA-256 digest of its loaded GGUF file at startup. The parent publishes it in the approved peer capacity snapshot. The coordinator routes to a peer only when its digest matches the local worker's digest. A replacement child with a changed digest remains unavailable. The peer client rejects absent or malformed digests.

The paired real-model tests confirmed that equal Q4_K_M files still route and that a live Q8_0 peer with the same model name is rejected when the local Q4_K_M worker is unavailable. A fake child test confirmed that a restart which reports different weights does not restore readiness. This requires an exact file match, including quantization; compatible but distinct checkpoint files cannot share this route. The digest adds a full-file read to worker startup. The remaining two-Mac measurements must check startup cost and load behavior with the intended model size.
