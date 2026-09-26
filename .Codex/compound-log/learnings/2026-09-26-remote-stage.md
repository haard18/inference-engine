# 2026-09-26 - Approved remote suffix stage

**Context:** Two partial-weight stage processes matched full-model scores on one host. The suffix needed a network path restricted to devices that the owner explicitly paired.

**Learning:** The existing mutual TLS certificate rules can protect a separate suffix-stage endpoint. An approved peer sends a bounded binary activation with the current request UUID and remaining deadline. The server checks a bounded queue, forwards the frame to its partial-weight child, and returns raw f32 scores. The client validates the expected score count and finite values. Four SmolLM2 Q4_K_M positions matched the full model exactly across this encrypted path. An unapproved certificate could not use the endpoint; a wrong request ID was rejected without advancing the correct session.

**Pattern:** Keep the stage child in the server's control, not in the HTTP handler. Before awaiting a step, take ownership of the child from shared state. Return it only after a complete response or a clean model-level rejection. Timeout, cancellation, or a broken pipe then drops and kills it, so the next request starts with a clean process. An idle supervisor checks and replaces a lost child. Keep the caller's remaining deadline through connection, queue wait, child execution, and response-body read.

**Anti-pattern:** Reusing a child after its handler was canceled. Its pipe might contain a late response for the previous request. Also, an authenticated stage endpoint alone does not make the chat API split-aware; request placement and generation still need orchestration.
