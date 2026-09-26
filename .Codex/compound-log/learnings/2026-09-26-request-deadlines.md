# 2026-09-26 - Request deadlines

The first local server could leave an accepted request waiting indefinitely if the worker stopped making progress. Every accepted request now has one deadline shared by its queue wait and model execution. An ordinary request returns HTTP 504 after that deadline. A streaming request sends a timeout error event and closes. Dropping the response receiver lets the worker skip expired queued work; the worker also checks the deadline before prefill, after prefill, and between generated tokens.

The streaming bridge bounds each channel send by the same deadline. Without that bound, a client that stopped reading could hold a forwarding task even after the model worker stopped sending. Terminal events now end the stream once, followed by `[DONE]`. Both response paths also check the clock explicitly while consuming events. Tokio can return an already-ready future even when its timeout deadline has passed, so a timer wrapper alone is not enough for a strict event-draining deadline.

An execution error does not kill the worker. A real-model test sends an invalid token job, observes its failure, and then completes a valid chat request. Per-job panics are caught, including session creation. If a Metal request panics, the worker rebuilds its GPU runtime before taking more jobs; failure to rebuild closes the worker so readiness reports unavailable.

The worker records the current job deadline while it is active. When a calculation remains active after that deadline, health and new chat requests report HTTP 503. They return to normal when the worker finishes and clears its active state. This reports lost capacity instead of showing a healthy server that only times out requests.

An in-process thread cannot be killed safely when a model calculation never returns. The HTTP deadline limits client wait but cannot free that thread. A restartable worker process remains necessary for full recovery.
