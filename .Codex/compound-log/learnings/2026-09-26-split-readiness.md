# 2026-09-26 - Split serving readiness

Split serving originally checked the suffix only when the prefix started or accepted a job. If the suffix disappeared while idle, the prefix still reported HTTP 200 from `/health`. A new request then failed during execution. This made the health route unreliable for a device acting as a serving hub.

The prefix now probes the approved suffix once a second. It checks the suffix worker readiness and the same model digest, adjacent layer range, hidden width, vocabulary size, and context length used at startup. Two failed probes mark the split service unavailable. A successful probe restores readiness. A full suffix queue remains a capacity condition rather than a health failure. The suffix flag is separate from the local child flag, so recovery of one stage cannot conceal failure of the other.

The real-model integration test stops and restarts the suffix. It confirms HTTP 503 for health and new requests during loss, HTTP 200 after recovery, and a cache miss after the suffix restarts. It also confirms that a stream which already emitted text ends with an inference error when the suffix is lost.

Readiness is sampled, not instantaneous. A loss can still admit a request before the second failed probe. The request path retains its own suffix check and reports the execution failure. The probe uses a weak status reference so it ends when the split service is dropped. A later [capacity probe change](2026-09-26-split-capacity-readiness.md) removed the worker-lock delay during a busy suffix step; network and runtime delays can still make a probe time out.
