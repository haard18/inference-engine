# 2026-09-26 - Deadline-bound active stage sessions

The suffix worker keeps up to eight request sessions. A gateway normally closes a session after a request, but a gateway crash can leave the suffix with active sessions until its five-minute idle cleanup. Eight such sessions can prevent new requests even when its request queue is free.

Active stage commands now carry a bounded lease based on the request's remaining deadline. The prefix sends that lease to its child. The approved peer client sends the full remaining lease in a private HTTP header while keeping its per-step timeout capped at 120 seconds. The suffix validates the header and passes the lease to its child. On the next command after an active lease expires, the stage worker removes the abandoned session. Saved conversation checkpoints still use their separate five-minute idle limit. Older internal commands and approved peer requests without a lease keep the previous five-minute behavior.

A real-model stage worker test showed an active session disappearing after its lease while a rewound checkpoint remained available. It also rejected an invalid zero-length lease. The existing eight-session capacity test passed. A real Metal split chat request passed through the new header and child protocol. The default test suite, Clippy with warnings denied, and the build passed.

Cleanup is triggered by the next stage command, so an idle worker can retain expired allocations until then. A peer can request at most a ten-minute lease; the default serving request deadline is two minutes. This reduces the normal gateway-crash admission delay but does not make it instantaneous. Physical two-device failure behavior remains unverified because the second Mac is not set up.
