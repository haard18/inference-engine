# Split readiness review

The suffix monitor checks the same stage identity and shape as startup, and it treats queue exhaustion as overload rather than device failure. The separate remote status flag keeps local process restart and suffix recovery independent. Repeated probe failure reduces the chance that one slow capacity response causes a readiness change.

The integration test covers suffix shutdown, admission rejection, restart, readiness restoration, cache invalidation, and failure after streaming begins. The status can lag actual network state by probe timing and the 750 ms capacity request timeout. A later [capacity probe change](2026-09-26-split-capacity-readiness.md) removed the worker-lock delay during long-running suffix steps. A two-physical-Mac test under load is still needed to measure false readiness changes and recovery time on a real network.

Verification: 57 release tests passed with ignored real-model tests included. Clippy passed with warnings denied, and the release build succeeded. Cargo reported an existing future-compatibility warning from the `block` dependency; it did not report a warning in this change.
