# Split capacity readiness review

The capacity route no longer waits behind the child process's token calculation. The readiness flag tracks the child through active work, restart, failed work, and cancellation. A request that drops a child cannot leave the flag ready because the child lease clears it on drop. Queue exhaustion remains a separate overload signal.

The focused test holds a fake stage step indefinitely, obtains a capacity response within 500 ms, checks `ready: true` and zero free queue slots, cancels the step, and confirms a later step succeeds with a replacement. The response timeout is below the prefix's 750 ms probe timeout.

All 59 release tests passed with real-model tests included. Clippy passed with warnings denied, the release build succeeded, and the diff had no whitespace errors. Cargo still reports its existing future-compatibility warning for `block` 0.1.6. A physical two-Mac test under load is still needed to measure response timing and recovery across a real network.
