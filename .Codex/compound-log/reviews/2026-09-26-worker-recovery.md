# Worker recovery review

The fault tool isolates its own server on loopback with a fresh temporary key and an unused port. It checks the direct child relationship twice before sending SIGKILL, and it verifies a replacement PID and a successful real-model chat in every cycle. Its startup and recovery waits are bounded; process-group cleanup runs even after failure. No API key is printed or passed on the command line.

The CPU and Metal runs each saw the expected unavailable-to-ready transition. The three canceled-stream cycles completed their follow-up requests. The small sample and idle failure point limit the result. Concurrent request failure, server-process failure, and two-device link loss still need separate measurements. The model child is killed intentionally; run this tool only in a test session that it starts itself.

Verification: the CPU and Metal fault runs completed, all 58 release tests passed with real-model tests included, Clippy passed with warnings denied, and the release build succeeded. Cargo still reports a future-compatibility warning from the existing `block` dependency.
