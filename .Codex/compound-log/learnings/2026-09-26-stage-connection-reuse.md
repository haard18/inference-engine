## 2026-09-26 - Reuse the activation connection

**Context**: After batched prompt evaluation, every generated token still makes a suffix activation request. The peer client opened a new TCP, mutual TLS, and HTTP connection for each one.

**Learning**: HTTP/1 connection reuse needs a single owner for the request sender until its bounded response body has been consumed. Stage readiness checks must use a separate connection so a slow model step cannot block health reporting behind that sender's lock. The existing loss test stopped only the server accept loop; an established connection kept working. A real device-loss test must close active sockets too.

**Pattern**: Share one authenticated sender among clones of the approved peer client for serial activations. Check whether it is closed before the next request. On a send or body failure, discard it and report the error; reconnect only for a later request. Keep activation replay forbidden because the suffix may have committed the step before the failure became visible.

**Anti-pattern**: Do not repeat the TLS handshake for each generated token, retry a failed activation when its commit status is unknown, or simulate device loss by stopping only new connection acceptance.
