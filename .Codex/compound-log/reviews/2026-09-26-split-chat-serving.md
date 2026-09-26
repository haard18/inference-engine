# Split chat serving review

The split path uses the existing authenticated local API and the previously approved peer TLS client. It creates one request ID per chat and keeps that placement fixed. The API and stage clients share the request deadline. A failed stage step cannot continue with partially committed state: the local child is discarded, and the remote close is attempted with a bounded cleanup time.

The real-model test covers ordinary response parity, streaming parity, wrong stage range, peer loss after text begins, and a later request when the peer is gone. The complete release suite passed 55 tests. Clippy passed with warnings denied.

Remaining risks: the current split path has no cross-request conversation cache reuse; readiness only reflects the local prefix process between requests, while suffix availability is checked when a chat job starts; a snapshot cannot reserve distributed capacity atomically. Physical two-device performance and failure recovery are unmeasured. These are explicit next layers, not claims of completion.
