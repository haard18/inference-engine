# 2026-09-26 - Reviewable LAN comparison

The benchmark now hashes completed response text as it arrives. It uses the same SHA-256 digest for ordinary JSON text and streamed content chunks, without retaining every response body. The benchmark reports digest counts and uses only completed requests for p50 and p95 latency. HTTP statuses, transport failures, and streaming application errors remain separate.

The comparison tool reads at least three complete, pooled, and split reports by default. It requires matching model and load settings, all requests successful, one identical text digest, and two distinct pooled owners when concurrency is at least two. It reports median request rates, p50/p95 latency, and first visible content for streaming runs. This is a data consistency check. Server startup records and separate process-tree memory samples are still needed to prove the physical topology and resource use.

A local real-model smoke run sent six ordinary and six streaming requests at concurrency two. All twelve returned HTTP 200, and both modes produced the same completed-text digest. The second physical Mac was not visible in the local SSH service list, so no two-Mac performance or loss result is claimed.
