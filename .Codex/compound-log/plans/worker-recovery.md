# Repeated worker recovery measurement

## Need

The isolated server replaces a child process after it exits, but the existing test injects only one hang and one exit. A serving hub needs evidence that repeated idle worker crashes do not leave the API stuck or leak process memory.

## Approach

Start the release server with a real GGUF model, a temporary local API key, and an unused loopback port. After one warm-up request, kill only the server's direct child process. Observe health, wait for a different child and HTTP 200 readiness, and send a successful chat request. Repeat several times. Sample the server process tree's resident memory while it recovers. End the process group in cleanup so no test worker remains.

## Acceptance

- Every injected child loss is followed by a different ready child and a successful real-model chat.
- The script reports each recovery duration, observed HTTP 503 status, and sampled process-tree resident memory without printing the temporary API key.
- The test is bounded by explicit startup and per-cycle timeouts and leaves no server process running.

## Limits

This is a one-host idle-crash test. It does not establish latency during concurrent load or predict recovery on another Mac. Memory values are samples and can miss peaks between `ps` calls.

## Progress

- [x] Add a bounded fault test for CPU and Metal workers. Use an internal temporary key, kill only the server's direct child, and clean up the process group.
- [x] Run five CPU and three Metal real-model crash/recovery cycles. Each cycle observed HTTP 503, then a new ready child, then a successful chat.
- [x] Drop three real-model streams after visible content and confirm later requests still complete.
