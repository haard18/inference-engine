# Local serving load plan

## Purpose

Verify that one real-model serving worker stays healthy through repeated ordinary and streaming requests. A short one-wave benchmark does not show whether memory keeps rising or a worker silently restarts.

## Approach

Start the release server on an unused loopback port with a fresh random Bearer key. Alternate ordinary and streaming `pool-bench` waves at the same request count, concurrency, and completion limit. Sample process-tree resident memory during each wave. Check every benchmark report for complete responses, one output digest, and matching text across modes. Check health and the worker process ID after every wave. Stop the server process group on success or failure.

## Acceptance

- A short smoke run verifies the runner's startup, cleanup, and report checks.
- A longer Q4_K_M run has no failed requests, output mismatch, worker restart, or failed health check.
- The report includes per-wave latency, throughput, and sampled memory. It records the limits of one host and one run.
- If worker memory keeps growing after its bounded session cache fills, isolate the process, fix the growth, and repeat the same load.
- The learning log and index record the result and any failure found.

## Result

The repeated load found steady memory growth in the Metal child process. A per-token Objective-C autorelease pool removed that growth in a matched eight-wave run. CPU and Metal checks completed every ordinary and streaming request with one stable worker. The numerical real-model tests still pass. Physical two-device and longer varied-prompt runs remain open.

A separate short varied-prompt run used six prompts, four repetitions per prompt, and both ordinary and streaming responses on each backend. CPU and Metal each completed 48 requests. Every repeated prompt and response mode produced the same text, and neither worker restarted. Process-tree resident memory after the ordinary and streaming waves changed by +640 KiB on CPU and +304 KiB on Metal. This two-wave check adds prompt diversity but does not establish long-term memory stability.
