# Serving measurement review

The streaming benchmark treats HTTP 200 with an SSE error or an incomplete stream as a failed request. It reports first visible content rather than claiming to know the model's internal first-token time. The response byte limit and request timeout remain active in both modes. The fixed prompt, warm-ups, request count, and concurrency are shared across complete, pooled, and split runs.

The RSS sampler follows each server's descendants, validates its input, and samples without reading command-line arguments or credentials. It cannot observe spikes between samples or separate shared memory from private memory; the reported figures are sampled process RSS sums.

One-host results are pilot evidence only. The physical two-Mac acceptance check still needs repeated runs, per-device memory samples, network conditions, and injected link loss. No result here supports a claim that split serving is faster than a complete worker or that a one-host pool predicts real LAN throughput.
