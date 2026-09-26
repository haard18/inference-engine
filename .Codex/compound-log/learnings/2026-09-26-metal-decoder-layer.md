# 2026-09-26 - One Metal command per decoder layer

The previous Metal path waited after attention, again after the attention-output matrix, and again after the feed-forward network. The CPU used the intermediate vectors for RMS normalization and residual additions. The new path uploads each layer's normalization weights once and runs normalization, query/key/value projection, rotary rotation, attention, attention-output projection, both residual additions, and feed-forward work in one ordered Metal command. The CPU receives the completed hidden vector after that layer.

The layer still uses a session-owned key/value buffer and a separate committed token position. A failed step cannot advance that position, and later work can overwrite an uncommitted slot. The 59-test release suite includes a 20-position cache-growth comparison, Metal conversation reuse, and real bf16/Q8_0/Q4_K_M score comparisons; all passed.

In one short sequential `Hello` eight-token probe, CPU runs took 1.15, 0.58, and 0.58 seconds; Metal runs took 0.35, 0.34, and 0.34 seconds. Both backends generated the same text. Separate maximum resident-memory samples were 121.9 MB on CPU and 231.7 MB with Metal. The first CPU run was cold. These measurements do not establish a stable serving speed ratio. CPU matrix weights remain loaded beside GPU buffers.

One completion wait remains per decoder layer, and the hidden vector crosses back to the CPU before the next layer. Final normalization and output projection also cross the CPU/GPU boundary. Keeping the hidden vector on the GPU across layers is the next Metal boundary.
