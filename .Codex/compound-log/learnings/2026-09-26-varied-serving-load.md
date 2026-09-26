# 2026-09-26 - Short varied serving load

The earlier local soak repeated one prompt. I ran a second check against the release server and the real SmolLM2-135M Q4_K_M GGUF. It used six different user prompts, four repetitions per prompt, eight requested completion tokens, and at most two concurrent clients. Each backend served 24 ordinary and 24 streaming requests through one worker. The prompts asked for a sky color, seven times six, a French greeting, a Moon fact, a Rust sentence continuation, and a planet name.

All 48 CPU requests and all 48 Metal requests completed. Each prompt produced one text digest across its repetitions, ordinary and streaming responses matched, and CPU and Metal produced the same digest for every prompt. Every stream included a finish event and `[DONE]`; no worker restarted, and both servers were healthy afterward.

With process-tree resident memory sampled about every 200 ms, CPU measured 131,184 KiB after the ordinary wave and 131,824 KiB after the streaming wave. Metal measured 265,712 KiB and 266,016 KiB. The sampled peaks were 151,504 KiB for CPU and 268,032 KiB for Metal. Each run had only two waves and one small model. These samples cannot rule out slow growth or short memory spikes. The actual two-Mac serving and split comparison is still open because the second Mac is not set up yet.
