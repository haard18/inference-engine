# 2026-09-26 - Drain temporary Metal objects in a long-lived worker

**Context:** A local serving check alternated ordinary and streaming Q4_K_M requests at concurrency two. The Metal worker kept the same process ID and returned identical text, but its resident memory rose by about 2 MiB per 30-request wave after the conversation cache had filled.

**Learning:** The Metal library creates temporary Objective-C objects during a decoder step. A command-line probe exits before their lifetime is visible. A server child loops on the same thread, so temporary autoreleased objects can accumulate. Wrap the Metal forward step in `objc::rc::autoreleasepool`; the model runtime and session cache retain their own Metal buffers across that pool.

**Evidence:** Before the change, an eight-wave, 240-request Metal run completed every request but combined server and worker RSS rose from 276,128 KiB after wave one to 291,600 KiB after wave eight. A separate four-wave run identified the worker as the source: worker RSS rose from 250,544 to 257,024 KiB while server RSS moved from 24,720 to 24,912 KiB. After the change, another eight-wave run completed all 240 requests with the same output digest and stable worker ID; worker RSS was 248,576 KiB after wave one and 248,384 KiB after wave eight. Its request rate stayed between 3.92 and 3.94 requests/s across the eight waves. The four-wave CPU run completed all 120 requests with one worker and no output mismatch. Release tests, the real GGUF score comparisons, and Clippy passed.

**Pattern:** Exercise long-lived model workers through repeated load, separate API and worker memory, and compare memory after the bounded request cache has warmed. Drain autoreleased Metal objects at each synchronous inference step. Keep the saved model and conversation buffers outside the pool through owned references.

**Limit:** `ps` sampling can miss short memory spikes. This run used one Mac, one Q4_K_M model, one prompt, and a single server process. It does not prove memory behavior for all models, prompt lengths, or two physical devices.
