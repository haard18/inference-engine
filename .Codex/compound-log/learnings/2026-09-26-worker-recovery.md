# 2026-09-26 - Repeated worker recovery

The server runs inference in a child process. A one-host fault tool now starts the release server with a temporary random API key, warms the real Q4_K_M model, kills only the server's direct child, waits for a different ready child, and completes a chat. It reports recovery time and sampled process-tree resident memory. The script ends the whole process group even if a cycle fails. It reads `ps` PID, parent PID, and RSS fields; it does not read command lines or print the key.

Five CPU cycles each exposed HTTP 503 during restart, then returned to HTTP 200 and completed a chat. Ready times were 673–714 ms, with a median of 702 ms. Time from the injected failure through the successful chat was 1,220–1,268 ms, with a median of 1,255 ms. Post-chat process-tree RSS samples were 149,360–149,472 KiB; the highest sample was 149,472 KiB (about 146.0 MiB).

Three Metal cycles also exposed HTTP 503, recovered, and completed a chat. Ready times were 711–728 ms, with a median of 722 ms. Failure-to-completed-chat times were 1,139–1,190 ms, with a median of 1,150 ms. Post-chat process-tree RSS samples were 257,920–258,032 KiB; the highest sample was 258,032 KiB (about 252.0 MiB).

A separate real-model test dropped three streaming responses after visible content and confirmed that a following ordinary request completed each time. This checks client-disconnect behavior in the isolated child path.

These are short idle-crash runs on one Apple Silicon Mac. They do not establish steady-state memory growth, response latency under load, or recovery over a physical network. `ps` sampling can miss memory spikes. The Metal process tree includes uploaded GPU-backed buffers as reported by process RSS; RSS alone does not measure all GPU memory.
