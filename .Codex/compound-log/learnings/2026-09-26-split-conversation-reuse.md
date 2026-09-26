# 2026-09-26 - Split conversation checkpoints

An exact prompt checkpoint must exist in both partial model processes before split serving can reuse it. The gateway now saves the prompt token IDs, prompt-end scores, and a private stage session UUID after both stages acknowledge a rewind to the prompt position. A later request probes both stages for that UUID and position. It reuses only when the new tokenized prompt starts with the saved sequence and both probes agree. Otherwise it evaluates the full supplied history with a new UUID.

The suffix can restart or evict independently of the prefix. A cached gateway entry alone therefore cannot prove that remote key/value state still exists. The probe made this failure explicit and lets a request recover by recomputing before generation. The gateway clears all its entries if its own prefix child restarts.

The stage worker keeps active sessions separate from saved checkpoints. At its eight-session limit it rejects a new session if all eight are active. It may evict the oldest saved checkpoint to admit a new one. This avoids an unrelated request silently evicting a live chat. Each stage also keeps its 128 MiB cache bound and five-minute idle expiry. The gateway keeps at most eight entries and 16 MiB of prompt IDs and scores, also for five minutes.

A real Q4_K_M integration test checked full reuse for an identical follow-up, reuse after a streamed response, partial reuse for extended history, zero reuse for a changed prefix, and full recomputation after a suffix restart. A remote-stage test filled eight active sessions, checked that a ninth was rejected, then marked one session as a checkpoint and confirmed that the ninth could proceed after its eviction. These tests ran with mutually authenticated TLS on one host. Physical two-device behavior and sustained cache hit rates remain unmeasured.

Remote queue rejection is a capacity event rather than a model failure. The serving event path now preserves this distinction: ordinary chat returns HTTP 429 with `queue_full`, and an already opened stream emits the same code before its terminator. A focused API test covers both response forms.

A capacity check can fail before the prefix child receives any command. Such a failure should not discard every saved prompt in that child. The supervisor tracks whether the current request reached the prefix pipe and keeps the child when rejection happened before that point. It remains conservative after a partial token step, where stage state may be out of sync.
