# 2026-09-26 - Capacity probes during split inference

The prefix's readiness monitor timed out a capacity request after 750 ms. The suffix's capacity route waited on a mutex that a token step held for its entire child-process call. A valid slow step could therefore look like device loss.

The capacity route now tries the mutex without waiting. If a request owns it, the route reads a readiness flag that stays true while the child is processing. Queue availability comes from the semaphore and can be zero while readiness remains true. If a child fails or its request is canceled, a lease drops the child and clears readiness before the worker lock is released. A replacement sets readiness after its identity is checked.

This fixes lock contention in the suffix response path. It cannot prevent a real network delay, a blocked HTTP runtime, or an actual child failure from exceeding the prefix's probe timeout. The two-Mac under-load check remains open.
