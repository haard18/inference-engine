# 2026-09-26 - Worker isolation review

## Delivered

- A persistent child process for GGUF model execution and a parent supervisor that restarts it after a timeout, disconnect, broken pipe, or exit.
- A private line-based protocol for startup, prompts, streamed text, errors, and finish events.
- A CLI launch path that keeps model weights outside the HTTP process and does not pass the API key to the worker.

## Validation

- Live local HTTP requests returned ordinary JSON and streamed replies from the real model.
- Real-model integration tests passed for both response forms through CPU and Metal child processes.
- A fake worker hang and exit test passed and verified replacement.
- Full release suite, strict Clippy, and formatting checks passed.

## Scope

The process supervisor has one active model worker. It has not yet been measured under repeated crashes or heavy concurrent traffic. The cross-device pool, encrypted pairing, reusable conversation state, and model splitting remain open.
