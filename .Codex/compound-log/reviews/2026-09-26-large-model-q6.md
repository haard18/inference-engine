# 2026-09-26 - Larger GGUF and split serving review

## Delivered

- Added Q6_K matrix decoding in CPU and Metal execution, with compact 210-byte blocks and scale validation.
- Accepted and validated the optional GGUF padding-token ID used by SmolLM2-1.7B-Instruct.
- Made the one-host split load script accept a layer count and split boundary. Its defaults still cover the 30-layer, 135M checkpoint.
- Added an opt-in Metal test for the official 24-layer, 1.7B Q4_K_M checkpoint.

## Evidence

- Official checkpoint SHA-256: `decd2598bc2c8ed08c19adc3c8fdd461ee19ed5708679d1c54ef54a5a30d4f33`.
- The synthetic Q6_K test checks all four packed value groups, signed subscales, matrix multiplication, length validation, and non-finite scale rejection.
- Complete and 12/12 split Metal execution agreed on next-token scores within `1e-3` and selected tokens across four positions. Stored-weight bytes were 1,053,827,072 complete, 579,010,560 prefix, and 557,391,872 suffix.
- With the same eight-token `Hello` prompt and greedy settings, two measured Metal runs per engine produced the same text as pinned llama.cpp commit `81bc6b83f827df746eb129235488d325c49cae52`. Fresh-process medians were 0.584 s and 0.347 s. The small sample includes load time.
- One-host serving completed two ordinary and two streaming requests for both complete and split modes (eight total), all with matching text and stable workers. Peak sampled process-tree RSS was 2,149,872 KiB complete, 1,121,312 KiB prefix, and 1,142,640 KiB suffix. This short smoke run does not establish sustained throughput.
- The regular release test suite, strict Clippy, and formatting check passed after the change.

## Remaining concern

Two physical Macs are still needed for the LAN acceptance check. The second Mac has not been set up. The present 1.7B run verifies partial-weight placement on separate processes on one Mac, not remote capacity or link behavior under load.
