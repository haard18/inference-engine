# Varied conversation parity plan

## Purpose

Verify that saved conversation state remains correct as message history grows and prompts change. Single-repeat tests do not cover several consecutive checkpoints or a streamed request between turns.

## Approach

Use the real Q4_K_M model through the isolated serving API. For each turn, send the full message history with one conversation ID and compare the answer with a fresh request that has no cached state. Check the exact reused prompt length when the new history extends the old prompt. Insert a streamed request, then confirm the following turn still reuses the saved prefix. Run the same sequence on CPU and Metal.

## Acceptance

- Every cached answer matches fresh full-history inference.
- Extended histories report the prior prompt length as cached tokens; changed history recomputes.
- A streamed response completes and does not break the next checkpoint.
- Existing eviction behavior and the full release test suite remain valid.

## Result

The real-model integration test now runs six distinct multi-turn conversations on both CPU and Metal. Each growing-history cached answer matched a fresh full-history answer and reported the exact prior prompt length as reused. A streamed turn completed and preserved the next checkpoint. Changed histories recomputed correctly on both backends. The CPU eviction assertion remains before the longer sequence so it still isolates the eight-entry limit.
