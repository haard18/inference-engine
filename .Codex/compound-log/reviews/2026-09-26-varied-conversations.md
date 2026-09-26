# Varied conversation parity review

The new real-model helper compares cached and fresh choices at each growing turn, checks exact cached-token counts, and parses the streamed response through its finish marker. Both backends then change the prefix and confirm zero cache reuse with correct output. The CPU test keeps its original eight-session eviction assertion before the longer conversation sequence; a new fresh comparison confirms its output as well as its cache count. The Metal test runs the same sequence against its own child worker.

The test is ignored by default because it needs the model file. Six histories per backend make it slower than a unit test but cover repeated checkpoint replacement and cache eviction. No serving implementation changed in this cycle. The result is correctness evidence for this model and one host, not a claim about concurrent network operation.
