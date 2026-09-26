# Metal token-command review

Each layer's output buffer is the next layer's input in the same ordered command. The decoder retains every intermediate buffer until command completion and checks the final command status before reading scores. Final normalization weights live in the prepared Metal runtime. The session does not commit a token position until the command succeeds and its scores are finite. Cache growth copies only previously committed positions, which are complete before the next token command starts.

The 59 release tests passed with real-model tests included. Clippy passed with warnings denied, the release build succeeded, and format and whitespace checks passed. Cargo still reports its existing future-compatibility warning for `block` 0.1.6.

One host and a short prompt cannot establish serving throughput. The GPU command retains more temporary buffers at once, while CPU and GPU matrix copies keep Metal resident memory higher than CPU. A two-physical-Mac split-serving comparison remains open for the broader engine goal.
