# Metal decoder-layer review

Each command orders separate encoders so a buffer is produced before its next consumer reads it. The RMS scale kernel sums the hidden vector in f32, then an application kernel multiplies by the stored normalization weights. Matrix shapes, head layout, buffer lengths, and grid sizes are checked before dispatch. The session position is committed only after all layers and logits finish; cache growth copies only previously committed slots.

All 59 release tests passed with real-model tests included. Clippy passed with warnings denied, the release build succeeded, and format and whitespace checks passed. Cargo still reports the existing future-compatibility warning for `block` 0.1.6.

The short local timing sample suggests lower overhead for this prompt. It does not establish serving throughput. The hidden vector still returns to CPU after each layer, and CPU and GPU weight copies keep Metal process memory higher than CPU process memory.
