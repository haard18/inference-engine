# Metal projection-to-attention review

The combined command retains separate encoder passes in one command buffer so each pass can use the preceding pass's buffers. Matrix shapes, head layout, grid sizes, and buffer lengths are checked before dispatch. The rotary table is finite for validated model settings, and the shader writes disjoint query pairs and key/value cache slots. Existing cache growth copies only committed positions; failed token work does not advance the session position.

The layer reduces one CPU/GPU wait and removes CPU reads of query, key, and value. It does not eliminate the later waits for attention output, feed-forward matrices, and logits. Process RSS remains more than the CPU run because CPU weights and uploaded weights coexist. The short timing sample does not justify a general speed claim.

Verification: all 59 release tests passed with real-model tests included. Clippy passed with warnings denied, the release build succeeded, and the diff had no whitespace errors. Cargo still reports the existing future-compatibility warning for `block` 0.1.6.
