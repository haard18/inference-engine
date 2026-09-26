# Metal feed-forward review

The new kernel applies the same SiLU formula as the CPU path. Gate and up dimensions must match, and the down matrix must consume their width. The command checks matrix and buffer sizes before dispatch and checks completion before reading the final output. Separate encoders preserve the gate/up, activation, and down dependency order. Intermediate buffers remain alive until completion.

All 59 release tests passed with real-model tests included. Clippy passed with warnings denied, and the release build succeeded. The diff had no whitespace errors. Cargo still reports the existing future-compatibility warning for `block` 0.1.6.

The short local timing sample suggests less overhead for this prompt. It does not establish a speed gain under serving load, and the Metal process still holds both CPU and GPU copies of matrix weights. Normalization, residual addition, and further command waits remain open.
