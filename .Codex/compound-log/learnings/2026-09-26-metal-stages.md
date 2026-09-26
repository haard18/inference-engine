# 2026-09-26 - Metal execution for partial model stages

The split model loader already keeps only the weights for its assigned decoder layers, but each stage session ran on CPU. The complete Metal decoder assumed every step ended with final normalization and vocabulary projection. A prefix stage ends at the hidden activation instead.

The Metal decoder now accepts a stage without final normalization or projection and returns its hidden activation after the last owned layer. A suffix stage uses its own uploaded layer weights, final normalization, and output projection. Stage sessions keep separate Metal key/value caches, commit positions only after a successful command, and retain the existing rewind behavior.

The real Q4_K_M SmolLM2 test split 30 layers at layer 15. Across four tokens, the two Metal stage sessions matched the complete Metal model's scores within 1e-3. Rewinding both stages to position two and replaying the third token reproduced its scores within 1e-3. `cargo test`, `cargo clippy --all-targets -- -D warnings`, and `cargo build` passed.

The serving child process still creates CPU stage sessions. The next layer is to carry a Metal backend choice through prefix and suffix child startup and verify split chat serving through those children. Physical two-Mac validation remains unavailable because the second Mac is not set up.
