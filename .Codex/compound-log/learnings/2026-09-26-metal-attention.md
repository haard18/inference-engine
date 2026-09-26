# 2026-09-26 - Metal attention and key/value state

Metal generation now stores each session's key/value vectors in growing Metal shared buffers. A score kernel computes scaled query/key products for each attention head and token position. A second kernel applies softmax weights to the value history. The session keeps only the committed token position in its CPU cache structure; it does not keep CPU key/value vectors. If a model step fails before producing logits, a retry overwrites the uncommitted slot. Rewinding a prompt checkpoint changes the committed position, so later steps overwrite positions beyond that checkpoint.

The Metal buffers begin with room for at most 16 positions and grow as needed, up to the model context limit. The retained buffer sizes count toward the existing 128 MiB conversation-checkpoint limit. This avoids allocating the full key/value context for every new session. Buffer growth copies only committed positions.

Tiny f32 and f16 models matched CPU scores within `1e-4`. A tiny-model run checked each of 20 positions, including growth past the initial 16-slot boundary. Real Q8_0 and Q4_K_M checkpoints matched CPU scores within `1e-3` at four positions and chose the same next token. The existing real-model Metal conversation test confirmed that a prompt checkpoint can be reused after rewind.

On this Apple Silicon Mac, one warm eight-token Q4_K_M probe with the prompt `Hello` took 0.58 seconds on CPU and 0.67 seconds with Metal attention. Maximum process RSS was 117.1 MiB and 223.5 MiB respectively. A preceding cold CPU run took 1.06 seconds, so these short observations do not support a stable speed claim. The Metal process still retains CPU weights in addition to uploaded weights.

Metal shared buffers are accessible by both CPU and GPU. The matrix path still returns query, key, and value vectors to the CPU for rotary positions, then writes key/value vectors into the shared buffers. RMS normalization and feed-forward activation also remain on the CPU. Each matrix group and attention command waits for completion. Removing these per-operation boundaries remains the next Metal optimization layer.
