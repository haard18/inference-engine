# Metal attention review

The Metal cache is owned by each generation session, while prepared matrix weights and the command queue remain shared. The committed position controls reads and rewind. A failed step can leave bytes in an uncommitted slot, but a retry writes that slot before attention reads it. Cache growth copies only committed data. Persistent Metal buffers are included in the serving checkpoint memory count; temporary score and output buffers are bounded by context and model dimensions but are not retained.

The GPU score and reduction kernels implement grouped-query attention with the same head mapping and rotary inputs as the CPU path. Tiny and real-model parity tests cover multiple positions and a capacity increase. The full serving test also covers Metal prompt reuse. There is still duplicated decoder-layer logic between CPU and Metal, which must be kept aligned when model behavior changes.

The pilot did not show a speed or memory improvement over CPU. CPU/GPU synchronization remains frequent, and CPU and GPU copies of model weights remain allocated. This layer moves attention execution and key/value storage into Metal buffers; it does not make the entire decoder GPU-resident. The Metal dependency still emits a future-compatibility warning for `block` 0.1.6.

Verification: all 59 release tests passed with real-model tests included, Clippy passed with warnings denied, the release build succeeded, and the diff had no whitespace errors.
