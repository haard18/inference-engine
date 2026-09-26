# 2026-09-26 - Metal feed-forward command

The Metal feed-forward path previously read gate and up vectors back to the CPU, applied SiLU and multiplication, then launched the down projection in a new command. A Metal kernel now computes the activation between the gate/up and down matrix encoders in one command buffer. The CPU receives only the down-projection output. This removes one completion wait and the gate/up vector readback per decoder layer.

The existing tiny f32 and f16 models, real bf16 model, and real Q8_0 and Q4_K_M models still match CPU scores within their established tolerances. A short `Hello` probe generated the same eight-token text with both backends. Three sequential Q4_K_M runs took 1.04, 0.58, and 0.58 seconds on CPU, then 0.48, 0.47, and 0.46 seconds with Metal. Separate maximum resident-memory samples were 121.9 MB for CPU and 232.5 MB for Metal. The first CPU run was cold; later timings are short single-process samples, not serving benchmarks. CPU weights remain beside GPU weights.

Normalization and residual addition still happen on the CPU. The attended vector, attention output projection, normalized feed-forward input, and feed-forward output still cross the CPU/GPU boundary. Removing these boundaries needs a larger GPU-resident decoder step.
