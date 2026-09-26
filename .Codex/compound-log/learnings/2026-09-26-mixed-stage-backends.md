# 2026-09-26 - Mixed CPU and Metal stages

The split service lets the prefix and suffix choose their backends independently, but the first real-model serving test used Metal on both sides. Device owners may have one Mac with GPU capacity and another device that runs a stage on CPU, so mixed execution needs separate evidence.

With SmolLM2-135M Q4_K_M split at layer 15, a CPU prefix with a Metal suffix and a Metal prefix with a CPU suffix each matched the complete CPU model's scores within 1e-3 across four token positions. Both combinations also completed real chat through the approved loopback TLS stage service. Chat choices and usage matched a complete Metal worker; streaming completed with the same text, and an exact repeated prompt reused its conversation checkpoint.

A suffix restart test now requires the fake child to receive `--metal` on both its first start and its replacement after a canceled step. That test checks backend propagation, not GPU arithmetic. The real-model tests check GPU arithmetic and process protocol separately. The complete default test suite, Clippy with warnings denied, and the build passed.

These checks cover backend interoperability on one Mac. They do not prove performance or reliability across two physical devices; the second Mac remains unconfigured.
