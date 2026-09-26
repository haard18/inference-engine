# CPU output-row review

The change schedules complete output rows independently and preserves each row's original calculation. Matrix construction already validates dimensions, so row slices stay within the same storage bounds as the old chunk iterator. The zero-width case exits before the new row indexing. Small matrices still use serial iteration. The test covers the parallel threshold, and real-model checks cover all supported GGUF matrix types.

The default Rayon pool uses the host's available threads. That improves one-worker latency on the measured Mac but could compete with other worker processes on the same device. The shared pool avoids creating a new thread set for every matrix call. A four-thread check retained the observed speed benefit. No public API or model format changed.

The benchmark compared two local binaries with alternating order and matching text hashes. Its fresh-process timing includes model loading and pool startup. The separate loopback API run verifies serving behavior after the change; it does not prove a server speedup factor. A short local run does not prove throughput under sustained load or on other devices.
