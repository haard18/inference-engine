# Local serving load and Metal memory review

The runner starts its own loopback server with a random API key and stops the process group in a `finally` block. It validates complete ordinary and streaming responses, identical generated text, health after each wave, and an unchanged worker PID. It reports sampled total, server, and worker RSS separately. The key is kept in the child environment and is absent from the report. A benchmark subprocess error includes its diagnostic output.

The Metal change places an Objective-C autorelease pool around one synchronous model step. The returned scores are owned Rust values. The model runtime and session key/value buffers are retained Metal objects, so they remain valid when the pool drains. Real-model CPU–Metal score parity and the 240-request serving run passed after the change.

The memory conclusion is limited to resident-memory samples every 200 ms, one model, and fixed prompts. The runner detects a worker restart but does not inject device loss, vary prompt size, or measure physical network service. Those remain separate acceptance work for the full engine goal.
