# Two-Mac validation runbook

This is the remaining physical acceptance check for whole-request pooling and model splitting. Mac A serves the local chat API. Mac B supplies either a complete peer worker or a suffix stage. The two machines must be on a network where Mac A can reach Mac B's selected peer port. The chat API stays on each machine's loopback address.

Use the same release commit and the same GGUF file on both Macs. On **each** Mac, record `git rev-parse HEAD`, `shasum -a 256 MODEL.gguf`, `hostname`, and the Mac's LAN address. Keep those records with the benchmark reports. A matching model name is insufficient: the file digests must match. Build `serve`, `device`, and `pool-bench` with `cargo build --release --bin serve --bin device --bin pool-bench` on both Macs. Replace `MODEL.gguf`, `A_IP`, `B_IP`, and the state paths below with local values. Create a different random API key of at least 32 visible characters on each Mac; use Mac A's key for every benchmark. For the tested 1.7B model, use split layer 12 of 24; for the tested 135M model, use layer 15 of 30.

## Baseline on Mac A

Start one complete worker on Mac A with `INFERENCE_API_KEY` set to a random secret of at least 32 visible characters:

```sh
target/release/serve --metal MODEL.gguf 8080
```

From a second terminal on Mac A, use the same key for three ordinary and three streaming trials. Keep request count, concurrency, completion limit, and `INFERENCE_BENCH_SYSTEM_PROMPT` identical in every later scenario:

```sh
INFERENCE_API_KEY="$INFERENCE_API_KEY" target/release/pool-bench 127.0.0.1:8080 local-smollm2 30 2 16 > whole-ordinary-1.json
INFERENCE_API_KEY="$INFERENCE_API_KEY" target/release/pool-bench 127.0.0.1:8080 local-smollm2 30 2 16 --stream > whole-stream-1.json
```

Repeat each command twice with report numbers 2 and 3. During a trial, sample the server and child memory on Mac A with `python3 scripts/process-rss.py --seconds 30 --root whole=SERVER_PID > whole-memory.json`. Choose a duration that covers the full trial, and record the actual server PID. Stop the baseline server before starting the pool.

## Whole-request pool

Use one private state directory per Mac for this scenario. Create identities on their own machines, exchange only the public offer JSON files, and compare the full displayed certificate fingerprints through a separate trusted channel before approving either offer:

```sh
# Mac A
target/release/device init A_POOL_STATE
target/release/device offer A_POOL_STATE > a-pool-offer.json
target/release/device trust A_POOL_STATE b-pool-offer.json B_IP:8443 VERIFIED_B_FINGERPRINT

# Mac B
target/release/device init B_POOL_STATE
target/release/device offer B_POOL_STATE > b-pool-offer.json
target/release/device trust B_POOL_STATE a-pool-offer.json A_IP:8443 VERIFIED_A_FINGERPRINT
```

Copy each offer file to the other Mac before its `trust` command. The offers contain public certificate data; keep the private state directories on their own machines. Check `target/release/device peers STATE_DIR` on both Macs. Then start the complete peer workers:

```sh
# Mac B
INFERENCE_API_KEY="$B_API_KEY" target/release/serve --metal --peer B_POOL_STATE 0.0.0.0:8443 MODEL.gguf 8080

# Mac A
INFERENCE_API_KEY="$INFERENCE_API_KEY" target/release/serve --metal --peer A_POOL_STATE 0.0.0.0:8443 MODEL.gguf 8080
```

Only approved certificates can use the peer listeners. Allow inbound TCP port 8443 between the two Macs. Run the same six benchmark trials from Mac A, saving them as `pool-ordinary-N.json` and `pool-stream-N.json`. Run `process-rss.py` separately on both Macs during each trial; keep the reports labeled by host and server PID. The pool report must show completed requests owned by both device IDs at concurrency two. Record the identity-to-host mapping from `device show`.

Stop both pool servers before the split scenario.

## Split model

Use **new** state directories and offers for this scenario. A trusted peer record has one address and cannot be silently changed from the pool port to the suffix port. Exchange the new public offers and verify their full fingerprints through a separate trusted channel, as in the pool setup:

```sh
# Mac A
target/release/device init A_SPLIT_STATE
target/release/device offer A_SPLIT_STATE > a-split-offer.json
target/release/device trust A_SPLIT_STATE b-split-offer.json B_IP:8444 VERIFIED_B_SPLIT_FINGERPRINT

# Mac B
target/release/device init B_SPLIT_STATE
target/release/device offer B_SPLIT_STATE > b-split-offer.json
target/release/device trust B_SPLIT_STATE a-split-offer.json A_IP:8444 VERIFIED_A_SPLIT_FINGERPRINT
target/release/device show B_SPLIT_STATE
```

Mac B uses Mac A's approved certificate to authenticate inbound stage requests; Mac A does not need a stage listener at `A_IP:8444`. Take `B_SPLIT_DEVICE_ID` from the `device show` output on Mac B. Record the new device IDs and fingerprints.

```sh
# Mac B, after mutual approval
target/release/serve --metal --stage-suffix B_SPLIT_STATE 0.0.0.0:8444 MODEL.gguf 12 24

# Mac A, with the same local API key used above
INFERENCE_API_KEY="$INFERENCE_API_KEY" target/release/serve --metal --split-prefix A_SPLIT_STATE B_SPLIT_DEVICE_ID MODEL.gguf 12 8080
```

Allow inbound TCP port 8444 on Mac B. Use `15 30` instead of `12 24` in the Mac B command and split layer `15` on Mac A for the 135M model. Run the same six trials from Mac A, saving `split-ordinary-N.json` and `split-stream-N.json`. Sample the prefix process tree on Mac A and the suffix process tree on Mac B during each trial.

## Compare and test loss

Run the comparison once for ordinary reports and once for streaming reports:

```sh
python3 scripts/lan-compare.py --whole whole-ordinary-{1,2,3}.json --pool pool-ordinary-{1,2,3}.json --split split-ordinary-{1,2,3}.json
python3 scripts/lan-compare.py --whole whole-stream-{1,2,3}.json --pool pool-stream-{1,2,3}.json --split split-stream-{1,2,3}.json
```

The comparison rejects failed requests, changed load settings, changed output text, and a pool that used only one device. It does not prove physical placement by itself; keep the host, model digest, process, and memory records with it.

After the successful-load trials, run separate loss checks. Stop Mac B's peer server during an active pooled request and during an active split stream. Record the caller's result, Mac A's `/health` response, and time to recovery after Mac B restarts. For the pool, distinguish a failure before output from a broken stream after output. For the split, verify that a broken stage ends the stream with a clear error and that later requests work after restart. Keep these injected failures out of the successful-throughput comparison.

The accepted result requires matching generated text, successful sustained trials, both pool devices doing work, lower stored model weights on each split stage where the format permits, per-device memory samples, and observed loss and recovery behavior. Report measured rates and latency even if splitting is slower than one complete worker.
