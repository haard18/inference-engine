use std::env;
use std::net::{SocketAddr, TcpListener};
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use inference_engine::pool::client::PeerClient;
use inference_engine::pool::{DeviceIdentity, PeerStore};
use inference_engine::{
    load_gguf, load_gguf_stage, ActivationFrame, GenerationSession, StageSession,
};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use uuid::Uuid;

#[tokio::test]
#[ignore = "requires SmolLM2-135M-Q4_K_M.gguf in SMOLLM2_DIR"]
async fn approved_peer_runs_the_suffix_stage_with_real_model_parity() {
    let directory = PathBuf::from(env::var("SMOLLM2_DIR").expect("set SMOLLM2_DIR"));
    let path = directory.join("SmolLM2-135M-Q4_K_M.gguf");
    let server_dir = tempfile::tempdir().unwrap();
    let approved_dir = tempfile::tempdir().unwrap();
    let unapproved_dir = tempfile::tempdir().unwrap();
    let server = DeviceIdentity::load_or_create(server_dir.path()).unwrap();
    let approved = DeviceIdentity::load_or_create(approved_dir.path()).unwrap();
    let unapproved = DeviceIdentity::load_or_create(unapproved_dir.path()).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address: SocketAddr = listener.local_addr().unwrap();
    let mut server_peers = PeerStore::load(server_dir.path()).unwrap();
    server_peers
        .trust(
            &server.device_id,
            &approved.offer(),
            address,
            &approved.fingerprint,
        )
        .unwrap();
    let mut approved_peers = PeerStore::load(approved_dir.path()).unwrap();
    approved_peers
        .trust(
            &approved.device_id,
            &server.offer(),
            address,
            &server.fingerprint,
        )
        .unwrap();
    let mut unapproved_peers = PeerStore::load(unapproved_dir.path()).unwrap();
    unapproved_peers
        .trust(
            &unapproved.device_id,
            &server.offer(),
            address,
            &server.fingerprint,
        )
        .unwrap();

    drop(listener);
    let mut serving = Command::new(env!("CARGO_BIN_EXE_serve"))
        .arg("--stage-suffix")
        .arg(server_dir.path())
        .arg(address.to_string())
        .arg(&path)
        .arg("15")
        .arg("30")
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let mut output = BufReader::new(serving.stdout.take().unwrap());
    let mut ready = String::new();
    tokio::time::timeout(Duration::from_secs(30), output.read_line(&mut ready))
        .await
        .unwrap()
        .unwrap();
    assert!(ready.contains("Serving approved suffix stage"), "{ready}");
    let client = PeerClient::new(&approved, approved_peers.peers()[0].clone()).unwrap();
    let foreign = PeerClient::new(&unapproved, unapproved_peers.peers()[0].clone()).unwrap();
    assert!(foreign.stage_snapshot().await.is_err());

    let prefix = load_gguf_stage(&path, 0..15).unwrap();
    let snapshot = client.stage_snapshot().await.unwrap();
    assert!(snapshot.ready);
    assert_eq!(snapshot.model_digest, hex::encode(prefix.model_digest()));
    assert_eq!(snapshot.layer_start, 15);
    assert_eq!(snapshot.layer_end, 30);
    assert_eq!(snapshot.hidden_size, 576);
    assert_eq!(snapshot.stored_weight_bytes, 59_348_736);
    assert_eq!(snapshot.queue_capacity, 4);
    let model = load_gguf(&path).unwrap();
    let mut whole = GenerationSession::new(&model);
    let mut first = StageSession::new(&prefix);
    let id = Uuid::new_v4();
    for token in [1, 2, 3, 30] {
        whole.prefill(&[token]).unwrap();
        let position = first.position();
        let hidden = first.forward_token(token).unwrap();
        let frame = ActivationFrame::new(&prefix, id, position, hidden)
            .unwrap()
            .encode();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
        if position == 0 {
            assert!(client
                .forward_stage(
                    frame.clone(),
                    Uuid::new_v4(),
                    model.config().vocab_size,
                    deadline
                )
                .await
                .is_err());
        }
        let scores = client
            .forward_stage(frame, id, model.config().vocab_size, deadline)
            .await
            .unwrap();
        assert_eq!(scores, whole.next_token_scores().unwrap());
    }
    client
        .close_stage(id, tokio::time::Instant::now() + Duration::from_secs(5))
        .await
        .unwrap();
    serving.kill().await.unwrap();
    serving.wait().await.unwrap();
    let hidden = first.forward_token(1).unwrap();
    let frame = ActivationFrame::new(&prefix, id, 4, hidden)
        .unwrap()
        .encode();
    assert!(client
        .forward_stage(
            frame,
            id,
            model.config().vocab_size,
            tokio::time::Instant::now() + Duration::from_secs(3),
        )
        .await
        .is_err());
}
