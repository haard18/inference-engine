use std::env;
use std::net::{SocketAddr, TcpListener};
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use inference_engine::pool::client::PeerClient;
use inference_engine::pool::{DeviceIdentity, PeerStore, PoolError};
use inference_engine::{
    load_gguf, load_gguf_stage, ActivationFrame, GenerationSession, StageSession,
};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::net::{TcpListener as TokioTcpListener, TcpStream};
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
    let proxy = TokioTcpListener::bind("127.0.0.1:0").await.unwrap();
    let proxy_address = proxy.local_addr().unwrap();
    let connections = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&connections);
    let proxy_task = tokio::spawn(async move {
        while let Ok((mut incoming, _)) = proxy.accept().await {
            counted.fetch_add(1, Ordering::SeqCst);
            tokio::spawn(async move {
                if let Ok(mut upstream) = TcpStream::connect(address).await {
                    let _ = tokio::io::copy_bidirectional(&mut incoming, &mut upstream).await;
                }
            });
        }
    });
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
            proxy_address,
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
    assert_eq!(
        client
            .reserve_stage(id, 4, tokio::time::Instant::now() + Duration::from_secs(30))
            .await
            .unwrap(),
        0
    );
    let prior_connections = connections.load(Ordering::SeqCst);
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
    assert_eq!(connections.load(Ordering::SeqCst), prior_connections + 1);
    client
        .close_stage(id, tokio::time::Instant::now() + Duration::from_secs(5))
        .await
        .unwrap();
    let batch_id = Uuid::new_v4();
    client
        .reserve_stage(
            batch_id,
            4,
            tokio::time::Instant::now() + Duration::from_secs(30),
        )
        .await
        .unwrap();
    let mut batch_prefix = StageSession::new(&prefix);
    let mut batch_whole = GenerationSession::new(&model);
    let mut batch = Vec::new();
    for token in [1, 2, 3, 30] {
        batch_whole.prefill(&[token]).unwrap();
        let position = batch_prefix.position();
        let hidden = batch_prefix.forward_token(token).unwrap();
        batch.extend(
            ActivationFrame::new(&prefix, batch_id, position, hidden)
                .unwrap()
                .encode(),
        );
    }
    let scores = client
        .forward_stage_batch(
            batch,
            4,
            batch_id,
            model.config().vocab_size,
            tokio::time::Instant::now() + Duration::from_secs(30),
        )
        .await
        .unwrap();
    assert_eq!(scores, batch_whole.next_token_scores().unwrap());
    assert_eq!(
        client
            .probe_stage(
                batch_id,
                tokio::time::Instant::now() + Duration::from_secs(5)
            )
            .await
            .unwrap(),
        Some(4)
    );
    client
        .close_stage(
            batch_id,
            tokio::time::Instant::now() + Duration::from_secs(5),
        )
        .await
        .unwrap();
    let invalid_id = Uuid::new_v4();
    client
        .reserve_stage(
            invalid_id,
            2,
            tokio::time::Instant::now() + Duration::from_secs(5),
        )
        .await
        .unwrap();
    let mut invalid_prefix = StageSession::new(&prefix);
    let hidden = invalid_prefix.forward_token(1).unwrap();
    let frame = ActivationFrame::new(&prefix, invalid_id, 0, hidden)
        .unwrap()
        .encode();
    let mut invalid_batch = frame.clone();
    invalid_batch.extend(frame);
    assert!(client
        .forward_stage_batch(
            invalid_batch,
            2,
            invalid_id,
            model.config().vocab_size,
            tokio::time::Instant::now() + Duration::from_secs(5),
        )
        .await
        .is_err());
    assert_eq!(
        client
            .probe_stage(
                invalid_id,
                tokio::time::Instant::now() + Duration::from_secs(5)
            )
            .await
            .unwrap(),
        None
    );
    let mut active = Vec::new();
    for _ in 0..8 {
        let stage_id = Uuid::new_v4();
        client
            .reserve_stage(
                stage_id,
                1,
                tokio::time::Instant::now() + Duration::from_secs(5),
            )
            .await
            .unwrap();
        let mut stage_session = StageSession::new(&prefix);
        let hidden = stage_session.forward_token(1).unwrap();
        let frame = ActivationFrame::new(&prefix, stage_id, 0, hidden)
            .unwrap()
            .encode();
        client
            .forward_stage(
                frame,
                stage_id,
                model.config().vocab_size,
                tokio::time::Instant::now() + Duration::from_secs(5),
            )
            .await
            .unwrap();
        active.push(stage_id);
    }
    let ninth = Uuid::new_v4();
    let mut stage_session = StageSession::new(&prefix);
    let hidden = stage_session.forward_token(1).unwrap();
    let frame = ActivationFrame::new(&prefix, ninth, 0, hidden)
        .unwrap()
        .encode();
    assert!(client
        .reserve_stage(
            ninth,
            1,
            tokio::time::Instant::now() + Duration::from_secs(5),
        )
        .await
        .is_err());
    client
        .rewind_stage(
            active[0],
            1,
            tokio::time::Instant::now() + Duration::from_secs(5),
        )
        .await
        .unwrap();
    client
        .reserve_stage(
            ninth,
            1,
            tokio::time::Instant::now() + Duration::from_secs(5),
        )
        .await
        .unwrap();
    client
        .forward_stage(
            frame,
            ninth,
            model.config().vocab_size,
            tokio::time::Instant::now() + Duration::from_secs(5),
        )
        .await
        .unwrap();
    assert_eq!(
        client
            .probe_stage(
                active[0],
                tokio::time::Instant::now() + Duration::from_secs(5)
            )
            .await
            .unwrap(),
        None
    );
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

    let mut restarted = Command::new(env!("CARGO_BIN_EXE_serve"))
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
    let mut restarted_output = BufReader::new(restarted.stdout.take().unwrap());
    ready.clear();
    tokio::time::timeout(
        Duration::from_secs(30),
        restarted_output.read_line(&mut ready),
    )
    .await
    .unwrap()
    .unwrap();
    assert!(ready.contains("Serving approved suffix stage"), "{ready}");
    let resumed = Uuid::new_v4();
    client
        .reserve_stage(
            resumed,
            1,
            tokio::time::Instant::now() + Duration::from_secs(5),
        )
        .await
        .unwrap();
    let mut resumed_prefix = StageSession::new(&prefix);
    let frame = ActivationFrame::new(
        &prefix,
        resumed,
        0,
        resumed_prefix.forward_token(1).unwrap(),
    )
    .unwrap()
    .encode();
    let scores = client
        .forward_stage(
            frame,
            resumed,
            model.config().vocab_size,
            tokio::time::Instant::now() + Duration::from_secs(5),
        )
        .await
        .unwrap();
    let mut resumed_whole = GenerationSession::new(&model);
    resumed_whole.prefill(&[1]).unwrap();
    assert_eq!(scores, resumed_whole.next_token_scores().unwrap());
    restarted.kill().await.unwrap();
    restarted.wait().await.unwrap();
    proxy_task.abort();
}

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore = "requires SMOLLM2_1_7B_GGUF with a 128 MiB stage cache budget"]
async fn remote_stage_reserves_full_context_before_admitting_another_request() {
    let path = PathBuf::from(env::var("SMOLLM2_1_7B_GGUF").expect("set SMOLLM2_1_7B_GGUF"));
    let server_dir = tempfile::tempdir().unwrap();
    let approved_dir = tempfile::tempdir().unwrap();
    let server = DeviceIdentity::load_or_create(server_dir.path()).unwrap();
    let approved = DeviceIdentity::load_or_create(approved_dir.path()).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
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
    drop(listener);
    let mut serving = Command::new(env!("CARGO_BIN_EXE_serve"))
        .arg("--metal")
        .arg("--stage-suffix")
        .arg(server_dir.path())
        .arg(address.to_string())
        .arg(&path)
        .arg("12")
        .arg("24")
        .env("INFERENCE_STAGE_CACHE_MIB", "128")
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
    let snapshot = client.stage_snapshot().await.unwrap();
    assert_eq!(snapshot.max_positions, 512);
    let first = Uuid::new_v4();
    let second = Uuid::new_v4();
    let deadline = || tokio::time::Instant::now() + Duration::from_secs(10);
    assert_eq!(
        client.reserve_stage(first, 400, deadline()).await.unwrap(),
        0
    );
    assert_eq!(
        client.probe_stage(first, deadline()).await.unwrap(),
        Some(0)
    );
    assert!(matches!(
        client.reserve_stage(second, 400, deadline()).await,
        Err(PoolError::Overloaded(_))
    ));
    assert_eq!(client.probe_stage(second, deadline()).await.unwrap(), None);
    client.close_stage(first, deadline()).await.unwrap();
    assert_eq!(
        client.reserve_stage(second, 400, deadline()).await.unwrap(),
        0
    );
    client.close_stage(second, deadline()).await.unwrap();
    serving.kill().await.unwrap();
    serving.wait().await.unwrap();
}
