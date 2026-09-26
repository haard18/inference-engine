use std::env;
use std::net::{SocketAddr, TcpListener};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use inference_engine::load_gguf_tokenizer;
use inference_engine::pool::client::PeerClient;
use inference_engine::pool::tls::{client_config, server_name};
use inference_engine::pool::{DeviceIdentity, PeerStore, TrustedPeer};
use inference_engine::serving::{
    start_isolated_paired, ServingBackend, ServingConfig, CONVERSATION_HEADER,
};
use serde_json::json;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;
use tokio_rustls::TlsConnector;
use tower::ServiceExt;

const KEY: &str = "test-only-key-with-at-least-32-characters";

async fn peer_request(
    address: SocketAddr,
    client: &DeviceIdentity,
    server: &TrustedPeer,
    request: &str,
) -> Result<String, String> {
    tokio::time::timeout(Duration::from_secs(10), async {
        let socket = tokio::net::TcpStream::connect(address)
            .await
            .map_err(|error| error.to_string())?;
        let connector = TlsConnector::from(Arc::new(
            client_config(client, server).map_err(|error| error.to_string())?,
        ));
        let mut stream = connector
            .connect(
                server_name(server).map_err(|error| error.to_string())?,
                socket,
            )
            .await
            .map_err(|error| error.to_string())?;
        stream
            .write_all(request.as_bytes())
            .await
            .map_err(|error| error.to_string())?;
        let mut response = Vec::new();
        stream
            .read_to_end(&mut response)
            .await
            .map_err(|error| error.to_string())?;
        String::from_utf8(response).map_err(|error| error.to_string())
    })
    .await
    .map_err(|_| "peer request timed out".to_string())?
}

#[tokio::test]
#[ignore = "requires SmolLM2-135M-Q4_K_M.gguf in SMOLLM2_DIR"]
async fn paired_peer_serves_real_model_and_rejects_unapproved_device() {
    let directory = PathBuf::from(env::var("SMOLLM2_DIR").expect("set SMOLLM2_DIR"));
    let model = directory.join("SmolLM2-135M-Q4_K_M.gguf");
    let server_dir = tempfile::tempdir().unwrap();
    let approved_dir = tempfile::tempdir().unwrap();
    let unapproved_dir = tempfile::tempdir().unwrap();
    let server = DeviceIdentity::load_or_create(server_dir.path()).unwrap();
    let approved = DeviceIdentity::load_or_create(approved_dir.path()).unwrap();
    let unapproved = DeviceIdentity::load_or_create(unapproved_dir.path()).unwrap();
    let address: SocketAddr = "127.0.0.1:18888".parse().unwrap();
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

    let (local, peer, supervisor) = start_isolated_paired(
        &model,
        load_gguf_tokenizer(&model).unwrap(),
        ServingConfig {
            model_id: "local-smollm2".into(),
            api_key: KEY.into(),
            queue_capacity: 2,
            max_completion_tokens: 4,
            request_timeout: Duration::from_secs(30),
            backend: ServingBackend::Cpu,
        },
        env!("CARGO_BIN_EXE_serve"),
        &server,
        &server_peers,
    )
    .await
    .unwrap();

    let local_response = local
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(local_response.status(), StatusCode::UNAUTHORIZED);

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let peer_address = listener.local_addr().unwrap();
    let handle = axum_server::Handle::new();
    let running = tokio::spawn(peer.serve(listener, handle.clone()));
    let mut trusted_server = approved_peers.peers()[0].clone();
    trusted_server.address = peer_address;
    let client = PeerClient::new(&approved, trusted_server).unwrap();
    let snapshot = client.snapshot().await.unwrap();
    assert_eq!(snapshot.model_id, "local-smollm2");
    assert!(snapshot.ready);
    assert_eq!(snapshot.queue_capacity, 2);
    let models = peer_request(
        peer_address,
        &approved,
        &approved_peers.peers()[0],
        "GET /v1/models HTTP/1.1\r\nHost: peer\r\nConnection: close\r\n\r\n",
    )
    .await
    .unwrap();
    assert!(models.starts_with("HTTP/1.1 200"), "{models}");
    assert!(models.contains("local-smollm2"));

    let body = json!({
        "model": "local-smollm2",
        "messages": [{"role": "user", "content": "Say hi"}],
        "max_completion_tokens": 2
    })
    .to_string();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: peer\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let chat = peer_request(
        peer_address,
        &approved,
        &approved_peers.peers()[0],
        &request,
    )
    .await
    .unwrap();
    assert!(chat.starts_with("HTTP/1.1 200"), "{chat}");
    assert!(chat.contains("Say hi"), "{chat}");
    let forwarded = client
        .forward_chat(body.into_bytes(), false, None, Duration::from_secs(30))
        .await
        .unwrap();
    assert_eq!(forwarded.status(), StatusCode::OK);
    let forwarded_body = axum::body::to_bytes(forwarded.into_body(), 1024 * 1024)
        .await
        .unwrap();
    assert!(String::from_utf8_lossy(&forwarded_body).contains("Say hi"));
    let foreign_id = format!("{}.{}", approved.device_id, uuid::Uuid::new_v4());
    let foreign = client
        .forward_chat(
            json!({
                "model": "local-smollm2",
                "messages": [{"role": "user", "content": "Say hi"}],
                "max_completion_tokens": 2
            })
            .to_string()
            .into_bytes(),
            false,
            Some(&foreign_id),
            Duration::from_secs(30),
        )
        .await
        .unwrap();
    assert_eq!(foreign.status(), StatusCode::CONFLICT);
    let streamed = client
        .forward_chat(
            json!({
                "model": "local-smollm2",
                "messages": [{"role": "user", "content": "Say hi"}],
                "max_completion_tokens": 2,
                "stream": true
            })
            .to_string()
            .into_bytes(),
            true,
            None,
            Duration::from_secs(30),
        )
        .await
        .unwrap();
    assert_eq!(streamed.status(), StatusCode::OK);
    let stream = axum::body::to_bytes(streamed.into_body(), 1024 * 1024)
        .await
        .unwrap();
    assert!(String::from_utf8_lossy(&stream).contains("data: [DONE]"));

    let rejected = peer_request(
        peer_address,
        &unapproved,
        &unapproved_peers.peers()[0],
        "GET /v1/models HTTP/1.1\r\nHost: peer\r\nConnection: close\r\n\r\n",
    )
    .await;
    assert!(!rejected
        .as_ref()
        .is_ok_and(|value| value.starts_with("HTTP/1.1 200")));

    handle.graceful_shutdown(Some(Duration::from_secs(2)));
    tokio::time::timeout(Duration::from_secs(5), running)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    drop(local);
    tokio::time::timeout(Duration::from_secs(5), supervisor)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
#[ignore = "requires SmolLM2-135M-Q4_K_M.gguf in SMOLLM2_DIR"]
async fn serve_cli_binds_loopback_and_approved_peer_listener() {
    let directory = PathBuf::from(env::var("SMOLLM2_DIR").expect("set SMOLLM2_DIR"));
    let model = directory.join("SmolLM2-135M-Q4_K_M.gguf");
    let server_dir = tempfile::tempdir().unwrap();
    let client_dir = tempfile::tempdir().unwrap();
    let server = DeviceIdentity::load_or_create(server_dir.path()).unwrap();
    let client = DeviceIdentity::load_or_create(client_dir.path()).unwrap();
    let reserved_local = TcpListener::bind("127.0.0.1:0").unwrap();
    let local_port = reserved_local.local_addr().unwrap().port();
    let reserved_peer = TcpListener::bind("127.0.0.1:0").unwrap();
    let peer_address = reserved_peer.local_addr().unwrap();
    let mut server_peers = PeerStore::load(server_dir.path()).unwrap();
    server_peers
        .trust(
            &server.device_id,
            &client.offer(),
            peer_address,
            &client.fingerprint,
        )
        .unwrap();
    let mut client_peers = PeerStore::load(client_dir.path()).unwrap();
    client_peers
        .trust(
            &client.device_id,
            &server.offer(),
            peer_address,
            &server.fingerprint,
        )
        .unwrap();
    drop(reserved_local);
    drop(reserved_peer);

    let mut child = Command::new(env!("CARGO_BIN_EXE_serve"))
        .arg("--peer")
        .arg(server_dir.path())
        .arg(peer_address.to_string())
        .arg(&model)
        .arg(local_port.to_string())
        .env("INFERENCE_API_KEY", KEY)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let stdout = child.stdout.take().unwrap();
    let mut lines = BufReader::new(stdout).lines();
    let startup = tokio::time::timeout(Duration::from_secs(15), async {
        let local = lines.next_line().await.unwrap().unwrap();
        let peer = lines.next_line().await.unwrap().unwrap();
        (local, peer)
    })
    .await
    .unwrap();
    assert!(startup.0.contains(&format!("127.0.0.1:{local_port}")));
    assert!(startup.1.contains(&format!("https://{peer_address}")));

    let response = peer_request(
        peer_address,
        &client,
        &client_peers.peers()[0],
        "GET /v1/models HTTP/1.1\r\nHost: peer\r\nConnection: close\r\n\r\n",
    )
    .await
    .unwrap();
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    child.kill().await.unwrap();
    child.wait().await.unwrap();
}

#[tokio::test]
#[ignore = "requires SmolLM2-135M-Q4_K_M.gguf in SMOLLM2_DIR"]
async fn coordinator_uses_peer_after_local_worker_loss_and_reports_peer_loss() {
    let directory = PathBuf::from(env::var("SMOLLM2_DIR").expect("set SMOLLM2_DIR"));
    let model = directory.join("SmolLM2-135M-Q4_K_M.gguf");
    let a_dir = tempfile::tempdir().unwrap();
    let b_dir = tempfile::tempdir().unwrap();
    let a = DeviceIdentity::load_or_create(a_dir.path()).unwrap();
    let b = DeviceIdentity::load_or_create(b_dir.path()).unwrap();
    let b_listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let b_address = b_listener.local_addr().unwrap();
    let mut a_peers = PeerStore::load(a_dir.path()).unwrap();
    a_peers
        .trust(&a.device_id, &b.offer(), b_address, &b.fingerprint)
        .unwrap();
    let mut b_peers = PeerStore::load(b_dir.path()).unwrap();
    b_peers
        .trust(&b.device_id, &a.offer(), b_address, &a.fingerprint)
        .unwrap();
    let config = || ServingConfig {
        model_id: "local-smollm2".into(),
        api_key: KEY.into(),
        queue_capacity: 2,
        max_completion_tokens: 4,
        request_timeout: Duration::from_secs(30),
        backend: ServingBackend::Cpu,
    };
    let (local_a, peer_a, worker_a) = start_isolated_paired(
        &model,
        load_gguf_tokenizer(&model).unwrap(),
        config(),
        env!("CARGO_BIN_EXE_serve"),
        &a,
        &a_peers,
    )
    .await
    .unwrap();
    let (local_b, peer_b, worker_b) = start_isolated_paired(
        &model,
        load_gguf_tokenizer(&model).unwrap(),
        config(),
        env!("CARGO_BIN_EXE_serve"),
        &b,
        &b_peers,
    )
    .await
    .unwrap();
    let handle_b = axum_server::Handle::new();
    let running_b = tokio::spawn(peer_b.serve(b_listener, handle_b.clone()));
    drop(peer_a);
    drop(local_b);
    worker_a.abort();
    assert!(worker_a.await.unwrap_err().is_cancelled());

    let body = json!({
        "model": "local-smollm2",
        "messages": [{"role": "user", "content": "Say hi"}],
        "max_completion_tokens": 2
    })
    .to_string();
    let request = || {
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", format!("Bearer {KEY}"))
            .header("content-type", "application/json")
            .body(Body::from(body.clone()))
            .unwrap()
    };
    let forwarded = local_a.clone().oneshot(request()).await.unwrap();
    assert_eq!(forwarded.status(), StatusCode::OK);
    let content = axum::body::to_bytes(forwarded.into_body(), 1024 * 1024)
        .await
        .unwrap();
    assert!(String::from_utf8_lossy(&content).contains("Say hi"));

    let streaming_body = json!({
        "model": "local-smollm2",
        "messages": [{"role": "user", "content": "Say hi"}],
        "max_completion_tokens": 2,
        "stream": true
    })
    .to_string();
    let streamed = local_a
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", format!("Bearer {KEY}"))
                .header("content-type", "application/json")
                .body(Body::from(streaming_body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(streamed.status(), StatusCode::OK);
    let stream = axum::body::to_bytes(streamed.into_body(), 1024 * 1024)
        .await
        .unwrap();
    assert!(String::from_utf8_lossy(&stream).contains("data: [DONE]"));

    handle_b.graceful_shutdown(Some(Duration::from_secs(2)));
    tokio::time::timeout(Duration::from_secs(5), running_b)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    let lost = local_a.clone().oneshot(request()).await.unwrap();
    assert_eq!(lost.status(), StatusCode::SERVICE_UNAVAILABLE);
    drop(local_a);
    tokio::time::timeout(Duration::from_secs(5), worker_b)
        .await
        .unwrap()
        .unwrap();
}

#[cfg(unix)]
#[tokio::test]
#[ignore = "requires SmolLM2-135M-Q4_K_M.gguf in SMOLLM2_DIR"]
async fn coordinator_sends_next_request_to_idle_peer_while_local_worker_is_busy() {
    use std::fs;
    use std::os::unix::fs::PermissionsExt;

    let directory = PathBuf::from(env::var("SMOLLM2_DIR").expect("set SMOLLM2_DIR"));
    let model = directory.join("SmolLM2-135M-Q4_K_M.gguf");
    let a_dir = tempfile::tempdir().unwrap();
    let b_dir = tempfile::tempdir().unwrap();
    let a = DeviceIdentity::load_or_create(a_dir.path()).unwrap();
    let b = DeviceIdentity::load_or_create(b_dir.path()).unwrap();
    let b_listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let b_address = b_listener.local_addr().unwrap();
    let mut a_peers = PeerStore::load(a_dir.path()).unwrap();
    a_peers
        .trust(&a.device_id, &b.offer(), b_address, &b.fingerprint)
        .unwrap();
    let mut b_peers = PeerStore::load(b_dir.path()).unwrap();
    b_peers
        .trust(&b.device_id, &a.offer(), b_address, &a.fingerprint)
        .unwrap();

    let fake_dir = tempfile::tempdir().unwrap();
    let script = fake_dir.path().join("worker");
    fs::write(
        &script,
        "#!/bin/sh\nprintf '%s\\n' '{\"kind\":\"ready\",\"max_positions\":2048}'\nif [ ! -f \"$2/active\" ]; then\n  IFS= read -r request\n  : > \"$2/active\"\n  exec sleep 60\nfi\nwhile IFS= read -r request; do\n  printf '%s\\n' '{\"kind\":\"delta\",\"text\":\"local\"}'\n  printf '%s\\n' '{\"kind\":\"finished\",\"reason\":\"length\",\"completion_tokens\":1}'\ndone\n",
    )
    .unwrap();
    fs::set_permissions(&script, fs::Permissions::from_mode(0o700)).unwrap();
    let config = || ServingConfig {
        model_id: "local-smollm2".into(),
        api_key: KEY.into(),
        queue_capacity: 2,
        max_completion_tokens: 4,
        request_timeout: Duration::from_secs(10),
        backend: ServingBackend::Cpu,
    };
    let (local_a, peer_a, worker_a) = start_isolated_paired(
        fake_dir.path(),
        load_gguf_tokenizer(&model).unwrap(),
        config(),
        &script,
        &a,
        &a_peers,
    )
    .await
    .unwrap();
    let (local_b, peer_b, worker_b) = start_isolated_paired(
        &model,
        load_gguf_tokenizer(&model).unwrap(),
        config(),
        env!("CARGO_BIN_EXE_serve"),
        &b,
        &b_peers,
    )
    .await
    .unwrap();
    let handle_b = axum_server::Handle::new();
    let running_b = tokio::spawn(peer_b.serve(b_listener, handle_b.clone()));
    drop(peer_a);
    drop(local_b);

    let body = json!({
        "model": "local-smollm2",
        "messages": [{"role": "user", "content": "Say hi"}],
        "max_completion_tokens": 2
    })
    .to_string();
    let request = || {
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", format!("Bearer {KEY}"))
            .header("content-type", "application/json")
            .body(Body::from(body.clone()))
            .unwrap()
    };
    let first_app = local_a.clone();
    let first_request = request();
    let first = tokio::spawn(async move { first_app.oneshot(first_request).await });
    tokio::time::timeout(Duration::from_secs(3), async {
        while !fake_dir.path().join("active").exists() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    let second = tokio::time::timeout(Duration::from_secs(3), local_a.clone().oneshot(request()))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(second.status(), StatusCode::OK);
    let conversation_id = second
        .headers()
        .get(CONVERSATION_HEADER)
        .unwrap()
        .to_str()
        .unwrap()
        .to_owned();
    let content = axum::body::to_bytes(second.into_body(), 1024 * 1024)
        .await
        .unwrap();
    assert!(String::from_utf8_lossy(&content).contains("Say hi"));

    first.abort();
    let _ = first.await;
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            let ready = local_a
                .clone()
                .oneshot(
                    Request::builder()
                        .uri("/health")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            if ready.status() == StatusCode::OK {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap();
    let followup = local_a
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", format!("Bearer {KEY}"))
                .header("content-type", "application/json")
                .header(CONVERSATION_HEADER, &conversation_id)
                .body(Body::from(body.clone()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(followup.status(), StatusCode::OK);
    assert_eq!(
        followup.headers().get(CONVERSATION_HEADER).unwrap(),
        conversation_id.as_str()
    );
    let followup = axum::body::to_bytes(followup.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let followup: serde_json::Value = serde_json::from_slice(&followup).unwrap();
    assert!(
        followup["usage"]["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert_eq!(followup["choices"][0]["message"]["content"], "Say hi");
    handle_b.graceful_shutdown(Some(Duration::from_secs(2)));
    tokio::time::timeout(Duration::from_secs(5), running_b)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    let migrated = local_a
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", format!("Bearer {KEY}"))
                .header("content-type", "application/json")
                .header(CONVERSATION_HEADER, &conversation_id)
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(migrated.status(), StatusCode::OK);
    let migrated_id = migrated
        .headers()
        .get(CONVERSATION_HEADER)
        .unwrap()
        .to_str()
        .unwrap();
    assert_ne!(migrated_id, conversation_id);
    assert!(migrated_id.starts_with(&a.device_id));
    let migrated_body = axum::body::to_bytes(migrated.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let migrated_body: serde_json::Value = serde_json::from_slice(&migrated_body).unwrap();
    assert_eq!(
        migrated_body["usage"]["prompt_tokens_details"]["cached_tokens"],
        0
    );
    assert_eq!(migrated_body["choices"][0]["message"]["content"], "local");
    drop(local_a);
    tokio::time::timeout(Duration::from_secs(5), worker_a)
        .await
        .unwrap()
        .unwrap();
    tokio::time::timeout(Duration::from_secs(5), worker_b)
        .await
        .unwrap()
        .unwrap();
}
