use std::env;
use std::net::{SocketAddr, TcpListener};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use inference_engine::load_gguf_tokenizer;
use inference_engine::pool::tls::{client_config, server_name};
use inference_engine::pool::{DeviceIdentity, PeerStore, TrustedPeer};
use inference_engine::serving::{start_isolated_paired, ServingBackend, ServingConfig};
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
