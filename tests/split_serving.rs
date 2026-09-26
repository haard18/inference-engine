use std::env;
use std::net::TcpListener;
use std::path::PathBuf;
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use inference_engine::load_gguf_tokenizer;
use inference_engine::pool::client::PeerClient;
use inference_engine::pool::{DeviceIdentity, PeerStore};
use inference_engine::serving::{
    start_isolated, start_split_prefix, start_stage_peer, ServingBackend, ServingConfig,
    CONVERSATION_HEADER,
};
use serde_json::{json, Value};
use tower::ServiceExt;

const KEY: &str = "split-serving-test-key-with-32-characters";

fn config() -> ServingConfig {
    ServingConfig {
        model_id: "local-smollm2".into(),
        api_key: KEY.into(),
        queue_capacity: 4,
        max_completion_tokens: 16,
        request_timeout: Duration::from_secs(60),
        backend: ServingBackend::Cpu,
    }
}

async fn chat(router: axum::Router, stream: bool) -> (StatusCode, String, Option<String>) {
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("authorization", format!("Bearer {KEY}"))
        .header("content-type", "application/json")
        .body(Body::from(
            json!({
                "model": "local-smollm2",
                "messages": [{"role": "user", "content": "Say hello."}],
                "max_tokens": 5,
                "stream": stream,
                "temperature": 0
            })
            .to_string(),
        ))
        .unwrap();
    let response = router.oneshot(request).await.unwrap();
    let status = response.status();
    let conversation_id = response
        .headers()
        .get(CONVERSATION_HEADER)
        .map(|value| value.to_str().unwrap().to_owned());
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    (
        status,
        String::from_utf8(body.to_vec()).unwrap(),
        conversation_id,
    )
}

async fn conversation_chat(
    router: axum::Router,
    messages: Value,
    conversation_id: Option<&str>,
) -> (StatusCode, String, Value) {
    let mut request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("authorization", format!("Bearer {KEY}"))
        .header("content-type", "application/json");
    if let Some(id) = conversation_id {
        request = request.header(CONVERSATION_HEADER, id);
    }
    let response = router
        .oneshot(
            request
                .body(Body::from(
                    json!({
                        "model": "local-smollm2",
                        "messages": messages,
                        "max_tokens": 5,
                        "temperature": 0
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let id = response
        .headers()
        .get(CONVERSATION_HEADER)
        .unwrap()
        .to_str()
        .unwrap()
        .to_owned();
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    (status, id, serde_json::from_slice(&body).unwrap())
}

fn stream_content(body: &str) -> String {
    let mut content = String::new();
    for line in body.lines().filter_map(|line| line.strip_prefix("data: ")) {
        if line == "[DONE]" {
            continue;
        }
        let value: Value = serde_json::from_str(line).unwrap();
        if let Some(text) = value["choices"][0]["delta"]["content"].as_str() {
            content.push_str(text);
        }
    }
    content
}

async fn wait_for_health(router: axum::Router, expected: StatusCode) {
    tokio::time::timeout(Duration::from_secs(8), async {
        loop {
            let response = router
                .clone()
                .oneshot(
                    Request::builder()
                        .uri("/health")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            if response.status() == expected {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("split health did not become {expected}"));
}

#[tokio::test]
#[ignore = "requires SmolLM2-135M-Q4_K_M.gguf in SMOLLM2_DIR"]
async fn split_chat_matches_whole_model_and_reports_peer_loss() {
    let directory = PathBuf::from(env::var("SMOLLM2_DIR").expect("set SMOLLM2_DIR"));
    let model = directory.join("SmolLM2-135M-Q4_K_M.gguf");
    let executable = env!("CARGO_BIN_EXE_serve");
    let suffix_dir = tempfile::tempdir().unwrap();
    let prefix_dir = tempfile::tempdir().unwrap();
    let suffix_identity = DeviceIdentity::load_or_create(suffix_dir.path()).unwrap();
    let prefix_identity = DeviceIdentity::load_or_create(prefix_dir.path()).unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let mut suffix_peers = PeerStore::load(suffix_dir.path()).unwrap();
    suffix_peers
        .trust(
            &suffix_identity.device_id,
            &prefix_identity.offer(),
            address,
            &prefix_identity.fingerprint,
        )
        .unwrap();
    let mut prefix_peers = PeerStore::load(prefix_dir.path()).unwrap();
    prefix_peers
        .trust(
            &prefix_identity.device_id,
            &suffix_identity.offer(),
            address,
            &suffix_identity.fingerprint,
        )
        .unwrap();
    let suffix = start_stage_peer(
        &model,
        executable,
        15,
        30,
        4,
        &suffix_identity,
        &suffix_peers,
    )
    .await
    .unwrap();
    let suffix_handle = axum_server::Handle::new();
    let handle = suffix_handle.clone();
    let suffix_task = tokio::spawn(async move { suffix.serve(listener, handle).await });
    let client = PeerClient::new(&prefix_identity, prefix_peers.peers()[0].clone()).unwrap();
    let inspector = client.clone();
    let mismatched = start_split_prefix(
        &model,
        load_gguf_tokenizer(&model).unwrap(),
        config(),
        executable,
        14,
        &prefix_identity,
        client.clone(),
    )
    .await;
    assert!(mismatched.is_err(), "a nonadjacent stage pair was accepted");
    let (split, split_worker) = start_split_prefix(
        &model,
        load_gguf_tokenizer(&model).unwrap(),
        config(),
        executable,
        15,
        &prefix_identity,
        client,
    )
    .await
    .unwrap();
    let (whole, whole_worker) = start_isolated(
        &model,
        load_gguf_tokenizer(&model).unwrap(),
        config(),
        executable,
    )
    .await
    .unwrap();
    let (whole_status, whole_body, _) = chat(whole, false).await;
    let (split_status, split_body, _) = chat(split.clone(), false).await;
    assert_eq!(whole_status, StatusCode::OK, "{whole_body}");
    assert_eq!(split_status, StatusCode::OK, "{split_body}");
    let whole_json: Value = serde_json::from_str(&whole_body).unwrap();
    let split_json: Value = serde_json::from_str(&split_body).unwrap();
    assert_eq!(split_json["choices"], whole_json["choices"]);
    assert_eq!(split_json["usage"], whole_json["usage"]);
    let (stream_status, stream_body, stream_id) = chat(split.clone(), true).await;
    assert_eq!(stream_status, StatusCode::OK, "{stream_body}");
    assert!(stream_body.contains("data: [DONE]"));
    assert_eq!(
        stream_content(&stream_body),
        split_json["choices"][0]["message"]["content"]
    );
    let messages = json!([{"role": "user", "content": "Say hello."}]);
    let (status, _, after_stream) =
        conversation_chat(split.clone(), messages.clone(), stream_id.as_deref()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        after_stream["usage"]["prompt_tokens_details"]["cached_tokens"],
        after_stream["usage"]["prompt_tokens"]
    );
    let (status, conversation_id, initial) =
        conversation_chat(split.clone(), messages.clone(), None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        initial["usage"]["prompt_tokens_details"]["cached_tokens"],
        0
    );
    let (status, same_id, repeated) =
        conversation_chat(split.clone(), messages.clone(), Some(&conversation_id)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(same_id, conversation_id);
    assert_eq!(repeated["choices"], initial["choices"]);
    assert_eq!(
        repeated["usage"]["prompt_tokens_details"]["cached_tokens"],
        repeated["usage"]["prompt_tokens"]
    );
    let extended = json!([
        {"role": "user", "content": "Say hello."},
        {"role": "assistant", "content": "Hello."},
        {"role": "user", "content": "Say it again."}
    ]);
    let (status, _, extended_response) =
        conversation_chat(split.clone(), extended, Some(&conversation_id)).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        extended_response["usage"]["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap()
            > 0
    );
    let changed = json!([{"role": "user", "content": "Different prompt."}]);
    let (status, _, changed_response) =
        conversation_chat(split.clone(), changed.clone(), Some(&conversation_id)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        changed_response["usage"]["prompt_tokens_details"]["cached_tokens"],
        0
    );
    suffix_handle.graceful_shutdown(Some(Duration::from_secs(1)));
    suffix_task.await.unwrap().unwrap();
    wait_for_health(split.clone(), StatusCode::SERVICE_UNAVAILABLE).await;
    let (lost_status, lost_body, _) = chat(split.clone(), false).await;
    assert_eq!(lost_status, StatusCode::SERVICE_UNAVAILABLE, "{lost_body}");
    assert!(lost_body.contains("worker_unavailable"), "{lost_body}");
    let listener = TcpListener::bind(address).unwrap();
    let suffix = start_stage_peer(
        &model,
        executable,
        15,
        30,
        4,
        &suffix_identity,
        &suffix_peers,
    )
    .await
    .unwrap();
    let handle = axum_server::Handle::new();
    let suffix_task = tokio::spawn(async move { suffix.serve(listener, handle).await });
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if inspector
                .stage_snapshot()
                .await
                .is_ok_and(|snapshot| snapshot.ready)
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .unwrap();
    wait_for_health(split.clone(), StatusCode::OK).await;
    let (status, _, after_restart) =
        conversation_chat(split.clone(), changed, Some(&conversation_id)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        after_restart["usage"]["prompt_tokens_details"]["cached_tokens"],
        0
    );
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("authorization", format!("Bearer {KEY}"))
        .header("content-type", "application/json")
        .body(Body::from(
            json!({
                "model": "local-smollm2",
                "messages": [{"role": "user", "content": "Write a long paragraph about the moon."}],
                "max_tokens": 16,
                "stream": true,
                "temperature": 0
            })
            .to_string(),
        ))
        .unwrap();
    let response = split.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body();
    let mut observed = String::new();
    while !observed.contains("\"content\"") {
        let frame = tokio::time::timeout(Duration::from_secs(20), body.frame())
            .await
            .unwrap()
            .expect("stream ended before content")
            .unwrap();
        observed.push_str(&String::from_utf8_lossy(&frame.into_data().unwrap()));
    }
    suffix_task.abort();
    let _ = suffix_task.await;
    while let Some(frame) = tokio::time::timeout(Duration::from_secs(20), body.frame())
        .await
        .unwrap()
    {
        observed.push_str(&String::from_utf8_lossy(
            &frame.unwrap().into_data().unwrap(),
        ));
    }
    assert!(observed.contains("inference_failed"), "{observed}");
    assert!(observed.contains("data: [DONE]"), "{observed}");
    wait_for_health(split.clone(), StatusCode::SERVICE_UNAVAILABLE).await;
    let (lost_status, lost_body, _) = chat(split.clone(), false).await;
    assert_eq!(lost_status, StatusCode::SERVICE_UNAVAILABLE, "{lost_body}");
    assert!(lost_body.contains("worker_unavailable"), "{lost_body}");
    split_worker.abort();
    whole_worker.abort();
}
