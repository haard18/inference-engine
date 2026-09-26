use std::env;
use std::path::PathBuf;
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use inference_engine::load_gguf_tokenizer;
use inference_engine::serving::{start_isolated, ServingBackend, ServingConfig};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tower::ServiceExt;

const KEY: &str = "test-only-key-with-at-least-32-characters";

#[tokio::test]
#[ignore = "requires SmolLM2-135M-Q4_K_M.gguf in SMOLLM2_DIR"]
async fn complete_worker_rejects_context_beyond_its_cache_budget_before_work() {
    let directory = PathBuf::from(env::var("SMOLLM2_DIR").expect("set SMOLLM2_DIR"));
    let model_path = directory.join("SmolLM2-135M-Q4_K_M.gguf");
    let mut child = Command::new(env!("CARGO_BIN_EXE_serve"))
        .arg("--internal-worker")
        .arg(&model_path)
        .env("INFERENCE_WORKER_CACHE_MIB", "16")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::inherit())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    let mut output = BufReader::new(child.stdout.take().unwrap());
    let mut line = String::new();
    tokio::time::timeout(Duration::from_secs(30), output.read_line(&mut line))
        .await
        .unwrap()
        .unwrap();
    let ready: Value = serde_json::from_str(&line).unwrap();
    assert_eq!(ready["kind"], "ready");
    assert_eq!(ready["max_positions"], 256);

    let oversized = json!({"prompt": vec![1; 256], "max_tokens": 1});
    input
        .write_all(format!("{oversized}\n").as_bytes())
        .await
        .unwrap();
    line.clear();
    tokio::time::timeout(Duration::from_secs(5), output.read_line(&mut line))
        .await
        .unwrap()
        .unwrap();
    let rejected: Value = serde_json::from_str(&line).unwrap();
    assert_eq!(rejected["kind"], "failed");
    assert_eq!(rejected["message"], "worker cache context limit reached");

    input
        .write_all(b"{\"prompt\":[1],\"max_tokens\":1}\n")
        .await
        .unwrap();
    loop {
        line.clear();
        tokio::time::timeout(Duration::from_secs(10), output.read_line(&mut line))
            .await
            .unwrap()
            .unwrap();
        let event: Value = serde_json::from_str(&line).unwrap();
        if event["kind"] == "finished" {
            break;
        }
        assert_eq!(event["kind"], "delta", "{event}");
    }
    drop(input);
    child.wait().await.unwrap();
}

fn request(stream: bool) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("authorization", format!("Bearer {KEY}"))
        .header("content-type", "application/json")
        .body(Body::from(
            json!({
                "model": "local-smollm2",
                "messages": [{"role": "user", "content": "Say hi"}],
                "max_completion_tokens": 2,
                "stream": stream
            })
            .to_string(),
        ))
        .unwrap()
}

fn long_stream_request() -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("authorization", format!("Bearer {KEY}"))
        .header("content-type", "application/json")
        .body(Body::from(
            json!({
                "model": "local-smollm2",
                "messages": [{"role": "user", "content": "Write a long paragraph about the moon."}],
                "max_completion_tokens": 64,
                "stream": true
            })
            .to_string(),
        ))
        .unwrap()
}

#[tokio::test]
#[ignore = "requires SmolLM2-135M-Q4_K_M.gguf in SMOLLM2_DIR"]
async fn real_model_serves_through_reusable_child_process() {
    assert_real_model_serves(ServingBackend::Cpu).await;
}

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore = "requires SmolLM2-135M-Q4_K_M.gguf in SMOLLM2_DIR"]
async fn real_model_serves_through_metal_child_process() {
    assert_real_model_serves(ServingBackend::Metal).await;
}

#[tokio::test]
#[ignore = "requires SmolLM2-135M-Q4_K_M.gguf in SMOLLM2_DIR"]
async fn disconnected_streams_do_not_block_following_real_model_requests() {
    let directory = PathBuf::from(env::var("SMOLLM2_DIR").expect("set SMOLLM2_DIR"));
    let model_path = directory.join("SmolLM2-135M-Q4_K_M.gguf");
    let (app, supervisor) = start_isolated(
        &model_path,
        load_gguf_tokenizer(&model_path).unwrap(),
        ServingConfig {
            model_id: "local-smollm2".into(),
            api_key: KEY.into(),
            queue_capacity: 2,
            max_completion_tokens: 64,
            request_timeout: Duration::from_secs(30),
            backend: ServingBackend::Cpu,
        },
        env!("CARGO_BIN_EXE_serve"),
    )
    .await
    .unwrap();

    for cycle in 0..3 {
        let response = app.clone().oneshot(long_stream_request()).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let mut stream = response.into_body();
        let mut saw_content = false;
        let mut observed = String::new();
        while !saw_content {
            let frame = tokio::time::timeout(Duration::from_secs(20), stream.frame())
                .await
                .unwrap()
                .expect("stream ended before content")
                .unwrap();
            if let Ok(bytes) = frame.into_data() {
                observed.push_str(&String::from_utf8_lossy(&bytes));
                assert!(observed.len() <= 1024 * 1024, "stream exceeded test limit");
                saw_content = observed.contains("\"content\"");
            }
        }
        drop(stream);

        let response = tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                let response = app.clone().oneshot(request(false)).await.unwrap();
                if response.status() == StatusCode::SERVICE_UNAVAILABLE {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    continue;
                }
                break response;
            }
        })
        .await
        .unwrap();
        assert_eq!(response.status(), StatusCode::OK, "cycle {cycle}");
        let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        let completion: Value = serde_json::from_slice(&body).unwrap();
        assert!(completion["choices"][0]["message"]["content"]
            .as_str()
            .is_some());
    }
    drop(app);
    tokio::time::timeout(Duration::from_secs(3), supervisor)
        .await
        .unwrap()
        .unwrap();
}

async fn assert_real_model_serves(backend: ServingBackend) {
    let directory = PathBuf::from(env::var("SMOLLM2_DIR").expect("set SMOLLM2_DIR"));
    let model_path = directory.join("SmolLM2-135M-Q4_K_M.gguf");
    let tokenizer = load_gguf_tokenizer(&model_path).unwrap();
    let (app, supervisor) = start_isolated(
        &model_path,
        tokenizer,
        ServingConfig {
            model_id: "local-smollm2".into(),
            api_key: KEY.into(),
            queue_capacity: 2,
            max_completion_tokens: 4,
            request_timeout: Duration::from_secs(30),
            backend,
        },
        env!("CARGO_BIN_EXE_serve"),
    )
    .await
    .unwrap();

    let response = app.clone().oneshot(request(false)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    let completion: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(completion["choices"][0]["message"]["content"], "Say hi");

    let response = app.clone().oneshot(request(true)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    let stream = String::from_utf8(body.to_vec()).unwrap();
    assert!(stream.contains("chat.completion.chunk"));
    assert!(stream.contains("data: [DONE]"));

    drop(app);
    tokio::time::timeout(Duration::from_secs(3), supervisor)
        .await
        .unwrap()
        .unwrap();
}
