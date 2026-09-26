use std::env;
use std::path::PathBuf;
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use inference_engine::load_gguf_tokenizer;
use inference_engine::serving::{
    start_isolated, ServingBackend, ServingConfig, CONVERSATION_HEADER,
};
use serde_json::{json, Value};
use tower::ServiceExt;

const KEY: &str = "test-only-key-with-at-least-32-characters";

async fn chat(
    app: &axum::Router,
    messages: Value,
    conversation_id: Option<&str>,
    stream: bool,
) -> (StatusCode, Option<String>, Value) {
    let mut builder = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("authorization", format!("Bearer {KEY}"))
        .header("content-type", "application/json");
    if let Some(id) = conversation_id {
        builder = builder.header(CONVERSATION_HEADER, id);
    }
    let response = app
        .clone()
        .oneshot(
            builder
                .body(Body::from(
                    json!({
                        "model": "local-smollm2",
                        "messages": messages,
                        "max_completion_tokens": 2,
                        "stream": stream
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
        .map(|value| value.to_str().unwrap().to_owned());
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    let value = if stream {
        json!({"stream": String::from_utf8(body.to_vec()).unwrap()})
    } else {
        serde_json::from_slice(&body).unwrap()
    };
    (status, id, value)
}

#[tokio::test]
#[ignore = "requires SmolLM2-135M-Q4_K_M.gguf in SMOLLM2_DIR"]
async fn conversation_reuses_exact_prompt_prefix_and_evicts_old_sessions() {
    let directory = PathBuf::from(env::var("SMOLLM2_DIR").expect("set SMOLLM2_DIR"));
    let model = directory.join("SmolLM2-135M-Q4_K_M.gguf");
    let (app, worker) = start_isolated(
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
    )
    .await
    .unwrap();
    let first_messages = json!([{"role": "user", "content": "Say hi"}]);
    let (status, conversation_id, first) = chat(&app, first_messages.clone(), None, false).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(first["usage"]["prompt_tokens_details"]["cached_tokens"], 0);
    let conversation_id = conversation_id.unwrap();

    let (status, next_id, repeated) =
        chat(&app, first_messages.clone(), Some(&conversation_id), false).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(next_id.as_deref(), Some(conversation_id.as_str()));
    assert_eq!(
        repeated["usage"]["prompt_tokens_details"]["cached_tokens"],
        repeated["usage"]["prompt_tokens"]
    );
    assert_eq!(
        repeated["choices"][0]["message"]["content"],
        first["choices"][0]["message"]["content"]
    );

    let extended = json!([
        {"role": "user", "content": "Say hi"},
        {"role": "assistant", "content": "Say hi"},
        {"role": "user", "content": "Say it again"}
    ]);
    let (status, _, extended_response) = chat(&app, extended, Some(&conversation_id), false).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        extended_response["usage"]["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap()
            > 0
    );

    let different = json!([{"role": "user", "content": "Different prompt"}]);
    let (status, _, mismatch) = chat(&app, different, Some(&conversation_id), false).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        mismatch["usage"]["prompt_tokens_details"]["cached_tokens"],
        0
    );

    for _ in 0..8 {
        let (status, _, _) = chat(&app, first_messages.clone(), None, false).await;
        assert_eq!(status, StatusCode::OK);
    }
    let (status, _, evicted) = chat(&app, first_messages, Some(&conversation_id), false).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        evicted["usage"]["prompt_tokens_details"]["cached_tokens"],
        0
    );

    let bad = chat(
        &app,
        json!([{"role": "user", "content": "Hi"}]),
        Some("bad"),
        false,
    )
    .await;
    assert_eq!(bad.0, StatusCode::BAD_REQUEST);
    drop(app);
    tokio::time::timeout(Duration::from_secs(5), worker)
        .await
        .unwrap()
        .unwrap();
}

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore = "requires SmolLM2-135M-Q4_K_M.gguf in SMOLLM2_DIR"]
async fn metal_worker_reuses_conversation_prefix() {
    let directory = PathBuf::from(env::var("SMOLLM2_DIR").expect("set SMOLLM2_DIR"));
    let model = directory.join("SmolLM2-135M-Q4_K_M.gguf");
    let (app, worker) = start_isolated(
        &model,
        load_gguf_tokenizer(&model).unwrap(),
        ServingConfig {
            model_id: "local-smollm2".into(),
            api_key: KEY.into(),
            queue_capacity: 2,
            max_completion_tokens: 4,
            request_timeout: Duration::from_secs(30),
            backend: ServingBackend::Metal,
        },
        env!("CARGO_BIN_EXE_serve"),
    )
    .await
    .unwrap();
    let messages = json!([{"role": "user", "content": "Say hi"}]);
    let (first_status, id, first) = chat(&app, messages.clone(), None, false).await;
    assert_eq!(first_status, StatusCode::OK);
    assert_eq!(first["usage"]["prompt_tokens_details"]["cached_tokens"], 0);
    let (second_status, _, second) = chat(&app, messages, id.as_deref(), false).await;
    assert_eq!(second_status, StatusCode::OK);
    assert_eq!(
        second["usage"]["prompt_tokens_details"]["cached_tokens"],
        second["usage"]["prompt_tokens"]
    );
    assert_eq!(
        second["choices"][0]["message"]["content"],
        first["choices"][0]["message"]["content"]
    );
    drop(app);
    tokio::time::timeout(Duration::from_secs(5), worker)
        .await
        .unwrap()
        .unwrap();
}
