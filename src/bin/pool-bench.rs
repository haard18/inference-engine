//! Measure the local API while it routes work across its approved device pool.

use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::net::SocketAddr;
use std::process;
use std::time::{Duration, Instant};

use axum::body::{to_bytes, Body, Bytes};
use axum::http::{Request, StatusCode};
use http_body_util::Full;
use hyper::client::conn::http1;
use hyper_util::rt::TokioIo;
use serde_json::{json, Value};
use tokio::net::TcpStream;
use tokio::task::JoinSet;

use inference_engine::serving::CONVERSATION_HEADER;

const MAX_RESPONSE_BYTES: usize = 2 * 1024 * 1024;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(130);

struct Sample {
    latency: Duration,
    status: StatusCode,
    owner: Option<String>,
    completion_tokens: usize,
}

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("{error}");
        process::exit(1);
    }
}

async fn run() -> Result<(), Box<dyn Error + Send + Sync>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 6 {
        return Err(format!(
            "usage: {} LOOPBACK_IP:PORT MODEL_ID REQUESTS CONCURRENCY MAX_TOKENS",
            args[0]
        )
        .into());
    }
    let address: SocketAddr = args[1].parse()?;
    if !address.ip().is_loopback() || address.port() == 0 {
        return Err("benchmark endpoint must be a loopback IP and nonzero port".into());
    }
    let model_id = args[2].clone();
    if model_id.is_empty() {
        return Err("model ID is empty".into());
    }
    let requests = args[3].parse::<usize>()?;
    let concurrency = args[4].parse::<usize>()?;
    let max_tokens = args[5].parse::<usize>()?;
    if requests == 0 || requests > 10_000 || concurrency == 0 || concurrency > 128 {
        return Err("requests must be 1..10000 and concurrency must be 1..128".into());
    }
    if max_tokens == 0 || max_tokens > 4096 {
        return Err("max tokens must be 1..4096".into());
    }
    let key = env::var("INFERENCE_API_KEY")
        .map_err(|_| "set INFERENCE_API_KEY to the local server's Bearer key")?;
    let body = json!({
        "model": model_id,
        "messages": [{"role": "user", "content": "Give a short factual answer: what is two plus two?"}],
        "max_completion_tokens": max_tokens
    })
    .to_string();

    // Warm the model and connection path before the measured interval.
    for _ in 0..2 {
        let warm = request_once(address, &key, &body).await?;
        if warm.status != StatusCode::OK {
            return Err(format!("warm-up returned HTTP {}", warm.status).into());
        }
    }

    let started = Instant::now();
    let mut samples = Vec::with_capacity(requests);
    let mut failures = BTreeMap::<String, usize>::new();
    let mut active = JoinSet::new();
    let mut launched = 0usize;
    while launched < requests || !active.is_empty() {
        while launched < requests && active.len() < concurrency {
            let key = key.clone();
            let body = body.clone();
            active.spawn(async move { request_once(address, &key, &body).await });
            launched += 1;
        }
        if let Some(result) = active.join_next().await {
            match result {
                Ok(Ok(sample)) => samples.push(sample),
                Ok(Err(error)) => *failures.entry(error).or_default() += 1,
                Err(error) => *failures.entry(format!("task failed: {error}")).or_default() += 1,
            }
        }
    }
    let elapsed = started.elapsed();
    let mut latencies = samples
        .iter()
        .map(|sample| sample.latency.as_secs_f64() * 1000.0)
        .collect::<Vec<_>>();
    latencies.sort_by(f64::total_cmp);
    let mut owners = BTreeMap::<String, usize>::new();
    let mut statuses = BTreeMap::<String, usize>::new();
    let mut completion_tokens = 0usize;
    for sample in &samples {
        *statuses
            .entry(sample.status.as_u16().to_string())
            .or_default() += 1;
        *owners
            .entry(sample.owner.clone().unwrap_or_else(|| "unknown".into()))
            .or_default() += 1;
        if sample.status == StatusCode::OK {
            completion_tokens += sample.completion_tokens;
        }
    }
    let successful = samples
        .iter()
        .filter(|sample| sample.status == StatusCode::OK)
        .count();
    println!(
        "{}",
        json!({
            "endpoint": address.to_string(),
            "model": model_id,
            "requested": requests,
            "concurrency": concurrency,
            "successful": successful,
            "transport_failures": failures,
            "http_statuses": statuses,
            "requests_by_device": owners,
            "elapsed_seconds": elapsed.as_secs_f64(),
            "requests_per_second": successful as f64 / elapsed.as_secs_f64(),
            "completion_tokens_per_second": completion_tokens as f64 / elapsed.as_secs_f64(),
            "p50_ms": percentile(&latencies, 0.50),
            "p95_ms": percentile(&latencies, 0.95)
        })
    );
    Ok(())
}

async fn request_once(address: SocketAddr, key: &str, body: &str) -> Result<Sample, String> {
    tokio::time::timeout(REQUEST_TIMEOUT, async {
        let started = Instant::now();
        let stream = TcpStream::connect(address)
            .await
            .map_err(|error| error.to_string())?;
        let (mut sender, connection) = http1::handshake(TokioIo::new(stream))
            .await
            .map_err(|error| error.to_string())?;
        tokio::spawn(async move {
            let _ = connection.await;
        });
        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("host", address.to_string())
            .header("authorization", format!("Bearer {key}"))
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(body.to_owned())))
            .map_err(|error| error.to_string())?;
        let response = sender
            .send_request(request)
            .await
            .map_err(|error| error.to_string())?;
        let status = response.status();
        let owner = response
            .headers()
            .get(CONVERSATION_HEADER)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.split_once('.'))
            .map(|(owner, _)| owner.to_owned());
        let bytes = to_bytes(Body::new(response.into_body()), MAX_RESPONSE_BYTES)
            .await
            .map_err(|error| error.to_string())?;
        let value: Value = serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
        let completion_tokens = if status == StatusCode::OK {
            value
                .get("usage")
                .and_then(|usage| usage.get("completion_tokens"))
                .and_then(Value::as_u64)
                .ok_or_else(|| "successful response has no completion-token count".to_owned())?
                as usize
        } else {
            0
        };
        Ok(Sample {
            latency: started.elapsed(),
            status,
            owner,
            completion_tokens,
        })
    })
    .await
    .map_err(|_| "request timed out".to_owned())?
}

fn percentile(sorted: &[f64], fraction: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let index = ((sorted.len() - 1) as f64 * fraction).ceil() as usize;
    Some(sorted[index])
}
