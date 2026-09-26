//! Measure the local API while it routes work across its approved device pool.

use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::net::SocketAddr;
use std::process;
use std::time::{Duration, Instant};

use axum::body::{to_bytes, Body, Bytes};
use axum::http::{Request, StatusCode};
use http_body_util::{BodyExt, Full};
use hyper::client::conn::http1;
use hyper_util::rt::TokioIo;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
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
    first_content: Option<Duration>,
    completed: bool,
    stream_error: Option<String>,
    completion_digest: Option<String>,
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
    let streaming = args.len() == 7 && args[6] == "--stream";
    if args.len() != 6 && !streaming {
        return Err(format!(
            "usage: {} LOOPBACK_IP:PORT MODEL_ID REQUESTS CONCURRENCY MAX_TOKENS [--stream]",
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
    let system_prompt = match env::var("INFERENCE_BENCH_SYSTEM_PROMPT") {
        Ok(prompt) if !prompt.is_empty() && prompt.len() <= 4096 => Some(prompt),
        Ok(_) => return Err("benchmark system prompt must contain 1..4096 bytes".into()),
        Err(env::VarError::NotPresent) => None,
        Err(env::VarError::NotUnicode(_)) => {
            return Err("benchmark system prompt must be text".into())
        }
    };
    let system_prompt_sha256 = system_prompt
        .as_ref()
        .map(|prompt| hex::encode(Sha256::digest(prompt.as_bytes())));
    let mut messages = Vec::with_capacity(2);
    if let Some(prompt) = &system_prompt {
        messages.push(json!({"role": "system", "content": prompt}));
    }
    messages.push(
        json!({"role": "user", "content": "Give a short factual answer: what is two plus two?"}),
    );
    let body = json!({
        "model": model_id,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "n": 1,
        "stream": streaming
    })
    .to_string();

    // Warm the model and connection path before the measured interval.
    for _ in 0..2 {
        let warm = request_once(address, &key, &body, streaming).await?;
        if !warm.completed {
            return Err(format!(
                "warm-up did not finish successfully: HTTP {} {:?}",
                warm.status, warm.stream_error
            )
            .into());
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
            active.spawn(async move { request_once(address, &key, &body, streaming).await });
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
        .filter(|sample| sample.completed)
        .map(|sample| sample.latency.as_secs_f64() * 1000.0)
        .collect::<Vec<_>>();
    latencies.sort_by(f64::total_cmp);
    let mut owners = BTreeMap::<String, usize>::new();
    let mut statuses = BTreeMap::<String, usize>::new();
    let mut application_errors = BTreeMap::<String, usize>::new();
    let mut completion_digests = BTreeMap::<String, usize>::new();
    let mut completion_tokens = 0usize;
    let mut first_content = Vec::new();
    for sample in &samples {
        *statuses
            .entry(sample.status.as_u16().to_string())
            .or_default() += 1;
        *owners
            .entry(sample.owner.clone().unwrap_or_else(|| "unknown".into()))
            .or_default() += 1;
        if let Some(error) = &sample.stream_error {
            *application_errors.entry(error.clone()).or_default() += 1;
        }
        if sample.completed {
            completion_tokens += sample.completion_tokens;
            if let Some(digest) = &sample.completion_digest {
                *completion_digests.entry(digest.clone()).or_default() += 1;
            }
            if let Some(time) = sample.first_content {
                first_content.push(time.as_secs_f64() * 1000.0);
            }
        }
    }
    first_content.sort_by(f64::total_cmp);
    let successful = samples.iter().filter(|sample| sample.completed).count();
    println!(
        "{}",
        json!({
            "schema_version": 1,
            "endpoint": address.to_string(),
            "model": model_id,
            "system_prompt_sha256": system_prompt_sha256,
            "mode": if streaming { "stream" } else { "ordinary" },
            "requested": requests,
            "concurrency": concurrency,
            "max_completion_tokens": max_tokens,
            "warmup_requests": 2,
            "successful": successful,
            "transport_failures": failures,
            "application_errors": application_errors,
            "completion_digests": completion_digests,
            "http_statuses": statuses,
            "requests_by_device": owners,
            "elapsed_seconds": elapsed.as_secs_f64(),
            "requests_per_second": successful as f64 / elapsed.as_secs_f64(),
            "completion_tokens_per_second": if streaming { None } else { Some(completion_tokens as f64 / elapsed.as_secs_f64()) },
            "p50_ms": percentile(&latencies, 0.50),
            "p95_ms": percentile(&latencies, 0.95),
            "first_content_p50_ms": percentile(&first_content, 0.50),
            "first_content_p95_ms": percentile(&first_content, 0.95),
            "responses_with_content": first_content.len()
        })
    );
    Ok(())
}

#[derive(Default)]
struct StreamState {
    pending: Vec<u8>,
    first_content: Option<Duration>,
    finished: bool,
    done: bool,
    error: Option<String>,
    digest: Sha256,
}

impl StreamState {
    fn feed(&mut self, bytes: &[u8], elapsed: Duration) -> Result<(), String> {
        self.pending.extend_from_slice(bytes);
        while let Some((end, delimiter_bytes)) = event_boundary(&self.pending) {
            let event = self
                .pending
                .drain(..end + delimiter_bytes)
                .collect::<Vec<_>>();
            for line in event.split(|byte| *byte == b'\n') {
                let line = line.strip_suffix(b"\r").unwrap_or(line);
                let Some(data) = line.strip_prefix(b"data: ") else {
                    continue;
                };
                if data == b"[DONE]" {
                    self.done = true;
                    continue;
                }
                let value: Value =
                    serde_json::from_slice(data).map_err(|error| error.to_string())?;
                if let Some(error) = value.get("error") {
                    self.error = Some(
                        error
                            .get("code")
                            .and_then(Value::as_str)
                            .unwrap_or("stream_error")
                            .to_owned(),
                    );
                    continue;
                }
                let choice = &value["choices"][0];
                if let Some(content) = choice["delta"]["content"].as_str() {
                    if !content.is_empty() {
                        self.first_content.get_or_insert(elapsed);
                        self.digest.update(content.as_bytes());
                    }
                }
                if choice["finish_reason"].as_str().is_some() {
                    self.finished = true;
                }
            }
        }
        Ok(())
    }

    fn outcome(self) -> (bool, Option<String>, Option<Duration>, Option<String>) {
        let completed =
            self.finished && self.done && self.error.is_none() && self.pending.is_empty();
        let error = if completed {
            None
        } else {
            Some(self.error.unwrap_or_else(|| "incomplete_stream".into()))
        };
        let digest = completed.then(|| hex::encode(self.digest.finalize()));
        (completed, error, self.first_content, digest)
    }
}

fn event_boundary(bytes: &[u8]) -> Option<(usize, usize)> {
    let lf = bytes.windows(2).position(|pair| pair == b"\n\n");
    let crlf = bytes.windows(4).position(|pair| pair == b"\r\n\r\n");
    match (lf, crlf) {
        (Some(left), Some(right)) if left < right => Some((left, 2)),
        (_, Some(right)) => Some((right, 4)),
        (Some(left), None) => Some((left, 2)),
        (None, None) => None,
    }
}

async fn request_once(
    address: SocketAddr,
    key: &str,
    body: &str,
    streaming: bool,
) -> Result<Sample, String> {
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
        let (completion_tokens, first_content, completed, stream_error, completion_digest) =
            if streaming && status == StatusCode::OK {
                let mut stream_body = response.into_body();
                let mut state = StreamState::default();
                let mut total_bytes = 0usize;
                while let Some(frame) = stream_body.frame().await {
                    let frame = frame.map_err(|error| error.to_string())?;
                    if let Ok(data) = frame.into_data() {
                        total_bytes = total_bytes
                            .checked_add(data.len())
                            .filter(|total| *total <= MAX_RESPONSE_BYTES)
                            .ok_or_else(|| "stream response exceeds size limit".to_owned())?;
                        state.feed(&data, started.elapsed())?;
                    }
                }
                let (completed, stream_error, first_content, digest) = state.outcome();
                (0, first_content, completed, stream_error, digest)
            } else {
                let bytes = to_bytes(Body::new(response.into_body()), MAX_RESPONSE_BYTES)
                    .await
                    .map_err(|error| error.to_string())?;
                let value: Value =
                    serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
                let completion_tokens = if status == StatusCode::OK {
                    value
                        .get("usage")
                        .and_then(|usage| usage.get("completion_tokens"))
                        .and_then(Value::as_u64)
                        .ok_or_else(|| {
                            "successful response has no completion-token count".to_owned()
                        })? as usize
                } else {
                    0
                };
                let digest = if status == StatusCode::OK {
                    let content = value["choices"][0]["message"]["content"]
                        .as_str()
                        .ok_or_else(|| "successful response has no completion text".to_owned())?;
                    Some(hex::encode(Sha256::digest(content.as_bytes())))
                } else {
                    None
                };
                (
                    completion_tokens,
                    None,
                    status == StatusCode::OK,
                    None,
                    digest,
                )
            };
        Ok(Sample {
            latency: started.elapsed(),
            status,
            owner,
            completion_tokens,
            first_content,
            completed,
            stream_error,
            completion_digest,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_parser_counts_only_completed_streams_and_records_first_content() {
        let mut stream = StreamState::default();
        stream
            .feed(b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n", Duration::from_millis(10))
            .unwrap();
        stream
            .feed(
                b"data: {\"choices\":[{\"delta\":{\"content\":\"he",
                Duration::from_millis(20),
            )
            .unwrap();
        stream
            .feed(
                b"llo\"},\"finish_reason\":null}]}\r\n\r\n",
                Duration::from_millis(30),
            )
            .unwrap();
        stream
            .feed(b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\ndata: [DONE]\n\n", Duration::from_millis(40))
            .unwrap();
        assert_eq!(
            stream.outcome(),
            (
                true,
                None,
                Some(Duration::from_millis(30)),
                Some(hex::encode(Sha256::digest(b"hello")))
            )
        );

        let mut failed = StreamState::default();
        failed
            .feed(
                b"data: {\"error\":{\"code\":\"inference_failed\"}}\n\ndata: [DONE]\n\n",
                Duration::from_millis(20),
            )
            .unwrap();
        assert_eq!(
            failed.outcome(),
            (false, Some("inference_failed".into()), None, None)
        );

        let mut truncated = StreamState::default();
        truncated
            .feed(
                b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
                Duration::from_millis(20),
            )
            .unwrap();
        assert_eq!(
            truncated.outcome(),
            (false, Some("incomplete_stream".into()), None, None)
        );
    }
}
