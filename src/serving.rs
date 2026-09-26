//! A bounded, authenticated, loopback serving layer for the supported text model.

use std::convert::Infallible;
use std::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc as std_mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::extract::{DefaultBodyLimit, State};
use axum::http::header::{AUTHORIZATION, CACHE_CONTROL, WWW_AUTHENTICATE};
use axum::http::{HeaderValue, StatusCode};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};
use subtle::ConstantTimeEq;
use tokio::runtime::Handle;
use tokio::sync::mpsc;
use tokio_stream::{wrappers::ReceiverStream, StreamExt};
use uuid::Uuid;

#[cfg(target_os = "macos")]
use crate::MetalRuntime;
use crate::{ByteBpeDecoder, ByteBpeTokenizer, GenerationSession, Model};

mod isolated;
pub use isolated::{run_worker_stdio, start_isolated};

const MAX_BODY_BYTES: usize = 64 * 1024;
const OUTPUT_CHANNEL_CAPACITY: usize = 8;
const SLOW_CLIENT_TIMEOUT: Duration = Duration::from_secs(15);
const MIN_REQUEST_TIMEOUT: Duration = Duration::from_millis(10);
const MAX_REQUEST_TIMEOUT: Duration = Duration::from_secs(600);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServingBackend {
    Cpu,
    Metal,
}

pub struct ServingConfig {
    pub model_id: String,
    pub api_key: String,
    pub queue_capacity: usize,
    pub max_completion_tokens: usize,
    pub request_timeout: Duration,
    pub backend: ServingBackend,
}

#[derive(Debug)]
pub enum ServingError {
    Configuration(&'static str),
    Worker(String),
}

impl fmt::Display for ServingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Configuration(message) => write!(f, "invalid serving configuration: {message}"),
            Self::Worker(message) => write!(f, "inference worker failed to start: {message}"),
        }
    }
}

impl std::error::Error for ServingError {}

struct AppState {
    model_id: String,
    api_key: Vec<u8>,
    tokenizer: Arc<ByteBpeTokenizer>,
    max_positions: usize,
    max_completion_tokens: usize,
    request_timeout: Duration,
    worker_status: Arc<WorkerStatus>,
    requests: mpsc::Sender<Job>,
}

#[derive(Default)]
struct WorkerStatus {
    active_deadline: Mutex<Option<Instant>>,
    unavailable: AtomicBool,
}

impl WorkerStatus {
    fn start(&self, deadline: Instant) {
        if let Ok(mut active) = self.active_deadline.lock() {
            *active = Some(deadline);
        }
    }

    fn clear(&self) {
        if let Ok(mut active) = self.active_deadline.lock() {
            *active = None;
        }
    }

    fn overdue(&self) -> bool {
        self.active_deadline.lock().map_or(true, |active| {
            active.is_some_and(|deadline| Instant::now() >= deadline)
        })
    }

    fn unavailable(&self) -> bool {
        self.unavailable.load(Ordering::Acquire) || self.overdue()
    }

    fn set_unavailable(&self, unavailable: bool) {
        self.unavailable.store(unavailable, Ordering::Release);
    }
}

struct ActiveJob(Arc<WorkerStatus>);

impl ActiveJob {
    fn new(status: Arc<WorkerStatus>, deadline: Instant) -> Self {
        status.start(deadline);
        Self(status)
    }
}

impl Drop for ActiveJob {
    fn drop(&mut self) {
        self.0.clear();
    }
}

struct Job {
    prompt: Vec<usize>,
    max_tokens: usize,
    deadline: Instant,
    output: mpsc::Sender<WorkerEvent>,
}

enum WorkerEvent {
    Started,
    Delta(String),
    Finished {
        reason: &'static str,
        completion_tokens: usize,
    },
    Failed(String),
    TimedOut,
    End,
}

enum JobFailure {
    Execution(String),
    Deadline,
    ClientGone,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    stream: bool,
    max_tokens: Option<usize>,
    max_completion_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    n: Option<usize>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatMessage {
    role: String,
    content: String,
}

/// Start one bounded inference worker. Call this from an active Tokio runtime.
pub fn start(
    model: Model,
    tokenizer: ByteBpeTokenizer,
    config: ServingConfig,
) -> Result<(Router, JoinHandle<()>), ServingError> {
    let (state, worker) = start_state(model, tokenizer, config)?;
    Ok((router(state), worker))
}

fn start_state(
    model: Model,
    tokenizer: ByteBpeTokenizer,
    config: ServingConfig,
) -> Result<(Arc<AppState>, JoinHandle<()>), ServingError> {
    validate_config(&config)?;
    let handle = Handle::try_current()
        .map_err(|_| ServingError::Configuration("a Tokio runtime is required"))?;
    let model = Arc::new(model);
    let tokenizer = Arc::new(tokenizer);
    let (sender, mut receiver) = mpsc::channel::<Job>(config.queue_capacity);
    let worker_status = Arc::new(WorkerStatus::default());
    let (ready_sender, ready_receiver) = std_mpsc::sync_channel(1);
    let worker_model = Arc::clone(&model);
    let worker_tokenizer = Arc::clone(&tokenizer);
    let worker_status_for_thread = Arc::clone(&worker_status);
    let mode = config.backend;
    let worker = thread::Builder::new()
        .name("inference-engine-worker".into())
        .spawn(move || {
            #[cfg(target_os = "macos")]
            let mut metal = if mode == ServingBackend::Metal {
                match MetalRuntime::new(&worker_model) {
                    Ok(runtime) => Some(runtime),
                    Err(error) => {
                        let _ = ready_sender.send(Err(error.to_string()));
                        return;
                    }
                }
            } else {
                None
            };
            #[cfg(not(target_os = "macos"))]
            let _ = mode;
            if ready_sender.send(Ok(())).is_err() {
                return;
            }
            while let Some(job) = receiver.blocking_recv() {
                if job.output.is_closed() {
                    continue;
                }
                let _active = ActiveJob::new(Arc::clone(&worker_status_for_thread), job.deadline);
                let result = catch_unwind(AssertUnwindSafe(|| {
                    #[cfg(target_os = "macos")]
                    let mut session = match metal.as_ref() {
                        Some(runtime) => runtime.session(),
                        None => GenerationSession::new(&worker_model),
                    };
                    #[cfg(not(target_os = "macos"))]
                    let mut session = GenerationSession::new(&worker_model);
                    process_job(&job, &mut session, &worker_tokenizer, &handle)
                }));
                match result {
                    Ok(Ok((reason, completion_tokens))) => {
                        if send_event(
                            &handle,
                            &job.output,
                            WorkerEvent::Finished {
                                reason,
                                completion_tokens,
                            },
                        ) {
                            let _ = send_event(&handle, &job.output, WorkerEvent::End);
                        }
                    }
                    Ok(Err(JobFailure::Execution(message))) => {
                        if send_event(&handle, &job.output, WorkerEvent::Failed(message)) {
                            let _ = send_event(&handle, &job.output, WorkerEvent::End);
                        }
                    }
                    Ok(Err(JobFailure::Deadline)) => {
                        if send_event(&handle, &job.output, WorkerEvent::TimedOut) {
                            let _ = send_event(&handle, &job.output, WorkerEvent::End);
                        }
                    }
                    Ok(Err(JobFailure::ClientGone)) => {}
                    Err(_) => {
                        if send_event(
                            &handle,
                            &job.output,
                            WorkerEvent::Failed("inference worker panicked".into()),
                        ) {
                            let _ = send_event(&handle, &job.output, WorkerEvent::End);
                        }
                        #[cfg(target_os = "macos")]
                        if mode == ServingBackend::Metal {
                            match catch_unwind(AssertUnwindSafe(|| {
                                MetalRuntime::new(&worker_model)
                            })) {
                                Ok(Ok(runtime)) => metal = Some(runtime),
                                _ => break,
                            }
                        }
                    }
                }
            }
        })
        .map_err(|error| ServingError::Worker(error.to_string()))?;
    ready_receiver
        .recv()
        .map_err(|_| ServingError::Worker("worker exited during startup".into()))?
        .map_err(ServingError::Worker)?;
    let state = Arc::new(AppState {
        model_id: config.model_id,
        api_key: config.api_key.into_bytes(),
        tokenizer,
        max_positions: model.config().max_positions,
        max_completion_tokens: config.max_completion_tokens,
        request_timeout: config.request_timeout,
        worker_status,
        requests: sender,
    });
    Ok((state, worker))
}

fn validate_config(config: &ServingConfig) -> Result<(), ServingError> {
    if config.model_id.is_empty() {
        return Err(ServingError::Configuration("model ID is empty"));
    }
    if config.api_key.len() < 32 || !config.api_key.bytes().all(|byte| byte.is_ascii_graphic()) {
        return Err(ServingError::Configuration(
            "API key must contain at least 32 visible ASCII characters",
        ));
    }
    if config.queue_capacity == 0 || config.queue_capacity > 1024 {
        return Err(ServingError::Configuration(
            "queue capacity must be between 1 and 1024",
        ));
    }
    if config.max_completion_tokens == 0 || config.max_completion_tokens > 4096 {
        return Err(ServingError::Configuration(
            "maximum completion length must be between 1 and 4096",
        ));
    }
    if config.request_timeout < MIN_REQUEST_TIMEOUT || config.request_timeout > MAX_REQUEST_TIMEOUT
    {
        return Err(ServingError::Configuration(
            "request timeout must be between 10 milliseconds and 10 minutes",
        ));
    }
    #[cfg(not(target_os = "macos"))]
    if config.backend == ServingBackend::Metal {
        return Err(ServingError::Configuration("Metal requires macOS"));
    }
    Ok(())
}

fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .merge(
            Router::new()
                .route("/v1/models", get(models))
                .route("/v1/chat/completions", post(chat_completions))
                .route_layer(middleware::from_fn_with_state(
                    Arc::clone(&state),
                    authenticate,
                )),
        )
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(state)
}

fn process_job(
    job: &Job,
    session: &mut GenerationSession<'_>,
    tokenizer: &ByteBpeTokenizer,
    handle: &Handle,
) -> Result<(&'static str, usize), JobFailure> {
    check_job(job)?;
    session
        .prefill(&job.prompt)
        .map_err(|error| JobFailure::Execution(error.to_string()))?;
    check_job(job)?;
    let stop = ["<|im_end|>", "<|endoftext|>"].map(|token| tokenizer.special_token_id(token));
    let mut decoder = ByteBpeDecoder::new();
    for step in 0..job.max_tokens {
        check_job(job)?;
        let token = session
            .selected_token()
            .map_err(|error| JobFailure::Execution(error.to_string()))?;
        let token_id = u32::try_from(token)
            .map_err(|_| JobFailure::Execution("token ID exceeds tokenizer range".into()))?;
        let stopped = stop.contains(&Some(token_id));
        if !stopped {
            let text = decoder
                .push(tokenizer, token_id)
                .map_err(|error| JobFailure::Execution(error.to_string()))?;
            if !text.is_empty() && !send_event(handle, &job.output, WorkerEvent::Delta(text)) {
                return Err(JobFailure::ClientGone);
            }
        }
        let completion_tokens = step + 1;
        if stopped || completion_tokens == job.max_tokens {
            let tail = decoder.finish();
            if !tail.is_empty() && !send_event(handle, &job.output, WorkerEvent::Delta(tail)) {
                return Err(JobFailure::ClientGone);
            }
            return Ok((if stopped { "stop" } else { "length" }, completion_tokens));
        }
        session
            .advance(token)
            .map_err(|error| JobFailure::Execution(error.to_string()))?;
    }
    Err(JobFailure::Execution("empty generation request".into()))
}

fn check_job(job: &Job) -> Result<(), JobFailure> {
    if job.output.is_closed() {
        Err(JobFailure::ClientGone)
    } else if Instant::now() >= job.deadline {
        Err(JobFailure::Deadline)
    } else {
        Ok(())
    }
}

fn send_event(handle: &Handle, sender: &mpsc::Sender<WorkerEvent>, event: WorkerEvent) -> bool {
    handle
        .block_on(sender.send_timeout(event, SLOW_CLIENT_TIMEOUT))
        .is_ok()
}

async fn authenticate(
    State(state): State<Arc<AppState>>,
    request: axum::extract::Request,
    next: Next,
) -> Response {
    let authorization: Vec<_> = request.headers().get_all(AUTHORIZATION).iter().collect();
    let valid = authorization.len() == 1
        && authorization[0]
            .to_str()
            .ok()
            .and_then(|header| header.strip_prefix("Bearer "))
            .is_some_and(|token| {
                token.len() == state.api_key.len()
                    && bool::from(token.as_bytes().ct_eq(&state.api_key))
            });
    let mut response = if valid {
        next.run(request).await
    } else {
        let mut response = error_response(
            StatusCode::UNAUTHORIZED,
            "invalid_api_key",
            "a valid Bearer token is required",
        );
        response
            .headers_mut()
            .insert(WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
        response
    };
    response
        .headers_mut()
        .insert(CACHE_CONTROL, HeaderValue::from_static("no-store"));
    response
}

async fn health(State(state): State<Arc<AppState>>) -> Response {
    if state.requests.is_closed() || state.worker_status.unavailable() {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"status": "unavailable"})),
        )
            .into_response()
    } else {
        Json(json!({"status": "ok"})).into_response()
    }
}

async fn models(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{"id": state.model_id, "object": "model", "owned_by": "local"}]
    }))
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    payload: Result<Json<Value>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Json(value) = match payload {
        Ok(value) => value,
        Err(error) => {
            return error_response(StatusCode::BAD_REQUEST, "invalid_json", &error.body_text())
        }
    };
    let request: ChatRequest = match serde_json::from_value(value) {
        Ok(request) => request,
        Err(error) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                &error.to_string(),
            )
        }
    };
    let (prompt, max_tokens) = match validate_request(&state, &request) {
        Ok(validated) => validated,
        Err(message) => {
            return error_response(StatusCode::BAD_REQUEST, "invalid_request", &message)
        }
    };
    let prompt_tokens = prompt.len();
    let id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs());
    let (output, receiver) = mpsc::channel(OUTPUT_CHANNEL_CAPACITY);
    if output.try_send(WorkerEvent::Started).is_err() {
        return error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "worker_unavailable",
            "could not start the response stream",
        );
    }
    let job = Job {
        prompt,
        max_tokens,
        deadline: Instant::now() + state.request_timeout,
        output,
    };
    let deadline = job.deadline;
    if state.worker_status.unavailable() {
        return error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "worker_unavailable",
            "the inference worker exceeded its deadline",
        );
    }
    match state.requests.try_send(job) {
        Ok(()) => {}
        Err(mpsc::error::TrySendError::Full(_)) => {
            return error_response(
                StatusCode::TOO_MANY_REQUESTS,
                "queue_full",
                "the inference queue is full; retry later",
            )
        }
        Err(mpsc::error::TrySendError::Closed(_)) => {
            return error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "worker_unavailable",
                "the inference worker is unavailable",
            )
        }
    }
    if request.stream {
        let model_id = state.model_id.clone();
        let stream = ReceiverStream::new(deadline_stream(receiver, deadline)).map(move |event| {
            Ok::<Event, Infallible>(
                Event::default().data(stream_data(event, &id, created, &model_id)),
            )
        });
        Sse::new(stream)
            .keep_alive(
                KeepAlive::new()
                    .interval(Duration::from_secs(10))
                    .text("keep-alive"),
            )
            .into_response()
    } else {
        match tokio::time::timeout_at(
            tokio::time::Instant::from_std(deadline),
            collect_response(
                receiver,
                &id,
                created,
                &state.model_id,
                prompt_tokens,
                deadline,
            ),
        )
        .await
        {
            Ok(response) => response,
            Err(_) => timeout_response(),
        }
    }
}

fn deadline_stream(
    mut receiver: mpsc::Receiver<WorkerEvent>,
    deadline: Instant,
) -> mpsc::Receiver<WorkerEvent> {
    let (sender, output) = mpsc::channel(OUTPUT_CHANNEL_CAPACITY);
    tokio::spawn(async move {
        loop {
            if Instant::now() >= deadline {
                if send_stream_event(&sender, WorkerEvent::TimedOut, deadline).await {
                    let _ = send_stream_event(&sender, WorkerEvent::End, deadline).await;
                }
                break;
            }
            let next = tokio::select! {
                biased;
                _ = sender.closed() => break,
                result = tokio::time::timeout_at(
                    tokio::time::Instant::from_std(deadline), receiver.recv()
                ) => result,
            };
            match next {
                Ok(Some(event)) => {
                    let terminal = matches!(
                        event,
                        WorkerEvent::Finished { .. }
                            | WorkerEvent::Failed(_)
                            | WorkerEvent::TimedOut
                            | WorkerEvent::End
                    );
                    let end = matches!(event, WorkerEvent::End);
                    if !send_stream_event(&sender, event, deadline).await {
                        break;
                    }
                    if terminal {
                        if !end {
                            let _ = send_stream_event(&sender, WorkerEvent::End, deadline).await;
                        }
                        break;
                    }
                }
                Ok(None) => {
                    if send_stream_event(
                        &sender,
                        WorkerEvent::Failed("inference worker closed the response".into()),
                        deadline,
                    )
                    .await
                    {
                        let _ = send_stream_event(&sender, WorkerEvent::End, deadline).await;
                    }
                    break;
                }
                Err(_) => {
                    if send_stream_event(&sender, WorkerEvent::TimedOut, deadline).await {
                        let _ = send_stream_event(&sender, WorkerEvent::End, deadline).await;
                    }
                    break;
                }
            }
        }
    });
    output
}

async fn send_stream_event(
    sender: &mpsc::Sender<WorkerEvent>,
    event: WorkerEvent,
    deadline: Instant,
) -> bool {
    tokio::time::timeout_at(tokio::time::Instant::from_std(deadline), sender.send(event))
        .await
        .is_ok_and(|result| result.is_ok())
}

fn validate_request(
    state: &AppState,
    request: &ChatRequest,
) -> Result<(Vec<usize>, usize), String> {
    if request.model != state.model_id {
        return Err("model ID is not available on this device".into());
    }
    if request.messages.is_empty() || request.messages.len() > 64 {
        return Err("messages must contain between 1 and 64 entries".into());
    }
    if request.temperature.is_some_and(|value| value != 0.0)
        || request.top_p.is_some_and(|value| value != 1.0)
        || request.n.is_some_and(|value| value != 1)
    {
        return Err("only greedy decoding with one choice is supported".into());
    }
    if request.max_tokens.is_some()
        && request.max_completion_tokens.is_some()
        && request.max_tokens != request.max_completion_tokens
    {
        return Err("max_tokens and max_completion_tokens disagree".into());
    }
    let max_tokens = request
        .max_completion_tokens
        .or(request.max_tokens)
        .unwrap_or(32);
    if max_tokens == 0 || max_tokens > state.max_completion_tokens {
        return Err(format!(
            "max_completion_tokens must be between 1 and {}",
            state.max_completion_tokens
        ));
    }
    let start = state
        .tokenizer
        .special_token_id("<|im_start|>")
        .ok_or("tokenizer lacks the chat start token")?;
    let end = state
        .tokenizer
        .special_token_id("<|im_end|>")
        .ok_or("tokenizer lacks the chat end token")?;
    let mut prompt = Vec::new();
    for message in &request.messages {
        if !matches!(
            message.role.as_str(),
            "system" | "developer" | "user" | "assistant"
        ) {
            return Err(format!("unsupported message role: {}", message.role));
        }
        prompt.push(start as usize);
        prompt.extend(
            state
                .tokenizer
                .encode_plain_text(&format!("{}\n", message.role))
                .map_err(|error| error.to_string())?
                .into_iter()
                .map(|token| token as usize),
        );
        prompt.extend(
            state
                .tokenizer
                .encode_plain_text(&message.content)
                .map_err(|error| error.to_string())?
                .into_iter()
                .map(|token| token as usize),
        );
        prompt.push(end as usize);
        prompt.extend(
            state
                .tokenizer
                .encode_plain_text("\n")
                .map_err(|error| error.to_string())?
                .into_iter()
                .map(|token| token as usize),
        );
    }
    prompt.push(start as usize);
    prompt.extend(
        state
            .tokenizer
            .encode_plain_text("assistant\n")
            .map_err(|error| error.to_string())?
            .into_iter()
            .map(|token| token as usize),
    );
    if prompt.len() >= state.max_positions || max_tokens > state.max_positions - prompt.len() {
        return Err("prompt and requested completion exceed the model context".into());
    }
    Ok((prompt, max_tokens))
}

async fn collect_response(
    mut receiver: mpsc::Receiver<WorkerEvent>,
    id: &str,
    created: u64,
    model_id: &str,
    prompt_tokens: usize,
    deadline: Instant,
) -> Response {
    let mut content = String::new();
    while let Some(event) = receiver.recv().await {
        if Instant::now() >= deadline {
            return timeout_response();
        }
        match event {
            WorkerEvent::Started | WorkerEvent::End => {}
            WorkerEvent::Delta(text) => content.push_str(&text),
            WorkerEvent::Finished {
                reason,
                completion_tokens,
            } => {
                return Json(json!({
                    "id": id,
                    "object": "chat.completion",
                    "created": created,
                    "model": model_id,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": reason
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    }
                }))
                .into_response();
            }
            WorkerEvent::Failed(message) => {
                return error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "inference_failed",
                    &message,
                )
            }
            WorkerEvent::TimedOut => return timeout_response(),
        }
    }
    error_response(
        StatusCode::SERVICE_UNAVAILABLE,
        "worker_unavailable",
        "the inference worker closed the response",
    )
}

fn stream_data(event: WorkerEvent, id: &str, created: u64, model_id: &str) -> String {
    let base = |delta: Value, finish_reason: Option<&str>| {
        json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
        })
        .to_string()
    };
    match event {
        WorkerEvent::Started => base(json!({"role": "assistant"}), None),
        WorkerEvent::Delta(content) => base(json!({"content": content}), None),
        WorkerEvent::Finished { reason, .. } => base(json!({}), Some(reason)),
        WorkerEvent::Failed(message) => json!({
            "error": {"message": message, "type": "server_error", "code": "inference_failed"}
        })
        .to_string(),
        WorkerEvent::TimedOut => json!({
            "error": {
                "message": "request deadline exceeded",
                "type": "server_error",
                "code": "request_timeout"
            }
        })
        .to_string(),
        WorkerEvent::End => "[DONE]".into(),
    }
}

fn timeout_response() -> Response {
    error_response(
        StatusCode::GATEWAY_TIMEOUT,
        "request_timeout",
        "request deadline exceeded",
    )
}

fn error_response(status: StatusCode, code: &str, message: &str) -> Response {
    let error_type = if status.is_server_error() {
        "server_error"
    } else if status == StatusCode::TOO_MANY_REQUESTS {
        "rate_limit_error"
    } else if status == StatusCode::UNAUTHORIZED {
        "authentication_error"
    } else {
        "invalid_request_error"
    };
    (
        status,
        Json(json!({
            "error": {"message": message, "type": error_type, "code": code}
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use std::env;
    use std::path::PathBuf;

    use axum::body::{to_bytes, Body};
    use axum::http::{Method, Request};
    use tower::ServiceExt;

    use super::*;
    use crate::{load_gguf, load_gguf_tokenizer};

    const KEY: &str = "test-only-key-with-at-least-32-characters";

    fn test_tokenizer() -> ByteBpeTokenizer {
        let mut pieces = vec![
            "<|im_start|>".to_owned(),
            "<|im_end|>".to_owned(),
            "<|endoftext|>".to_owned(),
        ];
        let mut types = vec![3, 3, 3];
        for piece in [
            "a", "s", "i", "t", "n", "u", "e", "r", "y", "h", "S", "Ċ", "Ġ",
        ] {
            pieces.push(piece.to_owned());
            types.push(1);
        }
        ByteBpeTokenizer::from_gguf_parts(pieces, Vec::new(), types).unwrap()
    }

    fn stalled_app(timeout: Duration) -> (Router, mpsc::Receiver<Job>, Arc<WorkerStatus>) {
        let tokenizer = test_tokenizer();
        assert!(!tokenizer
            .encode_plain_text("assistant\nSay hi")
            .unwrap()
            .is_empty());
        let (sender, receiver) = mpsc::channel(1);
        let worker_status = Arc::new(WorkerStatus::default());
        let app = router(Arc::new(AppState {
            model_id: "local-smollm2".into(),
            api_key: KEY.as_bytes().to_vec(),
            tokenizer: Arc::new(tokenizer),
            max_positions: 2048,
            max_completion_tokens: 4,
            request_timeout: timeout,
            worker_status: Arc::clone(&worker_status),
            requests: sender,
        }));
        (app, receiver, worker_status)
    }

    fn request(path: &str, body: Option<Value>, authorized: bool) -> Request<Body> {
        let mut builder = Request::builder().uri(path);
        if let Some(value) = body.as_ref() {
            builder = builder
                .method(Method::POST)
                .header("content-type", "application/json");
            if authorized {
                builder = builder.header(AUTHORIZATION, format!("Bearer {KEY}"));
            }
            builder.body(Body::from(value.to_string())).unwrap()
        } else {
            if authorized {
                builder = builder.header(AUTHORIZATION, format!("Bearer {KEY}"));
            }
            builder.body(Body::empty()).unwrap()
        }
    }

    fn chat(stream: bool) -> Value {
        json!({
            "model": "local-smollm2",
            "messages": [{"role": "user", "content": "Say hi"}],
            "max_completion_tokens": 2,
            "stream": stream
        })
    }

    async fn body_json(response: Response) -> Value {
        let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn queued_requests_expire_for_json_and_streaming() {
        let (app, mut jobs, _) = stalled_app(Duration::from_millis(25));
        let response = app
            .clone()
            .oneshot(request("/v1/chat/completions", Some(chat(false)), true))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
        assert_eq!(
            body_json(response).await["error"]["code"],
            "request_timeout"
        );
        let expired = jobs.recv().await.unwrap();
        assert!(expired.output.is_closed());
        assert!(matches!(check_job(&expired), Err(JobFailure::ClientGone)));

        let responder = tokio::spawn(async move {
            let job = jobs.recv().await.unwrap();
            job.output
                .send(WorkerEvent::Delta("recovered".into()))
                .await
                .unwrap();
            job.output
                .send(WorkerEvent::Finished {
                    reason: "length",
                    completion_tokens: 1,
                })
                .await
                .unwrap();
        });
        let response = app
            .oneshot(request("/v1/chat/completions", Some(chat(false)), true))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            body_json(response).await["choices"][0]["message"]["content"],
            "recovered"
        );
        responder.await.unwrap();

        let (app, mut jobs, _) = stalled_app(Duration::from_millis(25));
        let response = app
            .oneshot(request("/v1/chat/completions", Some(chat(true)), true))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let text = String::from_utf8(
            to_bytes(response.into_body(), 1024 * 1024)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        assert!(text.contains("\"code\":\"request_timeout\""));
        assert!(text.contains("data: [DONE]"));
        assert!(jobs.recv().await.unwrap().output.is_closed());
    }

    #[tokio::test]
    async fn overdue_worker_fails_readiness_and_new_requests() {
        let (app, _jobs, status) = stalled_app(Duration::from_secs(1));
        status.start(Instant::now() - Duration::from_secs(1));
        let response = app
            .clone()
            .oneshot(request("/health", None, false))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let response = app
            .clone()
            .oneshot(request("/v1/chat/completions", Some(chat(false)), true))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            body_json(response).await["error"]["code"],
            "worker_unavailable"
        );
        status.clear();
        let response = app.oneshot(request("/health", None, false)).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    #[ignore = "requires SmolLM2-135M-Q4_K_M.gguf in SMOLLM2_DIR"]
    async fn authenticated_chat_streams_and_reports_overload() {
        let directory = PathBuf::from(env::var("SMOLLM2_DIR").expect("set SMOLLM2_DIR"));
        let path = directory.join("SmolLM2-135M-Q4_K_M.gguf");
        let tokenizer = load_gguf_tokenizer(&path).unwrap();
        let model = load_gguf(&path).unwrap();
        let (state, worker) = start_state(
            model,
            tokenizer,
            ServingConfig {
                model_id: "local-smollm2".into(),
                api_key: KEY.into(),
                queue_capacity: 1,
                max_completion_tokens: 4,
                request_timeout: Duration::from_secs(30),
                backend: ServingBackend::Cpu,
            },
        )
        .unwrap();

        let (output, mut failed) = mpsc::channel(OUTPUT_CHANNEL_CAPACITY);
        state
            .requests
            .try_send(Job {
                prompt: vec![usize::MAX],
                max_tokens: 1,
                deadline: Instant::now() + Duration::from_secs(30),
                output,
            })
            .unwrap();
        assert!(matches!(failed.recv().await, Some(WorkerEvent::Failed(_))));
        assert!(matches!(failed.recv().await, Some(WorkerEvent::End)));
        let app = router(state);

        let response = app
            .clone()
            .oneshot(request("/health", None, false))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let response = app
            .clone()
            .oneshot(request("/v1/models", None, false))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(response.headers()[CACHE_CONTROL], "no-store");
        let response = app
            .clone()
            .oneshot(request("/v1/models", None, true))
            .await
            .unwrap();
        assert_eq!(body_json(response).await["data"][0]["id"], "local-smollm2");

        let mut unsupported = chat(false);
        unsupported["temperature"] = json!(0.7);
        let response = app
            .clone()
            .oneshot(request("/v1/chat/completions", Some(unsupported), true))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let response = app
            .clone()
            .oneshot(request("/v1/chat/completions", Some(chat(false)), true))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let completion = body_json(response).await;
        assert_eq!(completion["object"], "chat.completion");
        assert_eq!(completion["choices"][0]["message"]["role"], "assistant");
        assert!(completion["usage"]["completion_tokens"].as_u64().unwrap() <= 2);

        let response = app
            .clone()
            .oneshot(request("/v1/chat/completions", Some(chat(true)), true))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert!(response.headers()["content-type"]
            .to_str()
            .unwrap()
            .starts_with("text/event-stream"));
        let text = String::from_utf8(
            to_bytes(response.into_body(), 1024 * 1024)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        assert!(text.contains("chat.completion.chunk"));
        assert!(text.contains("data: [DONE]"));

        let tokenizer = load_gguf_tokenizer(&path).unwrap();
        let (sender, _receiver) = mpsc::channel(1);
        let (output, _output_receiver) = mpsc::channel(1);
        sender
            .try_send(Job {
                prompt: vec![1],
                max_tokens: 1,
                deadline: Instant::now() + Duration::from_secs(30),
                output,
            })
            .unwrap();
        let saturated = router(Arc::new(AppState {
            model_id: "local-smollm2".into(),
            api_key: KEY.as_bytes().to_vec(),
            tokenizer: Arc::new(tokenizer),
            max_positions: 2048,
            max_completion_tokens: 4,
            request_timeout: Duration::from_secs(30),
            worker_status: Arc::new(WorkerStatus::default()),
            requests: sender,
        }));
        let response = saturated
            .oneshot(request("/v1/chat/completions", Some(chat(false)), true))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(body_json(response).await["error"]["code"], "queue_full");

        drop(app);
        tokio::task::spawn_blocking(move || worker.join().unwrap())
            .await
            .unwrap();
    }
}
