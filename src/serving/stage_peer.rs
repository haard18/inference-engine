//! A suffix-stage service reachable only through explicitly approved peer TLS.

use std::fs;
use std::io;
use std::net::{SocketAddr, TcpListener};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Weak};
use std::time::Duration;

use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, StatusCode};
use axum::routing::{get, post};
use axum::{Json, Router};
use rustls::ServerConfig;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::{Mutex, Semaphore};
use uuid::Uuid;

use crate::pool::{tls::server_config, DeviceIdentity, PeerStore};

use super::ServingError;

const STARTUP_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_FRAME_BYTES: usize = 4 * 1024 * 1024;
const MAX_SCORE_BYTES: usize = 4 * 1024 * 1024;
const MAX_DEADLINE_MS: u64 = 120_000;
const MAX_QUEUE: usize = 16;
const REQUEST_ID_HEADER: &str = "x-inference-request-id";
const DEADLINE_HEADER: &str = "x-inference-deadline-ms";

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StageCapacitySnapshot {
    pub ready: bool,
    pub model_digest: String,
    pub layer_start: usize,
    pub layer_end: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub max_positions: usize,
    pub stored_weight_bytes: usize,
    pub queue_capacity: usize,
    pub queue_available: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub(super) struct StageReady {
    kind: String,
    pub(super) model_digest: String,
    pub(super) layer_start: usize,
    pub(super) layer_end: usize,
    pub(super) hidden_size: usize,
    pub(super) vocab_size: usize,
    pub(super) max_positions: usize,
    stored_weight_bytes: usize,
}

pub(super) struct StageChild {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
    pub(super) ready: StageReady,
}

struct StageState {
    worker: Mutex<Option<StageChild>>,
    expected: StageReady,
    executable: PathBuf,
    model_path: PathBuf,
    queue: Arc<Semaphore>,
    queue_capacity: usize,
}

pub struct StagePeerServer {
    routes: Router,
    tls: ServerConfig,
}

/// Start a suffix worker behind the same approved-certificate rules as whole-request peers.
pub async fn start_stage_peer(
    model_path: impl AsRef<Path>,
    executable: impl AsRef<Path>,
    layer_start: usize,
    layer_end: usize,
    queue_capacity: usize,
    identity: &DeviceIdentity,
    peers: &PeerStore,
) -> Result<StagePeerServer, ServingError> {
    if queue_capacity == 0 || queue_capacity > MAX_QUEUE {
        return Err(ServingError::Configuration(
            "stage queue capacity must be 1 through 16",
        ));
    }
    let tls =
        server_config(identity, peers).map_err(|error| ServingError::Peer(error.to_string()))?;
    let model_path = fs::canonicalize(model_path)
        .map_err(|error| ServingError::Worker(format!("stage model path: {error}")))?;
    let executable = fs::canonicalize(executable)
        .map_err(|error| ServingError::Worker(format!("stage executable: {error}")))?;
    let child = tokio::time::timeout(
        STARTUP_TIMEOUT,
        StageChild::spawn(&executable, &model_path, layer_start, layer_end),
    )
    .await
    .map_err(|_| ServingError::Worker("stage worker startup timed out".into()))?
    .map_err(ServingError::Worker)?;
    if child.ready.layer_start == 0 {
        return Err(ServingError::Configuration(
            "peer stage service requires a suffix range",
        ));
    }
    let state = Arc::new(StageState {
        expected: child.ready.clone(),
        worker: Mutex::new(Some(child)),
        executable,
        model_path,
        queue: Arc::new(Semaphore::new(queue_capacity)),
        queue_capacity,
    });
    let routes = Router::new()
        .route("/internal/stage/capacity", get(capacity))
        .route("/internal/stage/activation", post(activation))
        .route("/internal/stage/close", post(close))
        .route("/internal/stage/probe", post(probe))
        .route("/internal/stage/rewind", post(rewind))
        .layer(DefaultBodyLimit::max(MAX_FRAME_BYTES))
        .with_state(Arc::clone(&state));
    tokio::spawn(supervise_stage(Arc::downgrade(&state)));
    Ok(StagePeerServer { routes, tls })
}

async fn supervise_stage(weak: Weak<StageState>) {
    loop {
        tokio::time::sleep(Duration::from_secs(1)).await;
        let Some(state) = weak.upgrade() else { break };
        let mut worker = state.worker.lock().await;
        if worker
            .as_mut()
            .is_some_and(|child| !matches!(child.child.try_wait(), Ok(None)))
        {
            worker.take();
        }
        if worker.is_none() {
            if let Ok(Ok(child)) = tokio::time::timeout(
                STARTUP_TIMEOUT,
                StageChild::spawn(
                    &state.executable,
                    &state.model_path,
                    state.expected.layer_start,
                    state.expected.layer_end,
                ),
            )
            .await
            {
                if child.ready == state.expected {
                    *worker = Some(child);
                }
            }
        }
    }
}

impl StagePeerServer {
    pub async fn serve(
        self,
        listener: TcpListener,
        handle: axum_server::Handle<SocketAddr>,
    ) -> io::Result<()> {
        listener.set_nonblocking(true)?;
        let config = axum_server::tls_rustls::RustlsConfig::from_config(Arc::new(self.tls));
        axum_server::from_tcp_rustls(listener, config)?
            .handle(handle)
            .serve(self.routes.into_make_service())
            .await
    }
}

async fn capacity(State(state): State<Arc<StageState>>) -> Json<StageCapacitySnapshot> {
    let mut worker = state.worker.lock().await;
    if worker
        .as_mut()
        .is_some_and(|child| !matches!(child.child.try_wait(), Ok(None)))
    {
        worker.take();
    }
    Json(StageCapacitySnapshot {
        ready: worker.is_some(),
        model_digest: state.expected.model_digest.clone(),
        layer_start: state.expected.layer_start,
        layer_end: state.expected.layer_end,
        hidden_size: state.expected.hidden_size,
        vocab_size: state.expected.vocab_size,
        max_positions: state.expected.max_positions,
        stored_weight_bytes: state.expected.stored_weight_bytes,
        queue_capacity: state.queue_capacity,
        queue_available: state.queue.available_permits(),
    })
}

async fn activation(
    State(state): State<Arc<StageState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<([(&'static str, &'static str); 1], Vec<u8>), (StatusCode, String)> {
    if !(64..=MAX_FRAME_BYTES).contains(&body.len()) {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            "invalid activation size".into(),
        ));
    }
    let request_id = request_id(&headers)?;
    let deadline = deadline(&headers)?;
    let _permit = state
        .queue
        .clone()
        .try_acquire_owned()
        .map_err(|_| (StatusCode::TOO_MANY_REQUESTS, "stage queue is full".into()))?;
    let mut worker = tokio::time::timeout_at(deadline, state.worker.lock())
        .await
        .map_err(|_| {
            (
                StatusCode::GATEWAY_TIMEOUT,
                "stage queue wait timed out".into(),
            )
        })?;
    ensure_worker(&state, &mut worker, deadline).await?;
    // Cancellation drops this child and discards any incomplete pipe response.
    let mut child = worker.take().expect("worker started");
    let result = tokio::time::timeout_at(deadline, child.step(request_id, &body)).await;
    match result {
        Ok(Ok(scores)) => {
            *worker = Some(child);
            Ok(([("content-type", "application/octet-stream")], scores))
        }
        Ok(Err(StageStepError::Rejected(message))) => {
            *worker = Some(child);
            let status = if message.contains("capacity") || message.contains("memory limit") {
                StatusCode::TOO_MANY_REQUESTS
            } else {
                StatusCode::UNPROCESSABLE_ENTITY
            };
            Err((status, message))
        }
        Ok(Err(StageStepError::Broken(message))) => Err((StatusCode::SERVICE_UNAVAILABLE, message)),
        Err(_) => Err((StatusCode::GATEWAY_TIMEOUT, "stage step timed out".into())),
    }
}

async fn close(
    State(state): State<Arc<StageState>>,
    headers: HeaderMap,
) -> Result<StatusCode, (StatusCode, String)> {
    let request_id = request_id(&headers)?;
    let deadline = deadline(&headers)?;
    let _permit = state
        .queue
        .clone()
        .try_acquire_owned()
        .map_err(|_| (StatusCode::TOO_MANY_REQUESTS, "stage queue is full".into()))?;
    let mut worker = tokio::time::timeout_at(deadline, state.worker.lock())
        .await
        .map_err(|_| {
            (
                StatusCode::GATEWAY_TIMEOUT,
                "stage queue wait timed out".into(),
            )
        })?;
    if worker.is_none() {
        return Ok(StatusCode::NO_CONTENT);
    }
    let mut child = worker.take().expect("worker present");
    match tokio::time::timeout_at(deadline, child.close(request_id)).await {
        Ok(Ok(())) => {
            *worker = Some(child);
            Ok(StatusCode::NO_CONTENT)
        }
        _ => Err((StatusCode::SERVICE_UNAVAILABLE, "stage close failed".into())),
    }
}

async fn probe(
    State(state): State<Arc<StageState>>,
    headers: HeaderMap,
) -> Result<Json<Value>, (StatusCode, String)> {
    let request_id = request_id(&headers)?;
    let deadline = deadline(&headers)?;
    let _permit = state
        .queue
        .clone()
        .try_acquire_owned()
        .map_err(|_| (StatusCode::TOO_MANY_REQUESTS, "stage queue is full".into()))?;
    let mut worker = tokio::time::timeout_at(deadline, state.worker.lock())
        .await
        .map_err(|_| {
            (
                StatusCode::GATEWAY_TIMEOUT,
                "stage queue wait timed out".into(),
            )
        })?;
    if worker.as_mut().is_some_and(|child| !child.running()) {
        worker.take();
    }
    let Some(mut child) = worker.take() else {
        return Ok(Json(json!({"position": null})));
    };
    match tokio::time::timeout_at(deadline, child.probe(request_id)).await {
        Ok(Ok(position)) => {
            *worker = Some(child);
            Ok(Json(json!({"position": position})))
        }
        Ok(Err(StageStepError::Rejected(message))) => {
            *worker = Some(child);
            Err((StatusCode::UNPROCESSABLE_ENTITY, message))
        }
        Ok(Err(StageStepError::Broken(message))) => Err((StatusCode::SERVICE_UNAVAILABLE, message)),
        Err(_) => Err((StatusCode::GATEWAY_TIMEOUT, "stage probe timed out".into())),
    }
}

async fn rewind(
    State(state): State<Arc<StageState>>,
    headers: HeaderMap,
) -> Result<StatusCode, (StatusCode, String)> {
    let request_id = request_id(&headers)?;
    let deadline = deadline(&headers)?;
    let position = headers
        .get("x-inference-position")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|position| *position > 0 && *position <= state.expected.max_positions)
        .ok_or((StatusCode::BAD_REQUEST, "invalid stage position".into()))?;
    let _permit = state
        .queue
        .clone()
        .try_acquire_owned()
        .map_err(|_| (StatusCode::TOO_MANY_REQUESTS, "stage queue is full".into()))?;
    let mut worker = tokio::time::timeout_at(deadline, state.worker.lock())
        .await
        .map_err(|_| {
            (
                StatusCode::GATEWAY_TIMEOUT,
                "stage queue wait timed out".into(),
            )
        })?;
    if worker.as_mut().is_some_and(|child| !child.running()) {
        worker.take();
    }
    let Some(mut child) = worker.take() else {
        return Err((StatusCode::CONFLICT, "stage session is missing".into()));
    };
    match tokio::time::timeout_at(deadline, child.rewind(request_id, position)).await {
        Ok(Ok(())) => {
            *worker = Some(child);
            Ok(StatusCode::NO_CONTENT)
        }
        Ok(Err(StageStepError::Rejected(message))) => {
            *worker = Some(child);
            Err((StatusCode::CONFLICT, message))
        }
        Ok(Err(StageStepError::Broken(message))) => Err((StatusCode::SERVICE_UNAVAILABLE, message)),
        Err(_) => Err((StatusCode::GATEWAY_TIMEOUT, "stage rewind timed out".into())),
    }
}

async fn ensure_worker(
    state: &StageState,
    worker: &mut Option<StageChild>,
    deadline: tokio::time::Instant,
) -> Result<(), (StatusCode, String)> {
    if worker
        .as_mut()
        .is_some_and(|child| !matches!(child.child.try_wait(), Ok(None)))
    {
        worker.take();
    }
    if worker.is_none() {
        let remaining = tokio::time::timeout_at(
            deadline,
            StageChild::spawn(
                &state.executable,
                &state.model_path,
                state.expected.layer_start,
                state.expected.layer_end,
            ),
        )
        .await
        .map_err(|_| {
            (
                StatusCode::GATEWAY_TIMEOUT,
                "stage restart timed out".into(),
            )
        })?
        .map_err(|error| (StatusCode::SERVICE_UNAVAILABLE, error))?;
        if remaining.ready != state.expected {
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                "replacement stage model identity changed".into(),
            ));
        }
        *worker = Some(remaining);
    }
    Ok(())
}

fn request_id(headers: &HeaderMap) -> Result<Uuid, (StatusCode, String)> {
    headers
        .get(REQUEST_ID_HEADER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| Uuid::parse_str(value).ok())
        .ok_or((
            StatusCode::BAD_REQUEST,
            "missing or invalid request ID".into(),
        ))
}

fn deadline(headers: &HeaderMap) -> Result<tokio::time::Instant, (StatusCode, String)> {
    let millis = headers
        .get(DEADLINE_HEADER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| (1..=MAX_DEADLINE_MS).contains(value))
        .ok_or((StatusCode::BAD_REQUEST, "invalid stage deadline".into()))?;
    Ok(tokio::time::Instant::now() + Duration::from_millis(millis))
}

pub(super) enum StageStepError {
    Rejected(String),
    Broken(String),
}

impl StageChild {
    pub(super) async fn spawn(
        executable: &Path,
        model_path: &Path,
        start: usize,
        end: usize,
    ) -> Result<Self, String> {
        let mut child = Command::new(executable)
            .arg("--internal-stage-worker")
            .arg(model_path)
            .arg(start.to_string())
            .arg(end.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .env_remove("INFERENCE_API_KEY")
            .kill_on_drop(true)
            .spawn()
            .map_err(|error| error.to_string())?;
        let input = child.stdin.take().ok_or("stage input pipe unavailable")?;
        let stdout = child.stdout.take().ok_or("stage output pipe unavailable")?;
        let mut output = BufReader::new(stdout);
        let mut line = String::new();
        let count = output
            .read_line(&mut line)
            .await
            .map_err(|error| error.to_string())?;
        if count == 0 || count > 4096 {
            return Err("stage worker sent an invalid startup response".into());
        }
        let ready: StageReady = serde_json::from_str(&line).map_err(|error| error.to_string())?;
        if ready.kind != "ready"
            || ready.layer_start != start
            || ready.layer_end != end
            || ready.hidden_size == 0
            || ready.vocab_size == 0
            || ready.max_positions == 0
            || ready.stored_weight_bytes == 0
            || ready.model_digest.len() != 64
            || hex::decode(&ready.model_digest).is_err()
        {
            return Err("stage worker announced invalid model metadata".into());
        }
        Ok(Self {
            child,
            input,
            output,
            ready,
        })
    }

    pub(super) fn running(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    pub(super) async fn token(
        &mut self,
        request_id: Uuid,
        token_id: usize,
    ) -> Result<Vec<u8>, StageStepError> {
        let command = json!({
            "kind": "token",
            "request_id": request_id.to_string(),
            "token_id": token_id,
        });
        let mut line = serde_json::to_vec(&command)
            .map_err(|error| StageStepError::Broken(error.to_string()))?;
        line.push(b'\n');
        self.input
            .write_all(&line)
            .await
            .map_err(|error| StageStepError::Broken(error.to_string()))?;
        self.input
            .flush()
            .await
            .map_err(|error| StageStepError::Broken(error.to_string()))?;
        let reply = self.read_reply().await?;
        match reply["kind"].as_str() {
            Some("failed") => Err(StageStepError::Rejected(
                reply["message"]
                    .as_str()
                    .unwrap_or("stage rejected token")
                    .into(),
            )),
            Some("activation") => {
                let size = reply["payload_bytes"]
                    .as_u64()
                    .and_then(|size| usize::try_from(size).ok())
                    .ok_or_else(|| {
                        StageStepError::Broken("stage omitted activation size".into())
                    })?;
                let expected = self
                    .ready
                    .hidden_size
                    .checked_mul(4)
                    .and_then(|size| size.checked_add(64))
                    .ok_or_else(|| StageStepError::Broken("activation size overflows".into()))?;
                if size != expected || size > MAX_FRAME_BYTES {
                    return Err(StageStepError::Broken(
                        "stage activation size is invalid".into(),
                    ));
                }
                let mut frame = vec![0; size];
                self.output
                    .read_exact(&mut frame)
                    .await
                    .map_err(|error| StageStepError::Broken(error.to_string()))?;
                Ok(frame)
            }
            _ => Err(StageStepError::Broken(
                "stage sent an invalid response".into(),
            )),
        }
    }

    async fn step(&mut self, request_id: Uuid, body: &[u8]) -> Result<Vec<u8>, StageStepError> {
        let command = json!({
            "kind":"activation",
            "request_id":request_id.to_string(),
            "payload_bytes":body.len()
        });
        let mut line = serde_json::to_vec(&command)
            .map_err(|error| StageStepError::Broken(error.to_string()))?;
        line.push(b'\n');
        self.input
            .write_all(&line)
            .await
            .map_err(|error| StageStepError::Broken(error.to_string()))?;
        self.input
            .write_all(body)
            .await
            .map_err(|error| StageStepError::Broken(error.to_string()))?;
        self.input
            .flush()
            .await
            .map_err(|error| StageStepError::Broken(error.to_string()))?;
        let reply = self.read_reply().await?;
        match reply["kind"].as_str() {
            Some("failed") => Err(StageStepError::Rejected(
                reply["message"]
                    .as_str()
                    .unwrap_or("stage rejected input")
                    .into(),
            )),
            Some("scores") => {
                let count = reply["count"]
                    .as_u64()
                    .ok_or_else(|| StageStepError::Broken("stage omitted score count".into()))?
                    as usize;
                let bytes = count
                    .checked_mul(4)
                    .ok_or_else(|| StageStepError::Broken("stage score size overflows".into()))?;
                if count != self.ready.vocab_size || bytes > MAX_SCORE_BYTES {
                    return Err(StageStepError::Broken(
                        "stage score count is invalid".into(),
                    ));
                }
                let mut scores = vec![0; bytes];
                self.output
                    .read_exact(&mut scores)
                    .await
                    .map_err(|error| StageStepError::Broken(error.to_string()))?;
                Ok(scores)
            }
            _ => Err(StageStepError::Broken(
                "stage sent an invalid response".into(),
            )),
        }
    }

    pub(super) async fn close(&mut self, request_id: Uuid) -> Result<(), StageStepError> {
        let command = json!({"kind":"close","request_id":request_id.to_string()});
        let mut line = serde_json::to_vec(&command)
            .map_err(|error| StageStepError::Broken(error.to_string()))?;
        line.push(b'\n');
        self.input
            .write_all(&line)
            .await
            .map_err(|error| StageStepError::Broken(error.to_string()))?;
        self.input
            .flush()
            .await
            .map_err(|error| StageStepError::Broken(error.to_string()))?;
        let reply = self.read_reply().await?;
        if reply["kind"] == "closed" {
            Ok(())
        } else {
            Err(StageStepError::Broken("stage did not close request".into()))
        }
    }

    pub(super) async fn probe(
        &mut self,
        request_id: Uuid,
    ) -> Result<Option<usize>, StageStepError> {
        let command = json!({"kind":"probe","request_id":request_id.to_string()});
        let mut line = serde_json::to_vec(&command)
            .map_err(|error| StageStepError::Broken(error.to_string()))?;
        line.push(b'\n');
        self.input
            .write_all(&line)
            .await
            .map_err(|error| StageStepError::Broken(error.to_string()))?;
        self.input
            .flush()
            .await
            .map_err(|error| StageStepError::Broken(error.to_string()))?;
        let reply = self.read_reply().await?;
        if reply["kind"] != "position" {
            return Err(StageStepError::Broken(
                "stage did not report position".into(),
            ));
        }
        match reply.get("position") {
            Some(Value::Null) => Ok(None),
            Some(Value::Number(number)) => number
                .as_u64()
                .and_then(|position| usize::try_from(position).ok())
                .filter(|position| *position <= self.ready.max_positions)
                .map(Some)
                .ok_or_else(|| StageStepError::Broken("stage position is invalid".into())),
            _ => Err(StageStepError::Broken("stage position is invalid".into())),
        }
    }

    pub(super) async fn rewind(
        &mut self,
        request_id: Uuid,
        position: usize,
    ) -> Result<(), StageStepError> {
        let command =
            json!({"kind":"rewind","request_id":request_id.to_string(),"position":position});
        let mut line = serde_json::to_vec(&command)
            .map_err(|error| StageStepError::Broken(error.to_string()))?;
        line.push(b'\n');
        self.input
            .write_all(&line)
            .await
            .map_err(|error| StageStepError::Broken(error.to_string()))?;
        self.input
            .flush()
            .await
            .map_err(|error| StageStepError::Broken(error.to_string()))?;
        let reply = self.read_reply().await?;
        match reply["kind"].as_str() {
            Some("rewound") if reply["position"].as_u64() == Some(position as u64) => Ok(()),
            Some("failed") => Err(StageStepError::Rejected(
                reply["message"]
                    .as_str()
                    .unwrap_or("stage rewind failed")
                    .into(),
            )),
            _ => Err(StageStepError::Broken(
                "stage did not rewind session".into(),
            )),
        }
    }

    async fn read_reply(&mut self) -> Result<Value, StageStepError> {
        let mut line = String::new();
        let count = self
            .output
            .read_line(&mut line)
            .await
            .map_err(|error| StageStepError::Broken(error.to_string()))?;
        if count == 0 || count > 4096 {
            return Err(StageStepError::Broken("stage reply line is invalid".into()));
        }
        serde_json::from_str(&line).map_err(|error| StageStepError::Broken(error.to_string()))
    }
}

#[cfg(all(test, unix))]
mod tests {
    use std::os::unix::fs::PermissionsExt;

    use axum::body::{to_bytes, Body};
    use axum::http::Request;
    use tower::ServiceExt;

    use super::*;

    fn request() -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri("/internal/stage/activation")
            .header(REQUEST_ID_HEADER, Uuid::new_v4().to_string())
            .header(DEADLINE_HEADER, "3000")
            .body(Body::from(vec![0_u8; 64]))
            .unwrap()
    }

    #[tokio::test]
    async fn cancelling_a_stage_step_discards_the_child_before_the_next_request() {
        let directory = tempfile::tempdir().unwrap();
        let model = directory.path().join("model.gguf");
        fs::write(&model, []).unwrap();
        let script = directory.path().join("fake-stage.py");
        fs::write(
            &script,
            r#"#!/usr/bin/env python3
import json, os, struct, sys, time
marker = sys.argv[2] + '.started'
first = not os.path.exists(marker)
print(json.dumps({'kind':'ready','model_digest':'07'*32,'layer_start':1,'layer_end':2,'hidden_size':2,'vocab_size':2,'max_positions':4,'stored_weight_bytes':1}), flush=True)
command = json.loads(sys.stdin.buffer.readline())
sys.stdin.buffer.read(command['payload_bytes'])
if first:
    open(marker, 'w').close()
    time.sleep(60)
else:
    print(json.dumps({'kind':'scores','count':2}), flush=True)
    sys.stdout.buffer.write(struct.pack('<ff', 0.5, 1.0))
    sys.stdout.buffer.flush()
"#,
        )
        .unwrap();
        let mut permissions = fs::metadata(&script).unwrap().permissions();
        permissions.set_mode(0o700);
        fs::set_permissions(&script, permissions).unwrap();
        let own_dir = tempfile::tempdir().unwrap();
        let peer_dir = tempfile::tempdir().unwrap();
        let own = DeviceIdentity::load_or_create(own_dir.path()).unwrap();
        let peer = DeviceIdentity::load_or_create(peer_dir.path()).unwrap();
        let mut peers = PeerStore::load(own_dir.path()).unwrap();
        peers
            .trust(
                &own.device_id,
                &peer.offer(),
                "127.0.0.1:18888".parse().unwrap(),
                &peer.fingerprint,
            )
            .unwrap();
        let server = start_stage_peer(&model, &script, 1, 2, 1, &own, &peers)
            .await
            .unwrap();
        let app = server.routes.clone();
        let first = tokio::spawn({
            let app = app.clone();
            async move { app.oneshot(request()).await }
        });
        let marker = directory.path().join("model.gguf.started");
        tokio::time::timeout(Duration::from_secs(3), async {
            while !marker.exists() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        first.abort();
        let _ = first.await;
        let response = tokio::time::timeout(Duration::from_secs(5), app.oneshot(request()))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = to_bytes(response.into_body(), 32).await.unwrap();
        assert_eq!(bytes.as_ref(), &[0, 0, 0, 63, 0, 0, 128, 63]);
    }
}
