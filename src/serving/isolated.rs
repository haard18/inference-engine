//! Restartable model process for the local HTTP server.

use std::error::Error;
use std::fs;
use std::io::{self, BufRead, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::Router;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::mpsc;

use super::{
    router, validate_config, ActiveJob, AppState, Job, ServingBackend, ServingConfig, ServingError,
    WorkerEvent, WorkerStatus, SLOW_CLIENT_TIMEOUT,
};
#[cfg(target_os = "macos")]
use crate::MetalRuntime;
use crate::{load_gguf, load_gguf_tokenizer, ByteBpeDecoder, ByteBpeTokenizer, GenerationSession};

const STARTUP_TIMEOUT: Duration = Duration::from_secs(30);
const RESTART_DELAY: Duration = Duration::from_secs(1);

#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum WireEvent {
    Ready {
        max_positions: usize,
    },
    Delta {
        text: String,
    },
    Finished {
        reason: String,
        completion_tokens: usize,
    },
    Failed {
        message: String,
    },
}

#[derive(Serialize, Deserialize)]
struct WireRequest {
    prompt: Vec<usize>,
    max_tokens: usize,
}

struct ChildWorker {
    child: Child,
    stdin: ChildStdin,
    lines: Lines<BufReader<ChildStdout>>,
}

enum ProcessFailure {
    Model(String),
    Deadline,
    ClientGone,
    Broken(String),
}

/// Start the HTTP server with model execution in a restartable child process.
pub async fn start_isolated(
    model_path: impl AsRef<Path>,
    tokenizer: ByteBpeTokenizer,
    config: ServingConfig,
    executable: impl AsRef<Path>,
) -> Result<(Router, tokio::task::JoinHandle<()>), ServingError> {
    validate_config(&config)?;
    let model_path = fs::canonicalize(model_path)
        .map_err(|error| ServingError::Worker(format!("model path: {error}")))?;
    let executable = fs::canonicalize(executable)
        .map_err(|error| ServingError::Worker(format!("worker executable: {error}")))?;
    let (child, max_positions) = spawn_child(&executable, &model_path, config.backend)
        .await
        .map_err(ServingError::Worker)?;
    let (sender, receiver) = mpsc::channel(config.queue_capacity);
    let worker_status = Arc::new(WorkerStatus::default());
    let state = Arc::new(AppState {
        model_id: config.model_id,
        api_key: config.api_key.into_bytes(),
        tokenizer: Arc::new(tokenizer),
        max_positions,
        max_completion_tokens: config.max_completion_tokens,
        request_timeout: config.request_timeout,
        worker_status: Arc::clone(&worker_status),
        requests: sender,
    });
    let task = tokio::spawn(supervise(
        receiver,
        child,
        worker_status,
        executable,
        model_path,
        config.backend,
    ));
    Ok((router(state), task))
}

async fn spawn_child(
    executable: &Path,
    model_path: &Path,
    backend: ServingBackend,
) -> Result<(ChildWorker, usize), String> {
    let mut command = Command::new(executable);
    command
        .arg("--internal-worker")
        .arg(model_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .env_remove("INFERENCE_API_KEY")
        .kill_on_drop(true);
    if backend == ServingBackend::Metal {
        command.arg("--metal");
    }
    let mut child = command.spawn().map_err(|error| error.to_string())?;
    let stdin = child.stdin.take().ok_or("worker stdin is unavailable")?;
    let stdout = child.stdout.take().ok_or("worker stdout is unavailable")?;
    let mut lines = BufReader::new(stdout).lines();
    let ready = tokio::time::timeout(STARTUP_TIMEOUT, lines.next_line()).await;
    let max_positions = match ready {
        Ok(Ok(Some(line))) => match serde_json::from_str::<WireEvent>(&line) {
            Ok(WireEvent::Ready { max_positions }) if max_positions > 0 => max_positions,
            _ => {
                let _ = child.kill().await;
                return Err("worker sent an invalid startup response".into());
            }
        },
        Ok(Ok(None)) => {
            let _ = child.wait().await;
            return Err("worker exited before becoming ready".into());
        }
        Ok(Err(error)) => {
            let _ = child.kill().await;
            return Err(format!("worker startup read failed: {error}"));
        }
        Err(_) => {
            let _ = child.kill().await;
            return Err("worker did not become ready within 30 seconds".into());
        }
    };
    Ok((
        ChildWorker {
            child,
            stdin,
            lines,
        },
        max_positions,
    ))
}

async fn supervise(
    mut receiver: mpsc::Receiver<Job>,
    first_child: ChildWorker,
    status: Arc<WorkerStatus>,
    executable: PathBuf,
    model_path: PathBuf,
    backend: ServingBackend,
) {
    let mut worker = Some(first_child);
    loop {
        if worker.is_none() {
            if receiver.is_closed() {
                break;
            }
            match spawn_child(&executable, &model_path, backend).await {
                Ok((child, _)) => {
                    worker = Some(child);
                    status.set_unavailable(false);
                }
                Err(_) => {
                    status.set_unavailable(true);
                    tokio::time::sleep(RESTART_DELAY).await;
                    continue;
                }
            }
        }
        let job = tokio::select! {
            job = receiver.recv() => job,
            _ = tokio::time::sleep(Duration::from_millis(500)) => {
                if worker.as_mut().is_some_and(|child| match child.child.try_wait() {
                    Ok(None) => false,
                    Ok(Some(_)) | Err(_) => true,
                }) {
                    status.set_unavailable(true);
                    if let Some(mut child) = worker.take() {
                        let _ = child.child.kill().await;
                    }
                }
                continue;
            }
        };
        let Some(job) = job else { break };
        if job.output.is_closed() || Instant::now() >= job.deadline {
            continue;
        }
        let outcome = {
            let _active = ActiveJob::new(Arc::clone(&status), job.deadline);
            run_child_job(worker.as_mut().expect("worker is available"), &job).await
        };
        let restart = match outcome {
            Ok((reason, completion_tokens)) => {
                send_result(
                    &job,
                    WorkerEvent::Finished {
                        reason,
                        completion_tokens,
                    },
                )
                .await;
                false
            }
            Err(ProcessFailure::Model(message)) => {
                send_result(&job, WorkerEvent::Failed(message)).await;
                false
            }
            Err(ProcessFailure::Deadline) => {
                send_result(&job, WorkerEvent::TimedOut).await;
                true
            }
            Err(ProcessFailure::ClientGone) => true,
            Err(ProcessFailure::Broken(message)) => {
                send_result(&job, WorkerEvent::Failed(message)).await;
                true
            }
        };
        if restart {
            status.set_unavailable(true);
            if let Some(mut child) = worker.take() {
                let _ = child.child.kill().await;
            }
        }
    }
    if let Some(mut child) = worker {
        let _ = child.child.kill().await;
    }
}

async fn send_result(job: &Job, event: WorkerEvent) {
    let deadline = tokio::time::Instant::from_std(job.deadline);
    if tokio::time::timeout_at(deadline, job.output.send(event))
        .await
        .is_ok_and(|result| result.is_ok())
    {
        let _ = tokio::time::timeout_at(deadline, job.output.send(WorkerEvent::End)).await;
    }
}

async fn run_child_job(
    worker: &mut ChildWorker,
    job: &Job,
) -> Result<(&'static str, usize), ProcessFailure> {
    let request = WireRequest {
        prompt: job.prompt.clone(),
        max_tokens: job.max_tokens,
    };
    let mut bytes =
        serde_json::to_vec(&request).map_err(|error| ProcessFailure::Broken(error.to_string()))?;
    bytes.push(b'\n');
    let deadline = tokio::time::Instant::from_std(job.deadline);
    tokio::select! {
        _ = job.output.closed() => return Err(ProcessFailure::ClientGone),
        written = tokio::time::timeout_at(deadline, worker.stdin.write_all(&bytes)) => {
            match written {
                Ok(Ok(())) => {},
                Ok(Err(error)) => return Err(ProcessFailure::Broken(format!("worker input failed: {error}"))),
                Err(_) => return Err(ProcessFailure::Deadline),
            }
        }
    }
    loop {
        if Instant::now() >= job.deadline {
            return Err(ProcessFailure::Deadline);
        }
        let line = tokio::select! {
            _ = job.output.closed() => return Err(ProcessFailure::ClientGone),
            received = tokio::time::timeout_at(deadline, worker.lines.next_line()) => {
                match received {
                    Ok(Ok(Some(line))) => line,
                    Ok(Ok(None)) => return Err(ProcessFailure::Broken("worker exited during inference".into())),
                    Ok(Err(error)) => return Err(ProcessFailure::Broken(format!("worker output failed: {error}"))),
                    Err(_) => return Err(ProcessFailure::Deadline),
                }
            }
        };
        match serde_json::from_str::<WireEvent>(&line) {
            Ok(WireEvent::Delta { text }) => {
                let send_deadline = job.deadline.min(Instant::now() + SLOW_CLIENT_TIMEOUT);
                match tokio::time::timeout_at(
                    tokio::time::Instant::from_std(send_deadline),
                    job.output.send(WorkerEvent::Delta(text)),
                )
                .await
                {
                    Ok(Ok(())) => {}
                    _ => return Err(ProcessFailure::ClientGone),
                }
            }
            Ok(WireEvent::Finished {
                reason,
                completion_tokens,
            }) => {
                let reason = match reason.as_str() {
                    "stop" => "stop",
                    "length" => "length",
                    _ => {
                        return Err(ProcessFailure::Broken(
                            "worker sent an invalid finish reason".into(),
                        ))
                    }
                };
                return Ok((reason, completion_tokens));
            }
            Ok(WireEvent::Failed { message }) => return Err(ProcessFailure::Model(message)),
            _ => {
                return Err(ProcessFailure::Broken(
                    "worker sent an invalid inference event".into(),
                ))
            }
        }
    }
}

/// Run the private model process protocol on standard input and output.
pub fn run_worker_stdio(
    model_path: impl AsRef<Path>,
    backend: ServingBackend,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let model = load_gguf(&model_path)?;
    let tokenizer = load_gguf_tokenizer(&model_path)?;
    #[cfg(target_os = "macos")]
    let metal = if backend == ServingBackend::Metal {
        Some(MetalRuntime::new(&model)?)
    } else {
        None
    };
    #[cfg(not(target_os = "macos"))]
    if backend == ServingBackend::Metal {
        return Err("Metal requires macOS".into());
    }
    let stdout = io::stdout();
    let mut output = BufWriter::new(stdout.lock());
    write_event(
        &mut output,
        &WireEvent::Ready {
            max_positions: model.config().max_positions,
        },
    )?;
    for line in io::stdin().lock().lines() {
        let line = line?;
        let request: WireRequest = serde_json::from_str(&line)?;
        if request.max_tokens == 0 || request.max_tokens > 4096 {
            write_event(
                &mut output,
                &WireEvent::Failed {
                    message: "invalid generation length".into(),
                },
            )?;
            continue;
        }
        #[cfg(target_os = "macos")]
        let mut session = match metal.as_ref() {
            Some(runtime) => runtime.session(),
            None => GenerationSession::new(&model),
        };
        #[cfg(not(target_os = "macos"))]
        let mut session = GenerationSession::new(&model);
        if let Err(message) = generate_in_child(&mut session, &tokenizer, &request, &mut output) {
            write_event(&mut output, &WireEvent::Failed { message })?;
        }
    }
    Ok(())
}

fn generate_in_child(
    session: &mut GenerationSession<'_>,
    tokenizer: &ByteBpeTokenizer,
    request: &WireRequest,
    output: &mut impl Write,
) -> Result<(), String> {
    session
        .prefill(&request.prompt)
        .map_err(|error| error.to_string())?;
    let stop = ["<|im_end|>", "<|endoftext|>"].map(|token| tokenizer.special_token_id(token));
    let mut decoder = ByteBpeDecoder::new();
    for step in 0..request.max_tokens {
        let token = session
            .selected_token()
            .map_err(|error| error.to_string())?;
        let token_id =
            u32::try_from(token).map_err(|_| "token ID exceeds tokenizer range".to_owned())?;
        let stopped = stop.contains(&Some(token_id));
        if !stopped {
            let text = decoder
                .push(tokenizer, token_id)
                .map_err(|error| error.to_string())?;
            if !text.is_empty() {
                write_event(output, &WireEvent::Delta { text })
                    .map_err(|error| error.to_string())?;
            }
        }
        let completion_tokens = step + 1;
        if stopped || completion_tokens == request.max_tokens {
            let tail = decoder.finish();
            if !tail.is_empty() {
                write_event(output, &WireEvent::Delta { text: tail })
                    .map_err(|error| error.to_string())?;
            }
            let reason = if stopped { "stop" } else { "length" };
            write_event(
                output,
                &WireEvent::Finished {
                    reason: reason.into(),
                    completion_tokens,
                },
            )
            .map_err(|error| error.to_string())?;
            return Ok(());
        }
        session.advance(token).map_err(|error| error.to_string())?;
    }
    Err("empty generation request".into())
}

fn write_event(output: &mut impl Write, event: &WireEvent) -> io::Result<()> {
    serde_json::to_writer(&mut *output, event).map_err(io::Error::other)?;
    output.write_all(b"\n")?;
    output.flush()
}

#[cfg(all(test, unix))]
mod tests {
    use std::os::unix::fs::PermissionsExt;

    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use serde_json::{json, Value};
    use tower::ServiceExt;

    use super::*;

    fn test_tokenizer() -> ByteBpeTokenizer {
        let mut pieces = vec!["<|im_start|>".into(), "<|im_end|>".into()];
        let mut types = vec![3, 3];
        for piece in [
            "a", "s", "i", "t", "n", "u", "e", "r", "y", "h", "S", "Ċ", "Ġ",
        ] {
            pieces.push(piece.into());
            types.push(1);
        }
        ByteBpeTokenizer::from_gguf_parts(pieces, Vec::new(), types).unwrap()
    }

    fn chat_request() -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .header(
                "authorization",
                "Bearer test-only-key-with-at-least-32-characters",
            )
            .body(Body::from(
                json!({
                    "model": "local-smollm2",
                    "messages": [{"role": "user", "content": "Say hi"}],
                    "max_completion_tokens": 2
                })
                .to_string(),
            ))
            .unwrap()
    }

    async fn json_body(response: axum::response::Response) -> Value {
        let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn hung_child_is_killed_and_next_requests_use_replacement() {
        let directory = tempfile::tempdir().unwrap();
        let script = directory.path().join("fake-worker");
        fs::write(
            &script,
            r#"#!/bin/sh
model_dir="$2"
printf '%s\n' '{"kind":"ready","max_positions":2048}'
while IFS= read -r request; do
  if [ ! -f "$model_dir/seen" ]; then
    : > "$model_dir/seen"
    IFS= read -r ignored
  else
    printf '%s\n' '{"kind":"delta","text":"ok"}'
    printf '%s\n' '{"kind":"finished","reason":"length","completion_tokens":1}'
    exit 0
  fi
done
"#,
        )
        .unwrap();
        let mut permissions = fs::metadata(&script).unwrap().permissions();
        permissions.set_mode(0o700);
        fs::set_permissions(&script, permissions).unwrap();

        let (app, supervisor) = start_isolated(
            directory.path(),
            test_tokenizer(),
            ServingConfig {
                model_id: "local-smollm2".into(),
                api_key: "test-only-key-with-at-least-32-characters".into(),
                queue_capacity: 1,
                max_completion_tokens: 4,
                request_timeout: Duration::from_millis(100),
                backend: ServingBackend::Cpu,
            },
            &script,
        )
        .await
        .unwrap();
        let response = app.clone().oneshot(chat_request()).await.unwrap();
        assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
        assert_eq!(
            json_body(response).await["error"]["code"],
            "request_timeout"
        );

        let mut recovered = false;
        for _ in 0..30 {
            tokio::time::sleep(Duration::from_millis(50)).await;
            let response = app.clone().oneshot(chat_request()).await.unwrap();
            if response.status() == StatusCode::OK {
                assert_eq!(
                    json_body(response).await["choices"][0]["message"]["content"],
                    "ok"
                );
                recovered = true;
                break;
            }
        }
        assert!(
            recovered,
            "server did not recover after killing a hung worker"
        );

        let mut recovered_after_exit = false;
        for _ in 0..30 {
            tokio::time::sleep(Duration::from_millis(50)).await;
            let response = app.clone().oneshot(chat_request()).await.unwrap();
            if response.status() == StatusCode::OK {
                recovered_after_exit = true;
                break;
            }
        }
        assert!(
            recovered_after_exit,
            "server did not recover after worker exit"
        );
        drop(app);
        tokio::time::timeout(Duration::from_secs(3), supervisor)
            .await
            .unwrap()
            .unwrap();
    }
}
