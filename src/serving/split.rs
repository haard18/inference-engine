//! One chat request executed by a local prefix and an approved remote suffix.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::Router;
use tokio::sync::mpsc;
use uuid::Uuid;

use super::stage_peer::{StageChild, StageReady, StageStepError};
use super::{
    check_job, router, validate_config, ActiveJob, AppState, Job, JobFailure, ServingBackend,
    ServingConfig, ServingError, StageCapacitySnapshot, WorkerEvent, WorkerStatus,
    SLOW_CLIENT_TIMEOUT,
};
use crate::pool::client::PeerClient;
use crate::pool::DeviceIdentity;
use crate::{ByteBpeDecoder, ByteBpeTokenizer};

const STARTUP_TIMEOUT: Duration = Duration::from_secs(30);
const CLEANUP_TIMEOUT: Duration = Duration::from_secs(2);
const MAX_FRAME_BYTES: usize = 4 * 1024 * 1024;

/// Serve the usual chat API while each model step crosses one approved stage boundary.
/// The prefix process loads only its layer range; the suffix service must already be running.
pub async fn start_split_prefix(
    model_path: impl AsRef<Path>,
    tokenizer: ByteBpeTokenizer,
    config: ServingConfig,
    executable: impl AsRef<Path>,
    split_at: usize,
    identity: &DeviceIdentity,
    client: PeerClient,
) -> Result<(Router, tokio::task::JoinHandle<()>), ServingError> {
    validate_config(&config)?;
    if config.backend != ServingBackend::Cpu {
        return Err(ServingError::Configuration(
            "split stage execution currently requires the CPU backend",
        ));
    }
    if split_at == 0 {
        return Err(ServingError::Configuration(
            "split layer must be greater than zero",
        ));
    }
    let model_path = fs::canonicalize(model_path)
        .map_err(|error| ServingError::Worker(format!("model path: {error}")))?;
    let executable = fs::canonicalize(executable)
        .map_err(|error| ServingError::Worker(format!("worker executable: {error}")))?;
    let child = tokio::time::timeout(
        STARTUP_TIMEOUT,
        StageChild::spawn(&executable, &model_path, 0, split_at),
    )
    .await
    .map_err(|_| ServingError::Worker("prefix startup timed out".into()))?
    .map_err(ServingError::Worker)?;
    let ready = child.ready.clone();
    let remote = client
        .stage_snapshot()
        .await
        .map_err(|error| ServingError::Peer(error.to_string()))?;
    validate_pair(&ready, &remote).map_err(ServingError::Peer)?;
    if !remote.ready || remote.queue_available == 0 {
        return Err(ServingError::Peer(
            "suffix stage has no available capacity".into(),
        ));
    }
    let device_id = Uuid::parse_str(&identity.device_id)
        .map_err(|_| ServingError::Peer("device ID is invalid".into()))?;
    let (sender, receiver) = mpsc::channel(config.queue_capacity);
    let status = Arc::new(WorkerStatus::default());
    let tokenizer = Arc::new(tokenizer);
    let state = Arc::new(AppState {
        device_id,
        session_reuse: false,
        model_id: config.model_id,
        api_key: config.api_key.into_bytes(),
        tokenizer: Arc::clone(&tokenizer),
        max_positions: ready.max_positions,
        max_completion_tokens: config.max_completion_tokens,
        queue_capacity: config.queue_capacity,
        request_timeout: config.request_timeout,
        worker_status: Arc::clone(&status),
        requests: sender,
        coordinator: None,
    });
    let runtime = SplitRuntime {
        ready,
        status,
        tokenizer,
        client,
        executable,
        model_path,
    };
    let task = tokio::spawn(supervise(receiver, child, runtime));
    Ok((router(state), task))
}

fn validate_pair(prefix: &StageReady, suffix: &StageCapacitySnapshot) -> Result<(), String> {
    if prefix.layer_start != 0
        || prefix.layer_end != suffix.layer_start
        || prefix.model_digest != suffix.model_digest
        || prefix.hidden_size != suffix.hidden_size
        || prefix.vocab_size != suffix.vocab_size
        || prefix.max_positions != suffix.max_positions
        || suffix.layer_end <= suffix.layer_start
    {
        return Err("suffix stage does not match the local prefix model and range".into());
    }
    Ok(())
}

struct SplitRuntime {
    ready: StageReady,
    status: Arc<WorkerStatus>,
    tokenizer: Arc<ByteBpeTokenizer>,
    client: PeerClient,
    executable: PathBuf,
    model_path: PathBuf,
}

async fn supervise(
    mut receiver: mpsc::Receiver<Job>,
    first_child: StageChild,
    runtime: SplitRuntime,
) {
    let mut child = Some(first_child);
    loop {
        if receiver.is_closed() && receiver.is_empty() {
            break;
        }
        if child.as_mut().is_some_and(|child| !child.running()) {
            child = None;
        }
        if child.is_none() {
            runtime.status.set_unavailable(true);
            let replacement = tokio::time::timeout(
                STARTUP_TIMEOUT,
                StageChild::spawn(
                    &runtime.executable,
                    &runtime.model_path,
                    0,
                    runtime.ready.layer_end,
                ),
            )
            .await;
            match replacement {
                Ok(Ok(replacement)) if replacement.ready == runtime.ready => {
                    child = Some(replacement)
                }
                _ => {
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    continue;
                }
            }
            runtime.status.set_unavailable(false);
        }
        let job = tokio::select! {
            job = receiver.recv() => match job { Some(job) => job, None => break },
            _ = tokio::time::sleep(Duration::from_secs(1)) => continue,
        };
        if job.output.is_closed() {
            continue;
        }
        let _active = ActiveJob::new(Arc::clone(&runtime.status), job.deadline);
        let request_id = Uuid::new_v4();
        let prefix = child.as_mut().expect("prefix worker started");
        let result = run_job(
            &job,
            prefix,
            &runtime.ready,
            &runtime.client,
            &runtime.tokenizer,
            request_id,
        )
        .await;
        match &result {
            Ok((reason, completion_tokens)) => {
                send_terminal(
                    &job,
                    WorkerEvent::Finished {
                        reason,
                        completion_tokens: *completion_tokens,
                        reused_prompt_tokens: 0,
                    },
                )
                .await;
            }
            Err(JobFailure::Execution(message)) => {
                send_terminal(&job, WorkerEvent::Failed(message.clone())).await;
            }
            Err(JobFailure::Deadline) => send_terminal(&job, WorkerEvent::TimedOut).await,
            Err(JobFailure::ClientGone) => {}
        }
        // A failed step can leave the local pipe or cache out of sync. Replace that child.
        if result.is_err() {
            child = None;
        } else {
            let cleanup = tokio::time::timeout(CLEANUP_TIMEOUT, prefix.close(request_id)).await;
            if !matches!(cleanup, Ok(Ok(()))) {
                child = None;
            }
        }
        let _ = tokio::time::timeout(
            CLEANUP_TIMEOUT,
            runtime
                .client
                .close_stage(request_id, tokio::time::Instant::now() + CLEANUP_TIMEOUT),
        )
        .await;
    }
}

async fn run_job(
    job: &Job,
    prefix: &mut StageChild,
    ready: &StageReady,
    client: &PeerClient,
    tokenizer: &ByteBpeTokenizer,
    request_id: Uuid,
) -> Result<(&'static str, usize), JobFailure> {
    check_job(job)?;
    let snapshot = tokio::time::timeout_at(
        tokio::time::Instant::from_std(job.deadline),
        client.stage_snapshot(),
    )
    .await
    .map_err(|_| JobFailure::Deadline)?
    .map_err(|error| JobFailure::Execution(format!("suffix capacity: {error}")))?;
    validate_pair(ready, &snapshot).map_err(JobFailure::Execution)?;
    if !snapshot.ready || snapshot.queue_available == 0 {
        return Err(JobFailure::Execution(
            "suffix stage has no available capacity".into(),
        ));
    }
    let mut position = 0;
    let mut scores = Vec::new();
    for &token in &job.prompt {
        scores = step(job, prefix, client, ready, request_id, position, token).await?;
        position += 1;
    }
    let stop = ["<|im_end|>", "<|endoftext|>"].map(|token| tokenizer.special_token_id(token));
    let mut decoder = ByteBpeDecoder::new();
    for index in 0..job.max_tokens {
        check_job(job)?;
        let token = scores
            .iter()
            .enumerate()
            .reduce(|best, candidate| {
                if candidate.1 > best.1 {
                    candidate
                } else {
                    best
                }
            })
            .map(|(index, _)| index)
            .ok_or_else(|| JobFailure::Execution("suffix returned an empty vocabulary".into()))?;
        let token_id = u32::try_from(token)
            .map_err(|_| JobFailure::Execution("token ID exceeds tokenizer range".into()))?;
        let stopped = stop.contains(&Some(token_id));
        if !stopped {
            let text = decoder
                .push(tokenizer, token_id)
                .map_err(|error| JobFailure::Execution(error.to_string()))?;
            if !text.is_empty() {
                send(job, WorkerEvent::Delta(text)).await?;
            }
        }
        let completion_tokens = index + 1;
        if stopped || completion_tokens == job.max_tokens {
            let tail = decoder.finish();
            if !tail.is_empty() {
                send(job, WorkerEvent::Delta(tail)).await?;
            }
            return Ok((if stopped { "stop" } else { "length" }, completion_tokens));
        }
        scores = step(job, prefix, client, ready, request_id, position, token).await?;
        position += 1;
    }
    Err(JobFailure::Execution("empty generation request".into()))
}

async fn step(
    job: &Job,
    prefix: &mut StageChild,
    client: &PeerClient,
    ready: &StageReady,
    request_id: Uuid,
    position: usize,
    token: usize,
) -> Result<Vec<f32>, JobFailure> {
    check_job(job)?;
    let deadline = tokio::time::Instant::from_std(job.deadline);
    let frame = tokio::select! {
        _ = job.output.closed() => return Err(JobFailure::ClientGone),
        result = tokio::time::timeout_at(deadline, prefix.token(request_id, token)) => {
            result.map_err(|_| JobFailure::Deadline)?
                .map_err(|error| JobFailure::Execution(stage_error(error)))?
        }
    };
    validate_frame(&frame, ready, request_id, position).map_err(JobFailure::Execution)?;
    check_job(job)?;
    tokio::select! {
        _ = job.output.closed() => Err(JobFailure::ClientGone),
        result = client.forward_stage(frame, request_id, ready.vocab_size, deadline) => {
            result.map_err(|error| {
                if Instant::now() >= job.deadline {
                    JobFailure::Deadline
                } else {
                    JobFailure::Execution(format!("suffix stage failed: {error}"))
                }
            })
        }
    }
}

fn validate_frame(
    frame: &[u8],
    ready: &StageReady,
    request_id: Uuid,
    position: usize,
) -> Result<(), String> {
    let expected = ready
        .hidden_size
        .checked_mul(4)
        .and_then(|bytes| bytes.checked_add(64))
        .ok_or("activation size overflows")?;
    if frame.len() != expected || frame.len() > MAX_FRAME_BYTES || &frame[..4] != b"INFA" {
        return Err("prefix returned an invalid activation frame".into());
    }
    if u16::from_le_bytes(frame[4..6].try_into().unwrap()) != 1
        || frame[6..8] != [0, 0]
        || frame[8..40] != hex::decode(&ready.model_digest).map_err(|_| "invalid model digest")?
        || frame[40..56] != *request_id.as_bytes()
        || u32::from_le_bytes(frame[56..60].try_into().unwrap()) as usize != position
        || u32::from_le_bytes(frame[60..64].try_into().unwrap()) as usize != ready.hidden_size
        || frame[64..]
            .as_chunks::<4>()
            .0
            .iter()
            .any(|chunk| !f32::from_le_bytes(*chunk).is_finite())
    {
        return Err("prefix activation identity or values are invalid".into());
    }
    Ok(())
}

fn stage_error(error: StageStepError) -> String {
    match error {
        StageStepError::Rejected(message) | StageStepError::Broken(message) => message,
    }
}

async fn send(job: &Job, event: WorkerEvent) -> Result<(), JobFailure> {
    tokio::time::timeout_at(
        tokio::time::Instant::from_std(job.deadline)
            .min(tokio::time::Instant::now() + SLOW_CLIENT_TIMEOUT),
        job.output.send(event),
    )
    .await
    .map_err(|_| {
        if Instant::now() >= job.deadline {
            JobFailure::Deadline
        } else {
            JobFailure::ClientGone
        }
    })?
    .map_err(|_| JobFailure::ClientGone)
}

async fn send_terminal(job: &Job, event: WorkerEvent) {
    if send(job, event).await.is_ok() {
        let _ = send(job, WorkerEvent::End).await;
    }
}
