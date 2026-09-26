//! One chat request executed by a local prefix and an approved remote suffix.

use std::collections::HashMap;
use std::fs;
use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use axum::Router;
use tokio::sync::mpsc;
use uuid::Uuid;

use super::stage_peer::{StageChild, StageReady, StageStepError};
use super::{
    check_job, router, validate_config, ActiveJob, AppState, Job, JobFailure, ServingBackend,
    ServingConfig, ServingError, StageCapacitySnapshot, WorkerEvent, WorkerStatus,
    MAX_STAGE_BATCH_FRAMES, MAX_STAGE_LEASE_MS, SLOW_CLIENT_TIMEOUT,
};
use crate::pool::client::PeerClient;
use crate::pool::DeviceIdentity;
use crate::{ByteBpeDecoder, ByteBpeTokenizer};

const STARTUP_TIMEOUT: Duration = Duration::from_secs(30);
const CLEANUP_TIMEOUT: Duration = Duration::from_secs(2);
const MAX_FRAME_BYTES: usize = 4 * 1024 * 1024;
const MAX_CACHED_CONVERSATIONS: usize = 8;
const MAX_CACHE_BYTES: usize = 16 * 1024 * 1024;
const CACHE_TTL: Duration = Duration::from_secs(300);
const SUFFIX_PROBE_INTERVAL: Duration = Duration::from_secs(1);
const SUFFIX_FAILED_PROBES: u8 = 2;

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
    #[cfg(not(target_os = "macos"))]
    if config.backend == ServingBackend::Metal {
        return Err(ServingError::Configuration("Metal requires macOS"));
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
        StageChild::spawn_with_backend(&executable, &model_path, 0, split_at, config.backend),
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
        session_reuse: true,
        model_id: config.model_id,
        model_digest: ready.model_digest.clone(),
        api_key: config.api_key.into_bytes(),
        tokenizer: Arc::clone(&tokenizer),
        max_positions: ready.max_positions.min(remote.max_positions),
        max_completion_tokens: config.max_completion_tokens,
        queue_capacity: config.queue_capacity,
        request_timeout: config.request_timeout,
        worker_status: Arc::clone(&status),
        requests: sender,
        coordinator: None,
    });
    let runtime = SplitRuntime {
        ready: ready.clone(),
        status,
        tokenizer,
        client: client.clone(),
        executable,
        model_path,
        backend: config.backend,
    };
    tokio::spawn(monitor_suffix(
        Arc::downgrade(&runtime.status),
        client,
        ready,
        state.max_positions,
    ));
    let task = tokio::spawn(supervise(receiver, child, runtime));
    Ok((router(state), task))
}

async fn monitor_suffix(
    status: Weak<WorkerStatus>,
    client: PeerClient,
    ready: StageReady,
    served_positions: usize,
) {
    let mut interval = tokio::time::interval(SUFFIX_PROBE_INTERVAL);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut failed_probes = 0_u8;
    loop {
        interval.tick().await;
        let Some(status) = status.upgrade() else {
            break;
        };
        let healthy = client.stage_snapshot().await.is_ok_and(|snapshot| {
            snapshot.ready
                && snapshot.max_positions >= served_positions
                && validate_pair(&ready, &snapshot).is_ok()
        });
        if healthy {
            failed_probes = 0;
            status.set_remote_unavailable(false);
        } else {
            failed_probes = failed_probes.saturating_add(1);
            if failed_probes >= SUFFIX_FAILED_PROBES {
                status.set_remote_unavailable(true);
            }
        }
    }
}

fn validate_pair(prefix: &StageReady, suffix: &StageCapacitySnapshot) -> Result<(), String> {
    if prefix.layer_start != 0
        || prefix.layer_end != suffix.layer_start
        || prefix.model_digest != suffix.model_digest
        || prefix.hidden_size != suffix.hidden_size
        || prefix.vocab_size != suffix.vocab_size
        || prefix.model_max_positions != suffix.model_max_positions
        || suffix.max_positions == 0
        || suffix.max_positions > suffix.model_max_positions
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
    backend: ServingBackend,
}

struct CachedPrompt {
    prompt: Vec<usize>,
    stage_id: Uuid,
    scores: Vec<f32>,
    touched: Instant,
}

#[derive(Default)]
struct SplitCache {
    entries: HashMap<String, CachedPrompt>,
}

impl SplitCache {
    fn take_matching(&mut self, id: &str, prompt: &[usize]) -> Option<CachedPrompt> {
        self.expire();
        self.entries
            .remove(id)
            .filter(|entry| prompt.starts_with(&entry.prompt))
    }

    fn insert(&mut self, id: String, mut entry: CachedPrompt) {
        self.expire();
        entry.touched = Instant::now();
        self.entries.insert(id, entry);
        while self.entries.len() > MAX_CACHED_CONVERSATIONS || self.bytes() > MAX_CACHE_BYTES {
            let Some(oldest) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.touched)
                .map(|(id, _)| id.clone())
            else {
                break;
            };
            self.entries.remove(&oldest);
        }
    }

    fn expire(&mut self) {
        self.entries
            .retain(|_, entry| entry.touched.elapsed() < CACHE_TTL);
    }

    fn bytes(&self) -> usize {
        self.entries
            .iter()
            .map(|(id, entry)| {
                id.capacity()
                    + entry.prompt.capacity() * size_of::<usize>()
                    + entry.scores.capacity() * size_of::<f32>()
            })
            .sum()
    }
}

struct JobOutcome {
    reason: &'static str,
    completion_tokens: usize,
    reused_prompt_tokens: usize,
    prompt_scores: Vec<f32>,
}

async fn supervise(
    mut receiver: mpsc::Receiver<Job>,
    first_child: StageChild,
    runtime: SplitRuntime,
) {
    let mut child = Some(first_child);
    let mut cache = SplitCache::default();
    loop {
        if receiver.is_closed() && receiver.is_empty() {
            break;
        }
        if child.as_mut().is_some_and(|child| !child.running()) {
            child = None;
        }
        if child.is_none() {
            cache.entries.clear();
            runtime.status.set_unavailable(true);
            let replacement = tokio::time::timeout(
                STARTUP_TIMEOUT,
                StageChild::spawn_with_backend(
                    &runtime.executable,
                    &runtime.model_path,
                    0,
                    runtime.ready.layer_end,
                    runtime.backend,
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
        let mut request_id = Uuid::new_v4();
        let mut prefix_touched = false;
        let prefix = child.as_mut().expect("prefix worker started");
        let result = run_job(
            &job,
            prefix,
            &runtime,
            &mut cache,
            &mut request_id,
            &mut prefix_touched,
        )
        .await;
        match &result {
            Ok(outcome) => {
                send_terminal(
                    &job,
                    WorkerEvent::Finished {
                        reason: outcome.reason,
                        completion_tokens: outcome.completion_tokens,
                        reused_prompt_tokens: outcome.reused_prompt_tokens,
                    },
                )
                .await;
            }
            Err(JobFailure::Execution(message)) => {
                send_terminal(&job, WorkerEvent::Failed(message.clone())).await;
            }
            Err(JobFailure::Overloaded(message)) => {
                send_terminal(&job, WorkerEvent::Overloaded(message.clone())).await;
            }
            Err(JobFailure::Deadline) => send_terminal(&job, WorkerEvent::TimedOut).await,
            Err(JobFailure::ClientGone) => {}
        }
        let mut cached = false;
        let mut discard_child = result.is_err() && prefix_touched;
        if let (Ok(outcome), Some(conversation_id)) = (&result, &job.conversation_id) {
            let deadline = tokio::time::Instant::from_std(job.deadline)
                .min(tokio::time::Instant::now() + CLEANUP_TIMEOUT);
            let local_ok = matches!(
                tokio::time::timeout_at(deadline, prefix.rewind(request_id, job.prompt.len()))
                    .await,
                Ok(Ok(()))
            );
            let remote_ok = local_ok
                && runtime
                    .client
                    .rewind_stage(request_id, job.prompt.len(), deadline)
                    .await
                    .is_ok();
            if local_ok && remote_ok {
                cache.insert(
                    conversation_id.clone(),
                    CachedPrompt {
                        prompt: job.prompt.clone(),
                        stage_id: request_id,
                        scores: outcome.prompt_scores.clone(),
                        touched: Instant::now(),
                    },
                );
                cached = true;
            }
            if !local_ok {
                discard_child = true;
            }
        }
        // A failed step can leave the local pipe or cache out of sync. Replace that child.
        if !cached && !discard_child {
            let cleanup = tokio::time::timeout(CLEANUP_TIMEOUT, prefix.close(request_id)).await;
            if !matches!(cleanup, Ok(Ok(()))) {
                discard_child = true;
            }
        }
        if discard_child {
            child = None;
        }
        if !cached {
            let _ = tokio::time::timeout(
                CLEANUP_TIMEOUT,
                runtime
                    .client
                    .close_stage(request_id, tokio::time::Instant::now() + CLEANUP_TIMEOUT),
            )
            .await;
        }
    }
}

async fn run_job(
    job: &Job,
    prefix: &mut StageChild,
    runtime: &SplitRuntime,
    cache: &mut SplitCache,
    request_id: &mut Uuid,
    prefix_touched: &mut bool,
) -> Result<JobOutcome, JobFailure> {
    let ready = &runtime.ready;
    let client = &runtime.client;
    let tokenizer = runtime.tokenizer.as_ref();
    check_job(job)?;
    let snapshot = tokio::time::timeout_at(
        tokio::time::Instant::from_std(job.deadline),
        client.stage_snapshot(),
    )
    .await
    .map_err(|_| JobFailure::Deadline)?
    .map_err(|error| JobFailure::Execution(format!("suffix capacity: {error}")))?;
    validate_pair(ready, &snapshot).map_err(JobFailure::Execution)?;
    if !snapshot.ready {
        return Err(JobFailure::Execution("suffix stage is unavailable".into()));
    }
    if snapshot.queue_available == 0 {
        return Err(JobFailure::Overloaded(
            "suffix stage has no available capacity".into(),
        ));
    }
    let cached = job
        .conversation_id
        .as_deref()
        .and_then(|id| cache.take_matching(id, &job.prompt));
    let (mut position, mut scores) = if let Some(entry) = cached {
        let expected = entry.prompt.len();
        let deadline = tokio::time::Instant::from_std(job.deadline);
        *prefix_touched = true;
        let local = tokio::time::timeout_at(deadline, prefix.probe(entry.stage_id))
            .await
            .map_err(|_| JobFailure::Deadline)?
            .map_err(|error| JobFailure::Execution(stage_error(error)))?;
        let remote = client
            .probe_stage(entry.stage_id, deadline)
            .await
            .map_err(|error| JobFailure::Execution(format!("suffix checkpoint: {error}")))?;
        if local == Some(expected) && remote == Some(expected) {
            *request_id = entry.stage_id;
            (expected, entry.scores)
        } else {
            (0, Vec::new())
        }
    } else {
        (0, Vec::new())
    };
    let requested_positions = job
        .prompt
        .len()
        .checked_add(job.max_tokens)
        .filter(|positions| *positions <= ready.max_positions.min(snapshot.max_positions))
        .ok_or_else(|| JobFailure::Execution("request exceeds stage context".into()))?;
    let deadline = tokio::time::Instant::from_std(job.deadline);
    let remote_position = client
        .reserve_stage(*request_id, requested_positions, deadline)
        .await
        .map_err(|error| match error {
            crate::pool::PoolError::Overloaded(message) => JobFailure::Overloaded(message),
            other => JobFailure::Execution(format!("suffix reservation: {other}")),
        })?;
    let remaining = deadline
        .checked_duration_since(tokio::time::Instant::now())
        .ok_or(JobFailure::Deadline)?;
    let lease_ms = remaining
        .as_millis()
        .clamp(1, u128::from(MAX_STAGE_LEASE_MS)) as u64;
    let local_position = tokio::time::timeout_at(
        deadline,
        prefix.reserve(*request_id, requested_positions, lease_ms),
    )
    .await
    .map_err(|_| JobFailure::Deadline)?
    .map_err(|error| match error {
        StageStepError::Rejected(message) if message.contains("capacity") => {
            JobFailure::Overloaded(message)
        }
        other => JobFailure::Execution(stage_error(other)),
    })?;
    if local_position != position || remote_position != position {
        return Err(JobFailure::Execution(
            "stage reservation changed the prompt position".into(),
        ));
    }
    let reused_prompt_tokens = position;
    let frame_bytes = ready
        .hidden_size
        .checked_mul(4)
        .and_then(|bytes| bytes.checked_add(64))
        .ok_or_else(|| JobFailure::Execution("activation size overflows".into()))?;
    let frames_per_batch = (MAX_FRAME_BYTES / frame_bytes).min(MAX_STAGE_BATCH_FRAMES);
    if frames_per_batch == 0 {
        return Err(JobFailure::Execution("activation exceeds limit".into()));
    }
    for tokens in job.prompt[position..].chunks(frames_per_batch) {
        let batch_bytes = frame_bytes
            .checked_mul(tokens.len())
            .filter(|bytes| *bytes <= MAX_FRAME_BYTES)
            .ok_or_else(|| JobFailure::Execution("activation batch exceeds limit".into()))?;
        let mut frames = Vec::with_capacity(batch_bytes);
        for &token in tokens {
            *prefix_touched = true;
            frames.extend(prefix_frame(job, prefix, ready, *request_id, position, token).await?);
            position += 1;
        }
        scores = forward_suffix(job, client, frames, tokens.len(), *request_id, ready).await?;
    }
    let prompt_scores = scores.clone();
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
            return Ok(JobOutcome {
                reason: if stopped { "stop" } else { "length" },
                completion_tokens,
                reused_prompt_tokens,
                prompt_scores,
            });
        }
        *prefix_touched = true;
        scores = step(job, prefix, client, ready, *request_id, position, token).await?;
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
    let frame = prefix_frame(job, prefix, ready, request_id, position, token).await?;
    forward_suffix(job, client, frame, 1, request_id, ready).await
}

async fn prefix_frame(
    job: &Job,
    prefix: &mut StageChild,
    ready: &StageReady,
    request_id: Uuid,
    position: usize,
    token: usize,
) -> Result<Vec<u8>, JobFailure> {
    check_job(job)?;
    let deadline = tokio::time::Instant::from_std(job.deadline);
    let lease_ms = job
        .deadline
        .checked_duration_since(Instant::now())
        .map_or(1, |remaining| {
            remaining
                .as_millis()
                .clamp(1, u128::from(MAX_STAGE_LEASE_MS)) as u64
        });
    let frame = tokio::select! {
        _ = job.output.closed() => return Err(JobFailure::ClientGone),
        result = tokio::time::timeout_at(deadline, prefix.token(request_id, token, lease_ms)) => {
            result.map_err(|_| JobFailure::Deadline)?
                .map_err(|error| JobFailure::Execution(stage_error(error)))?
        }
    };
    validate_frame(&frame, ready, request_id, position).map_err(JobFailure::Execution)?;
    check_job(job)?;
    Ok(frame)
}

async fn forward_suffix(
    job: &Job,
    client: &PeerClient,
    frames: Vec<u8>,
    frame_count: usize,
    request_id: Uuid,
    ready: &StageReady,
) -> Result<Vec<f32>, JobFailure> {
    check_job(job)?;
    let deadline = tokio::time::Instant::from_std(job.deadline);
    tokio::select! {
        _ = job.output.closed() => Err(JobFailure::ClientGone),
        result = client.forward_stage_batch(frames, frame_count, request_id, ready.vocab_size, deadline) => {
            result.map_err(|error| {
                if Instant::now() >= job.deadline {
                    JobFailure::Deadline
                } else {
                    match error {
                        crate::pool::PoolError::Overloaded(message) => JobFailure::Overloaded(message),
                        other => JobFailure::Execution(format!("suffix stage failed: {other}")),
                    }
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
