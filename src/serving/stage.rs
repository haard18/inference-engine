//! Private, bounded process protocol for one decoder stage.

use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::ServingBackend;
use crate::serving::{LEGACY_STAGE_LEASE_MS, MAX_STAGE_BATCH_FRAMES, MAX_STAGE_LEASE_MS};
#[cfg(target_os = "macos")]
use crate::MetalStageRuntime;
use crate::{load_gguf_stage, ActivationFrame, StageSession};

const MAX_SESSIONS: usize = 8;
const DEFAULT_CACHE_MIB: usize = 128;
const MAX_CACHE_MIB: usize = 8192;
const MAX_ACTIVATION_BYTES: usize = 4 * 1024 * 1024;
const MAX_SCORE_BYTES: usize = 4 * 1024 * 1024;
const MAX_COMMAND_LINE_BYTES: usize = 4096;
const SESSION_IDLE_LIMIT: Duration = Duration::from_millis(LEGACY_STAGE_LEASE_MS);

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum StageCommand {
    Reserve {
        request_id: String,
        max_positions: usize,
        #[serde(default)]
        lease_ms: Option<u64>,
    },
    Token {
        request_id: String,
        token_id: usize,
        #[serde(default)]
        lease_ms: Option<u64>,
    },
    Activation {
        request_id: String,
        payload_bytes: usize,
        #[serde(default = "one_frame")]
        frame_count: usize,
        #[serde(default)]
        lease_ms: Option<u64>,
    },
    Close {
        request_id: String,
    },
    Probe {
        request_id: String,
    },
    Rewind {
        request_id: String,
        position: usize,
    },
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum StageEvent<'a> {
    Ready {
        model_digest: String,
        layer_start: usize,
        layer_end: usize,
        hidden_size: usize,
        max_positions: usize,
        model_max_positions: usize,
        vocab_size: usize,
        stored_weight_bytes: usize,
    },
    Reserved {
        position: usize,
    },
    Activation {
        payload_bytes: usize,
    },
    Scores {
        count: usize,
    },
    Closed,
    Position {
        position: Option<usize>,
    },
    Rewound {
        position: usize,
    },
    Failed {
        message: &'a str,
    },
}

fn one_frame() -> usize {
    1
}

fn lease_duration(lease_ms: Option<u64>) -> Result<Duration, &'static str> {
    match lease_ms {
        None => Ok(SESSION_IDLE_LIMIT),
        Some(value) if (1..=MAX_STAGE_LEASE_MS).contains(&value) => {
            Ok(Duration::from_millis(value))
        }
        Some(_) => Err("invalid stage session lease"),
    }
}

fn cache_budget_bytes() -> Result<usize, String> {
    let mib = match env::var("INFERENCE_STAGE_CACHE_MIB") {
        Ok(value) => value
            .parse::<usize>()
            .map_err(|_| "INFERENCE_STAGE_CACHE_MIB must be an integer".to_owned())?,
        Err(env::VarError::NotPresent) => DEFAULT_CACHE_MIB,
        Err(env::VarError::NotUnicode(_)) => {
            return Err("INFERENCE_STAGE_CACHE_MIB must be text".into());
        }
    };
    if !(16..=MAX_CACHE_MIB).contains(&mib) {
        return Err(format!(
            "INFERENCE_STAGE_CACHE_MIB must be between 16 and {MAX_CACHE_MIB}"
        ));
    }
    mib.checked_mul(1024 * 1024)
        .ok_or_else(|| "stage cache budget overflows this platform".into())
}

fn cache_max_positions(
    model_max_positions: usize,
    bytes_per_position: usize,
    cache_budget: usize,
) -> Result<usize, String> {
    if bytes_per_position == 0 {
        return Err("stage cache size is invalid".into());
    }
    let possible = cache_budget / bytes_per_position;
    if possible == 0 || (possible < 16 && model_max_positions > possible) {
        return Err("stage cache budget is too small for this model".into());
    }
    let power_of_two = 1_usize << (usize::BITS - 1 - possible.leading_zeros());
    Ok(model_max_positions.min(power_of_two))
}

struct SessionEntry<'a> {
    session: StageSession<'a>,
    reserved_positions: usize,
    reserved_bytes: usize,
    touched: Instant,
    lease_until: Instant,
    checkpointed: bool,
}

/// Run one prefix or suffix stage in a private child process.
pub fn run_stage_worker_stdio(
    model_path: impl AsRef<Path>,
    start: usize,
    end: usize,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    run_stage_worker_stdio_with_backend(model_path, start, end, ServingBackend::Cpu)
}

pub fn run_stage_worker_stdio_with_backend(
    model_path: impl AsRef<Path>,
    start: usize,
    end: usize,
    backend: ServingBackend,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let stage = load_gguf_stage(model_path, start..end)?;
    let cache_budget = cache_budget_bytes()?;
    let bytes_per_position = stage
        .cache_bytes_per_position()
        .ok_or("stage cache size overflow")?;
    let cache_max_positions =
        cache_max_positions(stage.max_positions(), bytes_per_position, cache_budget)?;
    #[cfg(target_os = "macos")]
    let metal = (backend == ServingBackend::Metal)
        .then(|| MetalStageRuntime::new(&stage))
        .transpose()?;
    #[cfg(not(target_os = "macos"))]
    if backend == ServingBackend::Metal {
        return Err("Metal requires macOS".into());
    }
    let new_session = || -> StageSession<'_> {
        #[cfg(target_os = "macos")]
        if let Some(runtime) = &metal {
            return runtime.session();
        }
        StageSession::new(&stage)
    };
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut input = BufReader::new(stdin.lock());
    let mut output = BufWriter::new(stdout.lock());
    write_event(
        &mut output,
        &StageEvent::Ready {
            model_digest: hex::encode(stage.model_digest()),
            layer_start: start,
            layer_end: end,
            hidden_size: stage.hidden_size(),
            max_positions: cache_max_positions,
            model_max_positions: stage.max_positions(),
            vocab_size: stage.vocab_size(),
            stored_weight_bytes: stage.stored_weight_bytes(),
        },
        &[],
    )?;
    let mut sessions: HashMap<Uuid, SessionEntry<'_>> = HashMap::new();
    let mut line = Vec::new();
    loop {
        line.clear();
        let count = (&mut input)
            .take((MAX_COMMAND_LINE_BYTES + 1) as u64)
            .read_until(b'\n', &mut line)?;
        if count == 0 {
            break;
        }
        if line.len() > MAX_COMMAND_LINE_BYTES || !line.ends_with(b"\n") {
            return Err("stage command exceeds line limit".into());
        }
        let command: StageCommand = serde_json::from_slice(&line)?;
        let now = Instant::now();
        sessions.retain(|_, entry| {
            if entry.checkpointed {
                now.duration_since(entry.touched) < SESSION_IDLE_LIMIT
            } else {
                now < entry.lease_until
            }
        });
        match command {
            StageCommand::Reserve {
                request_id,
                max_positions,
                lease_ms,
            } => {
                let lease = match lease_duration(lease_ms) {
                    Ok(lease) => lease,
                    Err(message) => {
                        fail(&mut output, message)?;
                        continue;
                    }
                };
                let request_id = match Uuid::parse_str(&request_id) {
                    Ok(id) => id,
                    Err(_) => {
                        fail(&mut output, "invalid request ID")?;
                        continue;
                    }
                };
                if !(1..=cache_max_positions).contains(&max_positions) {
                    fail(&mut output, "stage cache context limit reached")?;
                    continue;
                }
                if sessions
                    .get(&request_id)
                    .is_some_and(|entry| entry.session.position() > max_positions)
                {
                    fail(&mut output, "reserved context precedes checkpoint")?;
                    continue;
                }
                let Some(reserved_bytes) = max_positions
                    .checked_next_power_of_two()
                    .map(|capacity| capacity.min(stage.max_positions()))
                    .and_then(|capacity| capacity.checked_mul(bytes_per_position))
                else {
                    fail(&mut output, "stage cache reservation overflows")?;
                    continue;
                };
                let previous = sessions.get(&request_id).map_or(0, session_charge);
                let current_allocated = sessions
                    .get(&request_id)
                    .map_or(0, |entry| entry.session.allocated_cache_bytes());
                let mut total: usize = sessions.values().map(session_charge).sum();
                total = total
                    .saturating_sub(previous)
                    .saturating_add(reserved_bytes.max(current_allocated));
                while total > cache_budget {
                    let Some(oldest) = sessions
                        .iter()
                        .filter(|(id, entry)| **id != request_id && entry.checkpointed)
                        .min_by_key(|(_, entry)| entry.touched)
                        .map(|(id, _)| *id)
                    else {
                        break;
                    };
                    if let Some(removed) = sessions.remove(&oldest) {
                        total = total.saturating_sub(session_charge(&removed));
                    }
                }
                if total > cache_budget {
                    fail(&mut output, "stage cache capacity is full")?;
                    continue;
                }
                admit(&mut sessions, request_id);
                if sessions.len() >= MAX_SESSIONS && !sessions.contains_key(&request_id) {
                    fail(&mut output, "stage session capacity is full")?;
                    continue;
                }
                let now = Instant::now();
                let entry = sessions.entry(request_id).or_insert_with(|| SessionEntry {
                    session: new_session(),
                    reserved_positions: max_positions,
                    reserved_bytes,
                    touched: now,
                    lease_until: now + lease,
                    checkpointed: false,
                });
                entry.reserved_positions = max_positions;
                entry.reserved_bytes = reserved_bytes;
                entry.touched = now;
                entry.lease_until = now + lease;
                entry.checkpointed = false;
                write_event(
                    &mut output,
                    &StageEvent::Reserved {
                        position: entry.session.position(),
                    },
                    &[],
                )?;
            }
            StageCommand::Token {
                request_id,
                token_id,
                lease_ms,
            } => {
                let lease = match lease_duration(lease_ms) {
                    Ok(lease) => lease,
                    Err(message) => {
                        fail(&mut output, message)?;
                        continue;
                    }
                };
                let request_id = match Uuid::parse_str(&request_id) {
                    Ok(id) => id,
                    Err(_) => {
                        fail(&mut output, "invalid request ID")?;
                        continue;
                    }
                };
                if !stage.is_prefix() {
                    fail(&mut output, "token input requires a prefix stage")?;
                    continue;
                }
                let Some(entry) = sessions.get_mut(&request_id) else {
                    fail(&mut output, "stage reservation is missing")?;
                    continue;
                };
                if entry.checkpointed {
                    fail(&mut output, "stage reservation is missing")?;
                    continue;
                }
                entry.lease_until = Instant::now() + lease;
                let position = entry.session.position();
                if position >= entry.reserved_positions {
                    fail(&mut output, "stage cache context limit reached")?;
                    continue;
                }
                let result = entry
                    .session
                    .forward_token(token_id)
                    .map_err(|error| error.to_string())
                    .and_then(|hidden| {
                        ActivationFrame::new(&stage, request_id, position, hidden)
                            .map(|frame| frame.encode())
                            .map_err(|error| error.to_string())
                    });
                finish_step(
                    &mut sessions,
                    request_id,
                    result,
                    &mut output,
                    true,
                    cache_budget,
                )?;
            }
            StageCommand::Activation {
                request_id,
                payload_bytes,
                frame_count,
                lease_ms,
            } => {
                let frame_bytes = stage
                    .hidden_size()
                    .checked_mul(4)
                    .and_then(|bytes| bytes.checked_add(64))
                    .ok_or("stage activation size overflows")?;
                if !(1..=MAX_STAGE_BATCH_FRAMES).contains(&frame_count)
                    || frame_count.checked_mul(frame_bytes) != Some(payload_bytes)
                    || payload_bytes > MAX_ACTIVATION_BYTES
                {
                    return Err("invalid activation payload length".into());
                }
                let mut payload = vec![0; payload_bytes];
                input.read_exact(&mut payload)?;
                let lease = match lease_duration(lease_ms) {
                    Ok(lease) => lease,
                    Err(message) => {
                        fail(&mut output, message)?;
                        continue;
                    }
                };
                let request_id = match Uuid::parse_str(&request_id) {
                    Ok(id) => id,
                    Err(_) => {
                        fail(&mut output, "invalid request ID")?;
                        continue;
                    }
                };
                if !stage.is_suffix() {
                    fail(&mut output, "activation input requires a suffix stage")?;
                    continue;
                }
                let Some(entry) = sessions.get_mut(&request_id) else {
                    fail(&mut output, "stage reservation is missing")?;
                    continue;
                };
                if entry.checkpointed {
                    fail(&mut output, "stage reservation is missing")?;
                    continue;
                }
                entry.lease_until = Instant::now() + lease;
                if entry
                    .session
                    .position()
                    .checked_add(frame_count)
                    .is_none_or(|end| end > entry.reserved_positions)
                {
                    fail(&mut output, "stage cache context limit reached")?;
                    continue;
                }
                let result = (|| -> Result<Vec<u8>, String> {
                    let mut scores = Vec::new();
                    for frame in payload.chunks_exact(frame_bytes) {
                        scores = entry
                            .session
                            .forward_frame(frame, request_id)
                            .map_err(|error| error.to_string())?;
                    }
                    let size = scores
                        .len()
                        .checked_mul(4)
                        .ok_or_else(|| "stage score size overflow".to_owned())?;
                    if size > MAX_SCORE_BYTES {
                        return Err("stage scores exceed response limit".into());
                    }
                    let mut bytes = Vec::with_capacity(scores.len() * 4);
                    for score in scores {
                        bytes.extend_from_slice(&score.to_le_bytes());
                    }
                    Ok(bytes)
                })();
                finish_step(
                    &mut sessions,
                    request_id,
                    result,
                    &mut output,
                    false,
                    cache_budget,
                )?;
            }
            StageCommand::Close { request_id } => {
                let id = match Uuid::parse_str(&request_id) {
                    Ok(id) => id,
                    Err(_) => {
                        fail(&mut output, "invalid request ID")?;
                        continue;
                    }
                };
                sessions.remove(&id);
                write_event(&mut output, &StageEvent::Closed, &[])?;
            }
            StageCommand::Probe { request_id } => {
                let id = match Uuid::parse_str(&request_id) {
                    Ok(id) => id,
                    Err(_) => {
                        fail(&mut output, "invalid request ID")?;
                        continue;
                    }
                };
                let position = sessions.get_mut(&id).map(|entry| {
                    entry.touched = Instant::now();
                    entry.session.position()
                });
                write_event(&mut output, &StageEvent::Position { position }, &[])?;
            }
            StageCommand::Rewind {
                request_id,
                position,
            } => {
                let id = match Uuid::parse_str(&request_id) {
                    Ok(id) => id,
                    Err(_) => {
                        fail(&mut output, "invalid request ID")?;
                        continue;
                    }
                };
                let Some(entry) = sessions.get_mut(&id) else {
                    fail(&mut output, "stage session is missing")?;
                    continue;
                };
                if let Err(error) = entry.session.rewind(position) {
                    fail(&mut output, &error.to_string())?;
                    continue;
                }
                entry.touched = Instant::now();
                entry.reserved_positions = position;
                entry.reserved_bytes = entry.session.allocated_cache_bytes();
                entry.checkpointed = true;
                write_event(&mut output, &StageEvent::Rewound { position }, &[])?;
            }
        }
    }
    Ok(())
}

fn session_charge(entry: &SessionEntry<'_>) -> usize {
    entry
        .reserved_bytes
        .max(entry.session.allocated_cache_bytes())
}

fn admit(sessions: &mut HashMap<Uuid, SessionEntry<'_>>, request_id: Uuid) {
    if !sessions.contains_key(&request_id) && sessions.len() >= MAX_SESSIONS {
        if let Some(oldest) = sessions
            .iter()
            .filter(|(_, entry)| entry.checkpointed)
            .min_by_key(|(_, entry)| entry.touched)
            .map(|(id, _)| *id)
        {
            sessions.remove(&oldest);
        }
    }
}

fn finish_step(
    sessions: &mut HashMap<Uuid, SessionEntry<'_>>,
    request_id: Uuid,
    result: Result<Vec<u8>, String>,
    output: &mut impl Write,
    activation: bool,
    cache_budget: usize,
) -> io::Result<()> {
    match result {
        Ok(bytes) => {
            if !activation && bytes.len() > MAX_SCORE_BYTES {
                sessions.remove(&request_id);
                return fail(output, "stage scores exceed response limit");
            }
            let mut allocated: usize = sessions.values().map(session_charge).sum();
            while allocated > cache_budget {
                let Some(oldest) = sessions
                    .iter()
                    .filter(|(id, entry)| **id != request_id && entry.checkpointed)
                    .min_by_key(|(_, entry)| entry.touched)
                    .map(|(id, _)| *id)
                else {
                    break;
                };
                if let Some(removed) = sessions.remove(&oldest) {
                    allocated = allocated.saturating_sub(session_charge(&removed));
                }
            }
            if allocated > cache_budget {
                sessions.remove(&request_id);
                return fail(output, "stage cache memory limit reached");
            }
            if let Some(entry) = sessions.get_mut(&request_id) {
                entry.touched = Instant::now();
            }
            if activation {
                write_event(
                    output,
                    &StageEvent::Activation {
                        payload_bytes: bytes.len(),
                    },
                    &bytes,
                )
            } else {
                write_event(
                    output,
                    &StageEvent::Scores {
                        count: bytes.len() / 4,
                    },
                    &bytes,
                )
            }
        }
        Err(message) => {
            sessions.remove(&request_id);
            fail(output, &message)
        }
    }
}

fn fail(output: &mut impl Write, message: &str) -> io::Result<()> {
    write_event(output, &StageEvent::Failed { message }, &[])
}

fn write_event(output: &mut impl Write, event: &StageEvent<'_>, payload: &[u8]) -> io::Result<()> {
    serde_json::to_writer(&mut *output, event).map_err(io::Error::other)?;
    output.write_all(b"\n")?;
    output.write_all(payload)?;
    output.flush()
}

#[cfg(test)]
mod tests {
    use super::cache_max_positions;

    #[test]
    fn advertised_stage_context_fits_the_cache_budget() {
        let per_position = 12 * 2 * (2048 * 4 + std::mem::size_of::<Vec<f32>>());
        assert_eq!(
            cache_max_positions(8192, per_position, 128 * 1024 * 1024).unwrap(),
            512
        );
        assert_eq!(
            cache_max_positions(8192, per_position, 2048 * 1024 * 1024).unwrap(),
            8192
        );
        assert!(cache_max_positions(8192, per_position, 1024).is_err());
    }
}
