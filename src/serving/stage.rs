//! Private, bounded process protocol for one decoder stage.

use std::collections::HashMap;
use std::error::Error;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{load_gguf_stage, ActivationFrame, StageSession};

const MAX_SESSIONS: usize = 8;
const MAX_CACHE_BYTES: usize = 128 * 1024 * 1024;
const MAX_ACTIVATION_BYTES: usize = 4 * 1024 * 1024;
const MAX_SCORE_BYTES: usize = 4 * 1024 * 1024;
const MAX_COMMAND_LINE_BYTES: usize = 4096;
const SESSION_IDLE_LIMIT: Duration = Duration::from_secs(300);

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum StageCommand {
    Token {
        request_id: String,
        token_id: usize,
    },
    Activation {
        request_id: String,
        payload_bytes: usize,
    },
    Close {
        request_id: String,
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
        vocab_size: usize,
        stored_weight_bytes: usize,
    },
    Activation {
        payload_bytes: usize,
    },
    Scores {
        count: usize,
    },
    Closed,
    Failed {
        message: &'a str,
    },
}

struct SessionEntry<'a> {
    session: StageSession<'a>,
    touched: Instant,
}

/// Run one prefix or suffix stage in a private child process.
pub fn run_stage_worker_stdio(
    model_path: impl AsRef<Path>,
    start: usize,
    end: usize,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let stage = load_gguf_stage(model_path, start..end)?;
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
            max_positions: stage.max_positions(),
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
        sessions.retain(|_, entry| entry.touched.elapsed() < SESSION_IDLE_LIMIT);
        match command {
            StageCommand::Token {
                request_id,
                token_id,
            } => {
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
                if !admit(&sessions, request_id, &mut output)? {
                    continue;
                }
                let entry = sessions.entry(request_id).or_insert_with(|| SessionEntry {
                    session: StageSession::new(&stage),
                    touched: Instant::now(),
                });
                let position = entry.session.position();
                let result = entry
                    .session
                    .forward_token(token_id)
                    .map_err(|error| error.to_string())
                    .and_then(|hidden| {
                        ActivationFrame::new(&stage, request_id, position, hidden)
                            .map(|frame| frame.encode())
                            .map_err(|error| error.to_string())
                    });
                finish_step(&mut sessions, request_id, result, &mut output, true)?;
            }
            StageCommand::Activation {
                request_id,
                payload_bytes,
            } => {
                if !(64..=MAX_ACTIVATION_BYTES).contains(&payload_bytes) {
                    return Err("invalid activation payload length".into());
                }
                let mut payload = vec![0; payload_bytes];
                input.read_exact(&mut payload)?;
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
                if !admit(&sessions, request_id, &mut output)? {
                    continue;
                }
                let entry = sessions.entry(request_id).or_insert_with(|| SessionEntry {
                    session: StageSession::new(&stage),
                    touched: Instant::now(),
                });
                let result = entry
                    .session
                    .forward_frame(&payload, request_id)
                    .map_err(|error| error.to_string())
                    .and_then(|scores| {
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
                    });
                finish_step(&mut sessions, request_id, result, &mut output, false)?;
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
        }
    }
    Ok(())
}

fn admit(
    sessions: &HashMap<Uuid, SessionEntry<'_>>,
    request_id: Uuid,
    output: &mut impl Write,
) -> io::Result<bool> {
    if !sessions.contains_key(&request_id) && sessions.len() >= MAX_SESSIONS {
        fail(output, "stage session capacity is full")?;
        return Ok(false);
    }
    Ok(true)
}

fn finish_step(
    sessions: &mut HashMap<Uuid, SessionEntry<'_>>,
    request_id: Uuid,
    result: Result<Vec<u8>, String>,
    output: &mut impl Write,
    activation: bool,
) -> io::Result<()> {
    match result {
        Ok(bytes) => {
            if !activation && bytes.len() > MAX_SCORE_BYTES {
                sessions.remove(&request_id);
                return fail(output, "stage scores exceed response limit");
            }
            let allocated: usize = sessions
                .values()
                .map(|entry| entry.session.allocated_cache_bytes())
                .sum();
            if allocated > MAX_CACHE_BYTES {
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
