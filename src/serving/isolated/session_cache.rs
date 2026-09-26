//! Bounded prompt-prefix checkpoints retained inside one model worker process.

use std::collections::HashMap;
use std::mem::size_of;
use std::time::{Duration, Instant};

use crate::GenerationSession;

const MAX_SESSIONS: usize = 8;
const MAX_CACHE_BYTES: usize = 128 * 1024 * 1024;
const SESSION_TTL: Duration = Duration::from_secs(5 * 60);

struct Entry<'a> {
    prompt: Vec<usize>,
    session: GenerationSession<'a>,
    last_used: Instant,
    bytes: usize,
}

pub(super) struct SessionCache<'a> {
    entries: HashMap<String, Entry<'a>>,
    bytes: usize,
}

impl<'a> SessionCache<'a> {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            bytes: 0,
        }
    }

    pub fn take_matching(
        &mut self,
        conversation_id: &str,
        prompt: &[usize],
    ) -> Option<(GenerationSession<'a>, usize)> {
        self.remove_expired();
        let entry = self.entries.remove(conversation_id)?;
        self.bytes -= entry.bytes;
        if !prompt.starts_with(&entry.prompt) {
            return None;
        }
        let reused = entry.prompt.len();
        Some((entry.session, reused))
    }

    pub fn insert(
        &mut self,
        conversation_id: String,
        prompt: Vec<usize>,
        session: GenerationSession<'a>,
    ) {
        self.remove_expired();
        if let Some(old) = self.entries.remove(&conversation_id) {
            self.bytes -= old.bytes;
        }
        let bytes = session.allocated_bytes()
            + prompt.capacity() * size_of::<usize>()
            + conversation_id.capacity();
        if bytes > MAX_CACHE_BYTES {
            return;
        }
        while self.entries.len() >= MAX_SESSIONS || self.bytes + bytes > MAX_CACHE_BYTES {
            let Some(oldest) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(id, _)| id.clone())
            else {
                break;
            };
            if let Some(entry) = self.entries.remove(&oldest) {
                self.bytes -= entry.bytes;
            }
        }
        self.bytes += bytes;
        self.entries.insert(
            conversation_id,
            Entry {
                prompt,
                session,
                last_used: Instant::now(),
                bytes,
            },
        );
    }

    fn remove_expired(&mut self) {
        let now = Instant::now();
        self.entries.retain(|_, entry| {
            let fresh = now.duration_since(entry.last_used) < SESSION_TTL;
            if !fresh {
                self.bytes -= entry.bytes;
            }
            fresh
        });
    }
}
