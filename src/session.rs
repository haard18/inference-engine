#[cfg(target_os = "macos")]
use crate::metal_backend::{MetalBackend, MetalKvCache, MAX_METAL_BATCH};
use crate::model::KvCache;
use crate::{EngineError, Model};
use std::mem::size_of;
#[cfg(target_os = "macos")]
use std::sync::{Arc, Mutex};

enum Backend<'a> {
    Cpu(std::marker::PhantomData<&'a Model>),
    #[cfg(target_os = "macos")]
    Metal {
        backend: Arc<Mutex<MetalBackend>>,
        cache: MetalKvCache,
    },
}

/// State for one token-generation request.
pub struct GenerationSession<'a> {
    model: &'a Model,
    cache: KvCache,
    next_logits: Option<Vec<f32>>,
    backend: Backend<'a>,
}

pub(crate) struct SessionCheckpoint {
    position: usize,
    next_logits: Option<Vec<f32>>,
}

impl<'a> GenerationSession<'a> {
    pub fn new(model: &'a Model) -> Self {
        Self {
            model,
            cache: KvCache::new(model.config().num_layers),
            next_logits: None,
            backend: Backend::Cpu(std::marker::PhantomData),
        }
    }

    /// Run matrix operations on the Mac GPU while keeping attention and cache logic on the CPU.
    #[cfg(target_os = "macos")]
    pub fn on_metal(model: &'a Model) -> Result<Self, EngineError> {
        Ok(Self::with_metal_backend(
            model,
            Arc::new(Mutex::new(MetalBackend::new(model)?)),
        ))
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn with_metal_backend(model: &'a Model, backend: Arc<Mutex<MetalBackend>>) -> Self {
        Self {
            cache: KvCache::new(0),
            next_logits: None,
            model,
            backend: Backend::Metal {
                backend,
                cache: MetalKvCache::new(
                    model.config().num_layers,
                    model.config().num_key_value_heads
                        * (model.config().hidden_size / model.config().num_attention_heads),
                    model.config().max_positions,
                ),
            },
        }
    }

    #[cfg(not(target_os = "macos"))]
    pub fn on_metal(_model: &'a Model) -> Result<Self, EngineError> {
        Err(EngineError::Backend("Metal requires macOS".into()))
    }

    pub fn position(&self) -> usize {
        self.cache.position()
    }

    pub(crate) fn checkpoint(&self) -> SessionCheckpoint {
        SessionCheckpoint {
            position: self.cache.position(),
            next_logits: self.next_logits.clone(),
        }
    }

    pub(crate) fn rewind(&mut self, checkpoint: SessionCheckpoint) {
        self.cache.truncate(checkpoint.position);
        self.next_logits = checkpoint.next_logits;
    }

    pub(crate) fn allocated_bytes(&self) -> usize {
        self.cache.allocated_bytes()
            + self
                .next_logits
                .as_ref()
                .map_or(0, |logits| logits.capacity() * size_of::<f32>())
            + match &self.backend {
                Backend::Cpu(_) => 0,
                #[cfg(target_os = "macos")]
                Backend::Metal { cache, .. } => cache.allocated_bytes(),
            }
    }

    pub fn prefill(&mut self, tokens: &[usize]) -> Result<(), EngineError> {
        if tokens.is_empty() {
            return Err(EngineError::EmptyPrompt);
        }
        if tokens.len() > self.model.config().max_positions - self.cache.position() {
            return Err(EngineError::ContextFull);
        }
        for &token in tokens {
            if token >= self.model.config().vocab_size {
                return Err(EngineError::InvalidToken(token));
            }
        }
        #[cfg(target_os = "macos")]
        if let Backend::Metal { backend, cache } = &mut self.backend {
            let mut backend = backend
                .lock()
                .map_err(|_| EngineError::Backend("Metal runtime lock failed".into()))?;
            self.next_logits = None;
            let mut chunks = tokens.chunks(MAX_METAL_BATCH).peekable();
            while let Some(chunk) = chunks.next() {
                let is_last = chunks.peek().is_none();
                let result = objc::rc::autoreleasepool(|| {
                    self.model.prefill_tokens_metal(
                        chunk,
                        &mut self.cache,
                        cache,
                        &mut backend,
                        is_last,
                    )
                })?;
                if is_last {
                    self.next_logits = Some(result);
                }
            }
            return Ok(());
        }
        self.next_logits = None;
        for (index, &token) in tokens.iter().enumerate() {
            let is_last = index + 1 == tokens.len();
            let result = self.forward_token(token, is_last)?;
            if is_last {
                self.next_logits = Some(result);
            }
        }
        Ok(())
    }

    pub fn next_token(&mut self) -> Result<usize, EngineError> {
        let token = self.selected_token()?;
        self.advance(token)?;
        Ok(token)
    }

    /// Select a token from the current scores without running the next model step.
    pub fn selected_token(&self) -> Result<usize, EngineError> {
        let logits = self.next_logits.as_ref().ok_or(EngineError::EmptyPrompt)?;
        logits
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
            .ok_or(EngineError::InvalidConfig("model has an empty vocabulary"))
    }

    /// Add a selected token to the context so that another token can be selected.
    pub fn advance(&mut self, token: usize) -> Result<(), EngineError> {
        if self.cache.position() >= self.model.config().max_positions {
            return Err(EngineError::ContextFull);
        }
        self.next_logits = Some(self.forward_token(token, true)?);
        Ok(())
    }

    pub fn next_token_scores(&self) -> Option<&[f32]> {
        self.next_logits.as_deref()
    }

    fn forward_token(
        &mut self,
        token: usize,
        project_scores: bool,
    ) -> Result<Vec<f32>, EngineError> {
        match &mut self.backend {
            Backend::Cpu(_) => {
                self.model
                    .forward_token(token, &mut self.cache, |matrices, input| {
                        matrices
                            .iter()
                            .map(|matrix| matrix.mul_vec(input))
                            .collect()
                    })
            }
            #[cfg(target_os = "macos")]
            Backend::Metal { backend, cache } => {
                let mut backend = backend
                    .lock()
                    .map_err(|_| EngineError::Backend("Metal runtime lock failed".into()))?;
                // Long-lived workers need to drain Metal's temporary Objective-C objects
                // after each token while retained cache and model buffers stay alive.
                objc::rc::autoreleasepool(|| {
                    self.model.forward_token_metal(
                        token,
                        &mut self.cache,
                        cache,
                        &mut backend,
                        project_scores,
                    )
                })
            }
        }
    }
}
