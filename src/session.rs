#[cfg(target_os = "macos")]
use crate::metal_backend::MetalBackend;
use crate::model::KvCache;
use crate::{EngineError, Model};
#[cfg(target_os = "macos")]
use std::sync::{Arc, Mutex};

enum Backend<'a> {
    Cpu(std::marker::PhantomData<&'a Model>),
    #[cfg(target_os = "macos")]
    Metal(Arc<Mutex<MetalBackend<'a>>>),
}

/// State for one token-generation request.
pub struct GenerationSession<'a> {
    model: &'a Model,
    cache: KvCache,
    next_logits: Option<Vec<f32>>,
    backend: Backend<'a>,
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
    pub(crate) fn with_metal_backend(
        model: &'a Model,
        backend: Arc<Mutex<MetalBackend<'a>>>,
    ) -> Self {
        Self {
            cache: KvCache::new(model.config().num_layers),
            next_logits: None,
            model,
            backend: Backend::Metal(backend),
        }
    }

    #[cfg(not(target_os = "macos"))]
    pub fn on_metal(_model: &'a Model) -> Result<Self, EngineError> {
        Err(EngineError::Backend("Metal requires macOS".into()))
    }

    pub fn position(&self) -> usize {
        self.cache.position()
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
        for &token in tokens {
            self.next_logits = Some(self.forward_token(token)?);
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
        self.next_logits = Some(self.forward_token(token)?);
        Ok(())
    }

    pub fn next_token_scores(&self) -> Option<&[f32]> {
        self.next_logits.as_deref()
    }

    fn forward_token(&mut self, token: usize) -> Result<Vec<f32>, EngineError> {
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
            Backend::Metal(backend) => {
                self.model
                    .forward_token(token, &mut self.cache, |matrices, input| {
                        backend
                            .lock()
                            .map_err(|_| EngineError::Backend("Metal runtime lock failed".into()))?
                            .mul_vec_many(matrices, input)
                    })
            }
        }
    }
}
