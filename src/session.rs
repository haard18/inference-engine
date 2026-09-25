use crate::model::KvCache;
use crate::{EngineError, Model};

/// State for one token-generation request.
pub struct GenerationSession<'a> {
    model: &'a Model,
    cache: KvCache,
    next_logits: Option<Vec<f32>>,
}

impl<'a> GenerationSession<'a> {
    pub fn new(model: &'a Model) -> Self {
        Self {
            model,
            cache: KvCache::new(model.config().num_layers),
            next_logits: None,
        }
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
            self.next_logits = Some(self.model.forward_token(token, &mut self.cache)?);
        }
        Ok(())
    }

    pub fn next_token(&mut self) -> Result<usize, EngineError> {
        let logits = self.next_logits.as_ref().ok_or(EngineError::EmptyPrompt)?;
        if self.cache.position() >= self.model.config().max_positions {
            return Err(EngineError::ContextFull);
        }
        let token = logits
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
            .ok_or(EngineError::InvalidConfig("model has an empty vocabulary"))?;
        self.next_logits = Some(self.model.forward_token(token, &mut self.cache)?);
        Ok(token)
    }

    pub fn next_token_scores(&self) -> Option<&[f32]> {
        self.next_logits.as_deref()
    }
}
