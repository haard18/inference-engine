//! Bounded, versioned hidden-state transfer between adjacent model stages.

use std::fmt;

use uuid::Uuid;

use crate::{EngineError, ModelStage};

const MAGIC: [u8; 4] = *b"INFA";
const VERSION: u16 = 1;
const HEADER_BYTES: usize = 64;
const MAX_FRAME_BYTES: usize = 4 * 1024 * 1024;

#[derive(Debug, PartialEq)]
pub enum ActivationError {
    Invalid(&'static str),
    Engine(EngineError),
}

impl fmt::Display for ActivationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => write!(f, "invalid activation frame: {message}"),
            Self::Engine(error) => write!(f, "activation execution failed: {error}"),
        }
    }
}

impl std::error::Error for ActivationError {}

impl From<EngineError> for ActivationError {
    fn from(error: EngineError) -> Self {
        Self::Engine(error)
    }
}

/// One f32 hidden vector at a decoder layer boundary.
pub struct ActivationFrame {
    model_digest: [u8; 32],
    request_id: Uuid,
    position: u32,
    hidden: Vec<f32>,
}

impl ActivationFrame {
    pub fn new(
        stage: &ModelStage,
        request_id: Uuid,
        position: usize,
        hidden: Vec<f32>,
    ) -> Result<Self, ActivationError> {
        if !stage.is_prefix() {
            return Err(ActivationError::Invalid("sender is not a prefix stage"));
        }
        let position = u32::try_from(position)
            .map_err(|_| ActivationError::Invalid("token position exceeds protocol limit"))?;
        if hidden.len() != stage.hidden_size() {
            return Err(ActivationError::Invalid(
                "hidden width does not match model",
            ));
        }
        if !hidden.iter().all(|value| value.is_finite()) {
            return Err(ActivationError::Invalid("hidden values must be finite"));
        }
        let payload_bytes = hidden
            .len()
            .checked_mul(4)
            .and_then(|bytes| bytes.checked_add(HEADER_BYTES))
            .ok_or(ActivationError::Invalid("activation size overflows"))?;
        if payload_bytes > MAX_FRAME_BYTES {
            return Err(ActivationError::Invalid("activation exceeds size limit"));
        }
        Ok(Self {
            model_digest: stage.model_digest(),
            request_id,
            position,
            hidden,
        })
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(HEADER_BYTES + self.hidden.len() * 4);
        bytes.extend_from_slice(&MAGIC);
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&0_u16.to_le_bytes());
        bytes.extend_from_slice(&self.model_digest);
        bytes.extend_from_slice(self.request_id.as_bytes());
        bytes.extend_from_slice(&self.position.to_le_bytes());
        bytes.extend_from_slice(&(self.hidden.len() as u32).to_le_bytes());
        for value in &self.hidden {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    pub fn decode_for_stage(
        bytes: &[u8],
        stage: &ModelStage,
        request_id: Uuid,
        position: usize,
    ) -> Result<Self, ActivationError> {
        if !stage.is_suffix() {
            return Err(ActivationError::Invalid("receiver is not a suffix stage"));
        }
        if bytes.len() < HEADER_BYTES || bytes.len() > MAX_FRAME_BYTES {
            return Err(ActivationError::Invalid("frame length is outside limits"));
        }
        if bytes[..4] != MAGIC || u16::from_le_bytes(bytes[4..6].try_into().unwrap()) != VERSION {
            return Err(ActivationError::Invalid("magic or version does not match"));
        }
        if bytes[6..8] != [0, 0] {
            return Err(ActivationError::Invalid("reserved header bits are set"));
        }
        if bytes[8..40] != stage.model_digest() {
            return Err(ActivationError::Invalid("model identity does not match"));
        }
        if bytes[40..56] != *request_id.as_bytes() {
            return Err(ActivationError::Invalid("request identity does not match"));
        }
        let frame_position = u32::from_le_bytes(bytes[56..60].try_into().unwrap());
        if usize::try_from(frame_position).ok() != Some(position) {
            return Err(ActivationError::Invalid("token position does not match"));
        }
        let width = u32::from_le_bytes(bytes[60..64].try_into().unwrap()) as usize;
        let expected_bytes = width
            .checked_mul(4)
            .and_then(|size| size.checked_add(HEADER_BYTES));
        if width != stage.hidden_size() || expected_bytes != Some(bytes.len()) {
            return Err(ActivationError::Invalid(
                "hidden width or payload length does not match",
            ));
        }
        let mut hidden = Vec::with_capacity(width);
        for chunk in bytes[HEADER_BYTES..].as_chunks::<4>().0 {
            let value = f32::from_le_bytes(*chunk);
            if !value.is_finite() {
                return Err(ActivationError::Invalid("hidden values must be finite"));
            }
            hidden.push(value);
        }
        Ok(Self {
            model_digest: stage.model_digest(),
            request_id,
            position: frame_position,
            hidden,
        })
    }

    pub fn into_hidden(self) -> Vec<f32> {
        self.hidden
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::StageWeights;
    use crate::{LayerWeights, Matrix, ModelConfig};

    fn stages() -> (ModelStage, ModelStage) {
        fn matrix(rows: usize, columns: usize) -> Matrix {
            Matrix::new(rows, columns, vec![0.0; rows * columns]).unwrap()
        }
        let config = ModelConfig {
            vocab_size: 3,
            hidden_size: 2,
            intermediate_size: 2,
            num_layers: 2,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            max_positions: 4,
            rms_norm_epsilon: 1e-5,
            rope_theta: 10000.0,
            rope_interleaved: false,
        };
        let layer = LayerWeights {
            attention_norm: vec![1.0; 2],
            query: matrix(2, 2),
            key: matrix(2, 2),
            value: matrix(2, 2),
            attention_output: matrix(2, 2),
            feed_forward_norm: vec![1.0; 2],
            gate: matrix(2, 2),
            up: matrix(2, 2),
            down: matrix(2, 2),
        };
        let digest = [7; 32];
        let prefix = ModelStage::new(
            config.clone(),
            0..1,
            StageWeights {
                token_embeddings: Some(matrix(3, 2)),
                layers: vec![layer.clone()],
                final_norm: None,
                output: None,
            },
            digest,
        )
        .unwrap();
        let suffix = ModelStage::new(
            config,
            1..2,
            StageWeights {
                token_embeddings: None,
                layers: vec![layer],
                final_norm: Some(vec![1.0; 2]),
                output: Some(matrix(3, 2)),
            },
            digest,
        )
        .unwrap();
        (prefix, suffix)
    }

    #[test]
    fn frame_rejects_corruption_and_mismatched_stage_context() {
        let (prefix, suffix) = stages();
        let request = Uuid::new_v4();
        let frame = ActivationFrame::new(&prefix, request, 2, vec![0.25, -0.5]).unwrap();
        let bytes = frame.encode();
        assert_eq!(bytes.len(), 72);
        let decoded = ActivationFrame::decode_for_stage(&bytes, &suffix, request, 2).unwrap();
        assert_eq!(decoded.into_hidden(), vec![0.25, -0.5]);
        assert!(ActivationFrame::new(&suffix, request, 2, vec![0.0, 0.0]).is_err());
        assert!(ActivationFrame::new(&prefix, request, 2, vec![0.0]).is_err());
        assert!(ActivationFrame::new(&prefix, request, 2, vec![f32::NAN, 0.0]).is_err());
        assert!(ActivationFrame::decode_for_stage(&bytes, &prefix, request, 2).is_err());
        assert!(ActivationFrame::decode_for_stage(&bytes, &suffix, Uuid::new_v4(), 2).is_err());
        assert!(ActivationFrame::decode_for_stage(&bytes, &suffix, request, 3).is_err());
        assert!(ActivationFrame::decode_for_stage(&bytes[..71], &suffix, request, 2).is_err());

        for (index, value) in [(0, 0), (4, 2), (6, 1), (8, 9), (60, 3)] {
            let mut corrupted = bytes.clone();
            corrupted[index] = value;
            assert!(
                ActivationFrame::decode_for_stage(&corrupted, &suffix, request, 2).is_err(),
                "corruption at byte {index} was accepted"
            );
        }
        let mut non_finite = bytes;
        non_finite[64..68].copy_from_slice(&f32::INFINITY.to_le_bytes());
        assert!(ActivationFrame::decode_for_stage(&non_finite, &suffix, request, 2).is_err());
    }
}
