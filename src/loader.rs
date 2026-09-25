use std::fmt;
use std::fs;
use std::path::Path;

use half::{bf16, f16};
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;

use crate::{EngineError, LayerWeights, Matrix, Model, ModelConfig, ModelWeights};

#[derive(Debug)]
pub enum LoadError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Safetensors(safetensors::SafeTensorError),
    Engine(EngineError),
    Unsupported(String),
    InvalidTensor(String),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "could not read model file: {error}"),
            Self::Json(error) => write!(f, "invalid model configuration JSON: {error}"),
            Self::Safetensors(error) => write!(f, "invalid Safetensors file: {error}"),
            Self::Engine(error) => write!(f, "invalid model: {error}"),
            Self::Unsupported(message) => write!(f, "unsupported model option: {message}"),
            Self::InvalidTensor(message) => write!(f, "invalid model tensor: {message}"),
        }
    }
}

impl std::error::Error for LoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Json(error) => Some(error),
            Self::Safetensors(error) => Some(error),
            Self::Engine(error) => Some(error),
            Self::Unsupported(_) | Self::InvalidTensor(_) => None,
        }
    }
}

impl From<std::io::Error> for LoadError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for LoadError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

impl From<safetensors::SafeTensorError> for LoadError {
    fn from(error: safetensors::SafeTensorError) -> Self {
        Self::Safetensors(error)
    }
}

impl From<EngineError> for LoadError {
    fn from(error: EngineError) -> Self {
        Self::Engine(error)
    }
}

#[derive(Deserialize)]
struct LlamaConfig {
    model_type: String,
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    max_position_embeddings: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    tie_word_embeddings: bool,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    rope_interleaved: bool,
    #[serde(default)]
    rope_scaling: Option<serde_json::Value>,
    hidden_act: String,
}

/// Load a Llama-style model from a configuration file and Safetensors weights.
/// The first implementation converts weights to f32 and holds them in memory.
pub fn load_safetensors(
    config_path: impl AsRef<Path>,
    weights_path: impl AsRef<Path>,
) -> Result<Model, LoadError> {
    let config: LlamaConfig = serde_json::from_slice(&fs::read(config_path)?)?;
    if config.model_type != "llama" {
        return Err(LoadError::Unsupported(format!(
            "model type {}",
            config.model_type
        )));
    }
    if config.hidden_act != "silu" {
        return Err(LoadError::Unsupported(format!(
            "activation {}",
            config.hidden_act
        )));
    }
    if config.attention_bias {
        return Err(LoadError::Unsupported("attention bias".into()));
    }
    if config.rope_interleaved {
        return Err(LoadError::Unsupported(
            "interleaved rotary positions".into(),
        ));
    }
    if config.rope_scaling.is_some() {
        return Err(LoadError::Unsupported("rotary position scaling".into()));
    }
    let engine_config = ModelConfig {
        vocab_size: config.vocab_size,
        hidden_size: config.hidden_size,
        intermediate_size: config.intermediate_size,
        num_layers: config.num_hidden_layers,
        num_attention_heads: config.num_attention_heads,
        num_key_value_heads: config.num_key_value_heads,
        max_positions: config.max_position_embeddings,
        rms_norm_epsilon: config.rms_norm_eps,
        rope_theta: config.rope_theta,
    };
    let bytes = fs::read(weights_path)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let mut layers = Vec::with_capacity(engine_config.num_layers);
    for index in 0..engine_config.num_layers {
        let prefix = format!("model.layers.{index}");
        layers.push(LayerWeights {
            attention_norm: read_vector(&tensors, &format!("{prefix}.input_layernorm.weight"))?,
            query: read_matrix(&tensors, &format!("{prefix}.self_attn.q_proj.weight"))?,
            key: read_matrix(&tensors, &format!("{prefix}.self_attn.k_proj.weight"))?,
            value: read_matrix(&tensors, &format!("{prefix}.self_attn.v_proj.weight"))?,
            attention_output: read_matrix(&tensors, &format!("{prefix}.self_attn.o_proj.weight"))?,
            feed_forward_norm: read_vector(
                &tensors,
                &format!("{prefix}.post_attention_layernorm.weight"),
            )?,
            gate: read_matrix(&tensors, &format!("{prefix}.mlp.gate_proj.weight"))?,
            up: read_matrix(&tensors, &format!("{prefix}.mlp.up_proj.weight"))?,
            down: read_matrix(&tensors, &format!("{prefix}.mlp.down_proj.weight"))?,
        });
    }
    let weights = ModelWeights {
        token_embeddings: read_matrix(&tensors, "model.embed_tokens.weight")?,
        layers,
        final_norm: read_vector(&tensors, "model.norm.weight")?,
        output: if config.tie_word_embeddings {
            None
        } else {
            Some(read_matrix(&tensors, "lm_head.weight")?)
        },
    };
    Model::new(engine_config, weights).map_err(LoadError::from)
}

fn read_matrix(tensors: &SafeTensors<'_>, name: &str) -> Result<Matrix, LoadError> {
    let tensor = tensors.tensor(name)?;
    let shape = tensor.shape();
    if shape.len() != 2 {
        return Err(LoadError::InvalidTensor(format!(
            "{name} must have two dimensions, got {shape:?}"
        )));
    }
    Matrix::new(shape[0], shape[1], read_values(&tensor, name)?).map_err(LoadError::from)
}

fn read_vector(tensors: &SafeTensors<'_>, name: &str) -> Result<Vec<f32>, LoadError> {
    let tensor = tensors.tensor(name)?;
    if tensor.shape().len() != 1 {
        return Err(LoadError::InvalidTensor(format!(
            "{name} must have one dimension, got {:?}",
            tensor.shape()
        )));
    }
    read_values(&tensor, name)
}

fn read_values(
    tensor: &safetensors::tensor::TensorView<'_>,
    name: &str,
) -> Result<Vec<f32>, LoadError> {
    let bytes = tensor.data();
    let values = match tensor.dtype() {
        Dtype::F32 => {
            let (chunks, remainder) = bytes.as_chunks::<4>();
            if !remainder.is_empty() {
                return Err(LoadError::InvalidTensor(format!(
                    "{name} has incomplete f32 data"
                )));
            }
            chunks
                .iter()
                .map(|chunk| f32::from_le_bytes(*chunk))
                .collect()
        }
        Dtype::F16 => {
            let (chunks, remainder) = bytes.as_chunks::<2>();
            if !remainder.is_empty() {
                return Err(LoadError::InvalidTensor(format!(
                    "{name} has incomplete f16 data"
                )));
            }
            chunks
                .iter()
                .map(|chunk| f16::from_bits(u16::from_le_bytes(*chunk)).to_f32())
                .collect()
        }
        Dtype::BF16 => {
            let (chunks, remainder) = bytes.as_chunks::<2>();
            if !remainder.is_empty() {
                return Err(LoadError::InvalidTensor(format!(
                    "{name} has incomplete bf16 data"
                )));
            }
            chunks
                .iter()
                .map(|chunk| bf16::from_bits(u16::from_le_bytes(*chunk)).to_f32())
                .collect()
        }
        dtype => {
            return Err(LoadError::Unsupported(format!(
                "{name} has numeric type {dtype:?}"
            )))
        }
    };
    Ok(values)
}
