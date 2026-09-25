use std::fmt;
use std::fs;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use half::{bf16, f16};
use safetensors::tensor::{Metadata, TensorView};
use safetensors::{Dtype, SafeTensorError};
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
/// Matrix weights keep their source precision. Tensor payloads are read one at a time.
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
    let mut tensors = TensorArchive::new(File::open(weights_path)?)?;
    let mut layers = Vec::with_capacity(engine_config.num_layers);
    for index in 0..engine_config.num_layers {
        let prefix = format!("model.layers.{index}");
        layers.push(LayerWeights {
            attention_norm: read_vector(&mut tensors, &format!("{prefix}.input_layernorm.weight"))?,
            query: read_matrix(&mut tensors, &format!("{prefix}.self_attn.q_proj.weight"))?,
            key: read_matrix(&mut tensors, &format!("{prefix}.self_attn.k_proj.weight"))?,
            value: read_matrix(&mut tensors, &format!("{prefix}.self_attn.v_proj.weight"))?,
            attention_output: read_matrix(
                &mut tensors,
                &format!("{prefix}.self_attn.o_proj.weight"),
            )?,
            feed_forward_norm: read_vector(
                &mut tensors,
                &format!("{prefix}.post_attention_layernorm.weight"),
            )?,
            gate: read_matrix(&mut tensors, &format!("{prefix}.mlp.gate_proj.weight"))?,
            up: read_matrix(&mut tensors, &format!("{prefix}.mlp.up_proj.weight"))?,
            down: read_matrix(&mut tensors, &format!("{prefix}.mlp.down_proj.weight"))?,
        });
    }
    let weights = ModelWeights {
        token_embeddings: read_matrix(&mut tensors, "model.embed_tokens.weight")?,
        layers,
        final_norm: read_vector(&mut tensors, "model.norm.weight")?,
        output: if config.tie_word_embeddings {
            None
        } else {
            Some(read_matrix(&mut tensors, "lm_head.weight")?)
        },
    };
    Model::new(engine_config, weights).map_err(LoadError::from)
}

const MAX_HEADER_SIZE: u64 = 100_000_000;

struct TensorArchive<R> {
    source: R,
    metadata: Metadata,
    data_start: u64,
}

struct OwnedTensor {
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl OwnedTensor {
    fn view(&self) -> Result<TensorView<'_>, LoadError> {
        TensorView::new(self.dtype, self.shape.clone(), &self.data).map_err(LoadError::from)
    }
}

impl<R: Read + Seek> TensorArchive<R> {
    fn new(mut source: R) -> Result<Self, LoadError> {
        let file_len = source.seek(SeekFrom::End(0))?;
        if file_len < 8 {
            return Err(SafeTensorError::HeaderTooSmall.into());
        }
        source.rewind()?;
        let mut header_size_bytes = [0; 8];
        source.read_exact(&mut header_size_bytes)?;
        let header_size = u64::from_le_bytes(header_size_bytes);
        if header_size > MAX_HEADER_SIZE {
            return Err(SafeTensorError::HeaderTooLarge.into());
        }
        let data_start = header_size
            .checked_add(8)
            .ok_or(SafeTensorError::InvalidHeaderLength)?;
        if data_start > file_len {
            return Err(SafeTensorError::InvalidHeaderLength.into());
        }
        let mut header = vec![0; header_size as usize];
        source.read_exact(&mut header)?;
        if header.first() != Some(&b'{') {
            return Err(SafeTensorError::InvalidHeaderLength.into());
        }
        let metadata: Metadata = serde_json::from_slice(&header)
            .map_err(SafeTensorError::InvalidHeaderDeserialization)?;
        let expected_len = data_start
            .checked_add(metadata.data_len() as u64)
            .ok_or(SafeTensorError::ValidationOverflow)?;
        if expected_len != file_len {
            return Err(SafeTensorError::MetadataIncompleteBuffer.into());
        }
        Ok(Self {
            source,
            metadata,
            data_start,
        })
    }

    fn tensor(&mut self, name: &str) -> Result<OwnedTensor, LoadError> {
        let info = self
            .metadata
            .info(name)
            .ok_or_else(|| SafeTensorError::TensorNotFound(name.to_owned()))?;
        let (start, end) = info.data_offsets;
        let offset = self
            .data_start
            .checked_add(start as u64)
            .ok_or(SafeTensorError::ValidationOverflow)?;
        self.source.seek(SeekFrom::Start(offset))?;
        let mut data = vec![0; end - start];
        self.source.read_exact(&mut data)?;
        Ok(OwnedTensor {
            dtype: info.dtype,
            shape: info.shape.clone(),
            data,
        })
    }
}

fn read_matrix<R: Read + Seek>(
    tensors: &mut TensorArchive<R>,
    name: &str,
) -> Result<Matrix, LoadError> {
    let owned = tensors.tensor(name)?;
    let tensor = owned.view()?;
    let shape = tensor.shape();
    if shape.len() != 2 {
        return Err(LoadError::InvalidTensor(format!(
            "{name} must have two dimensions, got {shape:?}"
        )));
    }
    let matrix = match tensor.dtype() {
        Dtype::F32 => Matrix::new(shape[0], shape[1], read_values(&tensor, name)?)?,
        Dtype::F16 => Matrix::from_f16_bits(shape[0], shape[1], read_half_bits(&tensor, name)?)?,
        Dtype::BF16 => Matrix::from_bf16_bits(shape[0], shape[1], read_half_bits(&tensor, name)?)?,
        dtype => {
            return Err(LoadError::Unsupported(format!(
                "{name} has numeric type {dtype:?}"
            )))
        }
    };
    Ok(matrix)
}

fn read_half_bits(
    tensor: &safetensors::tensor::TensorView<'_>,
    name: &str,
) -> Result<Vec<u16>, LoadError> {
    let (chunks, remainder) = tensor.data().as_chunks::<2>();
    if !remainder.is_empty() {
        return Err(LoadError::InvalidTensor(format!(
            "{name} has incomplete 16-bit data"
        )));
    }
    Ok(chunks
        .iter()
        .map(|chunk| u16::from_le_bytes(*chunk))
        .collect())
}

fn read_vector<R: Read + Seek>(
    tensors: &mut TensorArchive<R>,
    name: &str,
) -> Result<Vec<f32>, LoadError> {
    let owned = tensors.tensor(name)?;
    let tensor = owned.view()?;
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

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::{LoadError, TensorArchive};
    use safetensors::SafeTensorError;

    fn file(header: &str, data: &[u8]) -> Vec<u8> {
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(data);
        bytes
    }

    #[test]
    fn reads_a_tensor_from_a_valid_archive() {
        let bytes = file(
            r#"{"weight":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}"#,
            &[0, 0, 0, 0],
        );
        let mut archive = TensorArchive::new(Cursor::new(bytes)).unwrap();
        let tensor = archive.tensor("weight").unwrap();
        assert_eq!(tensor.shape, [2]);
        assert_eq!(tensor.data, [0, 0, 0, 0]);
    }

    #[test]
    fn rejects_overlapping_or_incomplete_tensor_data() {
        let overlap = file(
            r#"{"a":{"dtype":"F16","shape":[2],"data_offsets":[0,4]},"b":{"dtype":"F16","shape":[2],"data_offsets":[2,6]}}"#,
            &[0; 6],
        );
        assert!(matches!(
            TensorArchive::new(Cursor::new(overlap)),
            Err(LoadError::Safetensors(
                SafeTensorError::InvalidHeaderDeserialization(_)
            ))
        ));

        let incomplete = file(
            r#"{"weight":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}"#,
            &[0; 3],
        );
        assert!(matches!(
            TensorArchive::new(Cursor::new(incomplete)),
            Err(LoadError::Safetensors(
                SafeTensorError::MetadataIncompleteBuffer
            ))
        ));
    }

    #[test]
    fn rejects_oversized_header_before_allocating_it() {
        let bytes = (100_000_001_u64).to_le_bytes();
        assert!(matches!(
            TensorArchive::new(Cursor::new(bytes)),
            Err(LoadError::Safetensors(SafeTensorError::HeaderTooLarge))
        ));
    }
}
