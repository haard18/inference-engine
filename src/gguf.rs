use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::Path;

use half::{bf16, f16};
use sha2::{Digest, Sha256};

use crate::model::StageWeights;
use crate::{
    ByteBpeTokenizer, EngineError, LayerWeights, Matrix, Model, ModelConfig, ModelStage,
    ModelWeights, TokenizerError,
};

const MAX_STRING_BYTES: u64 = 16_000_000;
const MAX_ENTRIES: u64 = 1_000_000;

#[derive(Debug)]
pub enum GgufError {
    Io(std::io::Error),
    Engine(EngineError),
    Tokenizer(TokenizerError),
    Invalid(String),
    Unsupported(String),
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "could not read GGUF file: {error}"),
            Self::Engine(error) => write!(f, "invalid GGUF model: {error}"),
            Self::Tokenizer(error) => write!(f, "invalid GGUF tokenizer: {error}"),
            Self::Invalid(message) => write!(f, "invalid GGUF file: {message}"),
            Self::Unsupported(message) => write!(f, "unsupported GGUF option: {message}"),
        }
    }
}

impl std::error::Error for GgufError {}

impl From<std::io::Error> for GgufError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<EngineError> for GgufError {
    fn from(error: EngineError) -> Self {
        Self::Engine(error)
    }
}

impl From<TokenizerError> for GgufError {
    fn from(error: TokenizerError) -> Self {
        Self::Tokenizer(error)
    }
}

enum Value {
    U32(u32),
    F32(f32),
    Bool(bool),
    String(String),
    Strings(Vec<String>),
    I32s(Vec<i32>),
}

struct TensorInfo {
    dimensions: Vec<usize>,
    kind: u32,
    offset: u64,
    size: u64,
}

struct GgufReader {
    file: File,
    data_start: u64,
    values: HashMap<String, Value>,
    metadata_keys: HashSet<String>,
    tensors: HashMap<String, TensorInfo>,
}

/// Load the supported Llama-style GGUF layout, retaining Q8_0 matrices in block form.
pub fn load_gguf(path: impl AsRef<Path>) -> Result<Model, GgufError> {
    let mut source = GgufReader::open(path, false)?;
    let config = read_model_config(&source)?;
    let mut layers = Vec::with_capacity(config.num_layers);
    for index in 0..config.num_layers {
        layers.push(read_layer(&mut source, index)?);
    }
    let output = if source.tensors.contains_key("output.weight") {
        Some(source.matrix("output.weight")?)
    } else {
        None
    };
    let weights = ModelWeights {
        token_embeddings: source.matrix("token_embd.weight")?,
        layers,
        final_norm: source.vector("output_norm.weight")?,
        output,
    };
    Model::new(config, weights).map_err(GgufError::from)
}

/// Load only one proper prefix or suffix range of a supported GGUF model.
pub fn load_gguf_stage(
    path: impl AsRef<Path>,
    range: Range<usize>,
) -> Result<ModelStage, GgufError> {
    let path = path.as_ref();
    let mut source = GgufReader::open(path, false)?;
    let config = read_model_config(&source)?;
    if range.start >= range.end
        || range.end > config.num_layers
        || (range.start > 0 && range.end < config.num_layers)
        || (range.start == 0 && range.end == config.num_layers)
    {
        return Err(GgufError::Invalid(
            "stage must be a proper prefix or suffix layer range".into(),
        ));
    }
    let mut layers = Vec::with_capacity(range.end - range.start);
    for index in range.clone() {
        layers.push(read_layer(&mut source, index)?);
    }
    let prefix = range.start == 0;
    let suffix = range.end == config.num_layers;
    let token_embeddings = if prefix {
        Some(source.matrix("token_embd.weight")?)
    } else {
        None
    };
    let final_norm = if suffix {
        Some(source.vector("output_norm.weight")?)
    } else {
        None
    };
    let output = if suffix {
        Some(if source.tensors.contains_key("output.weight") {
            source.matrix("output.weight")?
        } else {
            source.matrix("token_embd.weight")?
        })
    } else {
        None
    };
    let weights = StageWeights {
        token_embeddings,
        layers,
        final_norm,
        output,
    };
    let digest = file_digest(&mut source.file)?;
    ModelStage::new(config, range, weights, digest).map_err(GgufError::from)
}

fn file_digest(file: &mut File) -> Result<[u8; 32], GgufError> {
    file.rewind()?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(digest.finalize().into())
}

pub(crate) fn digest_file(path: impl AsRef<Path>) -> Result<[u8; 32], GgufError> {
    file_digest(&mut File::open(path)?)
}

fn read_model_config(source: &GgufReader) -> Result<ModelConfig, GgufError> {
    if source.string("general.architecture")? != "llama" {
        return Err(GgufError::Unsupported("model architecture".into()));
    }
    if source
        .metadata_keys
        .iter()
        .any(|key| key.starts_with("llama.rope.scaling."))
    {
        return Err(GgufError::Unsupported("rotary position scaling".into()));
    }
    let hidden_size = source.integer("llama.embedding_length")? as usize;
    let num_attention_heads = source.integer("llama.attention.head_count")? as usize;
    let num_key_value_heads = source.integer("llama.attention.head_count_kv")? as usize;
    let head_size = hidden_size
        .checked_div(num_attention_heads)
        .ok_or_else(|| GgufError::Invalid("zero attention heads".into()))?;
    if source
        .optional_integer("llama.attention.key_length")?
        .is_some_and(|v| v as usize != head_size)
        || source
            .optional_integer("llama.attention.value_length")?
            .is_some_and(|v| v as usize != head_size)
        || source
            .optional_integer("llama.rope.dimension_count")?
            .is_some_and(|v| v as usize != head_size)
    {
        return Err(GgufError::Unsupported(
            "attention or rotary head width".into(),
        ));
    }
    let config = ModelConfig {
        vocab_size: source.integer("llama.vocab_size")? as usize,
        hidden_size,
        intermediate_size: source.integer("llama.feed_forward_length")? as usize,
        num_layers: source.integer("llama.block_count")? as usize,
        num_attention_heads,
        num_key_value_heads,
        max_positions: source.integer("llama.context_length")? as usize,
        rms_norm_epsilon: source.float("llama.attention.layer_norm_rms_epsilon")?,
        rope_theta: source.float("llama.rope.freq_base")?,
        rope_interleaved: true,
    };
    Ok(config)
}

fn read_layer(source: &mut GgufReader, index: usize) -> Result<LayerWeights, GgufError> {
    let prefix = format!("blk.{index}");
    Ok(LayerWeights {
        attention_norm: source.vector(&format!("{prefix}.attn_norm.weight"))?,
        query: source.matrix(&format!("{prefix}.attn_q.weight"))?,
        key: source.matrix(&format!("{prefix}.attn_k.weight"))?,
        value: source.matrix(&format!("{prefix}.attn_v.weight"))?,
        attention_output: source.matrix(&format!("{prefix}.attn_output.weight"))?,
        feed_forward_norm: source.vector(&format!("{prefix}.ffn_norm.weight"))?,
        gate: source.matrix(&format!("{prefix}.ffn_gate.weight"))?,
        up: source.matrix(&format!("{prefix}.ffn_up.weight"))?,
        down: source.matrix(&format!("{prefix}.ffn_down.weight"))?,
    })
}

/// Load the supported SmolLM byte-level BPE tokenizer from GGUF metadata.
pub fn load_gguf_tokenizer(path: impl AsRef<Path>) -> Result<ByteBpeTokenizer, GgufError> {
    let mut source = GgufReader::open(path, true)?;
    if source.metadata_keys.iter().any(|key| {
        key.starts_with("tokenizer.ggml.")
            && !matches!(
                key.as_str(),
                "tokenizer.ggml.model"
                    | "tokenizer.ggml.pre"
                    | "tokenizer.ggml.tokens"
                    | "tokenizer.ggml.token_type"
                    | "tokenizer.ggml.merges"
                    | "tokenizer.ggml.add_space_prefix"
                    | "tokenizer.ggml.bos_token_id"
                    | "tokenizer.ggml.eos_token_id"
                    | "tokenizer.ggml.unknown_token_id"
                    | "tokenizer.ggml.add_bos_token"
            )
    }) {
        return Err(GgufError::Unsupported("tokenizer metadata".into()));
    }
    if source.string("tokenizer.ggml.model")? != "gpt2"
        || source.string("tokenizer.ggml.pre")? != "smollm"
        || source.boolean("tokenizer.ggml.add_space_prefix")?
        || source.boolean("tokenizer.ggml.add_bos_token")?
    {
        return Err(GgufError::Unsupported("tokenizer layout".into()));
    }
    let tokens = source.take_strings("tokenizer.ggml.tokens")?;
    let types = source.take_i32s("tokenizer.ggml.token_type")?;
    for key in [
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.unknown_token_id",
    ] {
        let index = source.integer(key)? as usize;
        if types.get(index) != Some(&3) {
            return Err(GgufError::Invalid(format!(
                "invalid special token ID {key}"
            )));
        }
    }
    ByteBpeTokenizer::from_gguf_parts(tokens, source.take_strings("tokenizer.ggml.merges")?, types)
        .map_err(GgufError::from)
}

impl GgufReader {
    fn open(path: impl AsRef<Path>, load_tokenizer: bool) -> Result<Self, GgufError> {
        let mut file = File::open(path)?;
        let file_len = file.seek(SeekFrom::End(0))?;
        file.rewind()?;
        if read_bytes::<4>(&mut file)? != *b"GGUF" {
            return Err(GgufError::Invalid("missing GGUF signature".into()));
        }
        let version = read_u32(&mut file)?;
        if version != 3 {
            return Err(GgufError::Unsupported(format!("GGUF version {version}")));
        }
        let tensor_count = read_u64(&mut file)?;
        let value_count = read_u64(&mut file)?;
        if tensor_count > MAX_ENTRIES || value_count > MAX_ENTRIES {
            return Err(GgufError::Invalid("too many header entries".into()));
        }
        let mut values = HashMap::new();
        let mut seen_keys = HashSet::new();
        let mut alignment = 32_u64;
        for _ in 0..value_count {
            let key = read_string(&mut file)?;
            if !seen_keys.insert(key.clone()) {
                return Err(GgufError::Invalid("duplicate metadata key".into()));
            }
            let kind = read_u32(&mut file)?;
            let keep = key == "general.architecture"
                || key == "general.alignment"
                || key.starts_with("llama.")
                || (load_tokenizer && key.starts_with("tokenizer.ggml."));
            let value = read_value(&mut file, file_len, kind, keep)?;
            if key == "general.alignment" {
                alignment = match value {
                    Some(Value::U32(value)) if value.is_power_of_two() => u64::from(value),
                    _ => return Err(GgufError::Invalid("invalid tensor alignment".into())),
                };
            } else if let Some(value) = value {
                values.insert(key, value);
            }
        }
        let mut tensors = HashMap::new();
        for _ in 0..tensor_count {
            let name = read_string(&mut file)?;
            let dimension_count = read_u32(&mut file)?;
            if !(1..=4).contains(&dimension_count) {
                return Err(GgufError::Unsupported("tensor dimension count".into()));
            }
            let mut dimensions = Vec::with_capacity(dimension_count as usize);
            for _ in 0..dimension_count {
                dimensions.push(
                    usize::try_from(read_u64(&mut file)?)
                        .map_err(|_| GgufError::Invalid("tensor dimension overflow".into()))?,
                );
            }
            let kind = read_u32(&mut file)?;
            let offset = read_u64(&mut file)?;
            let size = tensor_size(kind, &dimensions)?;
            if tensors
                .insert(
                    name,
                    TensorInfo {
                        dimensions,
                        kind,
                        offset,
                        size,
                    },
                )
                .is_some()
            {
                return Err(GgufError::Invalid("duplicate tensor name".into()));
            }
        }
        let descriptors_end = file.stream_position()?;
        let data_start = align_up(descriptors_end, alignment)?;
        if data_start > file_len {
            return Err(GgufError::Invalid(
                "tensor data begins after file end".into(),
            ));
        }
        let mut ranges: Vec<_> = tensors
            .values()
            .map(|tensor| (tensor.offset, tensor.size))
            .collect();
        ranges.sort_unstable_by_key(|range| range.0);
        let mut last_end = 0_u64;
        for (offset, size) in ranges {
            let end = offset
                .checked_add(size)
                .ok_or_else(|| GgufError::Invalid("tensor offset overflow".into()))?;
            if offset < last_end
                || data_start
                    .checked_add(end)
                    .is_none_or(|absolute| absolute > file_len)
            {
                return Err(GgufError::Invalid(
                    "overlapping or out-of-bounds tensor".into(),
                ));
            }
            last_end = end;
        }
        Ok(Self {
            file,
            data_start,
            values,
            metadata_keys: seen_keys,
            tensors,
        })
    }

    fn integer(&self, key: &str) -> Result<u32, GgufError> {
        match self.values.get(key) {
            Some(Value::U32(value)) => Ok(*value),
            _ => Err(GgufError::Invalid(format!(
                "missing unsigned integer {key}"
            ))),
        }
    }

    fn optional_integer(&self, key: &str) -> Result<Option<u32>, GgufError> {
        match self.values.get(key) {
            None => Ok(None),
            Some(Value::U32(value)) => Ok(Some(*value)),
            _ => Err(GgufError::Invalid(format!(
                "invalid unsigned integer {key}"
            ))),
        }
    }

    fn float(&self, key: &str) -> Result<f32, GgufError> {
        match self.values.get(key) {
            Some(Value::F32(value)) => Ok(*value),
            _ => Err(GgufError::Invalid(format!("missing float {key}"))),
        }
    }

    fn string(&self, key: &str) -> Result<&str, GgufError> {
        match self.values.get(key) {
            Some(Value::String(value)) => Ok(value),
            _ => Err(GgufError::Invalid(format!("missing string {key}"))),
        }
    }

    fn boolean(&self, key: &str) -> Result<bool, GgufError> {
        match self.values.get(key) {
            Some(Value::Bool(value)) => Ok(*value),
            _ => Err(GgufError::Invalid(format!("missing boolean {key}"))),
        }
    }

    fn take_strings(&mut self, key: &str) -> Result<Vec<String>, GgufError> {
        match self.values.remove(key) {
            Some(Value::Strings(values)) => Ok(values),
            _ => Err(GgufError::Invalid(format!("missing string array {key}"))),
        }
    }

    fn take_i32s(&mut self, key: &str) -> Result<Vec<i32>, GgufError> {
        match self.values.remove(key) {
            Some(Value::I32s(values)) => Ok(values),
            _ => Err(GgufError::Invalid(format!("missing integer array {key}"))),
        }
    }

    fn tensor(&mut self, name: &str) -> Result<(u32, Vec<usize>, Vec<u8>), GgufError> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| GgufError::Invalid(format!("missing tensor {name}")))?;
        let start = self.data_start + info.offset;
        self.file.seek(SeekFrom::Start(start))?;
        let size = usize::try_from(info.size)
            .map_err(|_| GgufError::Invalid("tensor size overflow".into()))?;
        let mut data = vec![0_u8; size];
        self.file.read_exact(&mut data)?;
        Ok((info.kind, info.dimensions.clone(), data))
    }

    fn matrix(&mut self, name: &str) -> Result<Matrix, GgufError> {
        let (kind, dims, data) = self.tensor(name)?;
        if dims.len() != 2 {
            return Err(GgufError::Invalid(format!("{name} must be a matrix")));
        }
        let (cols, rows) = (dims[0], dims[1]);
        match kind {
            0 => Matrix::new(rows, cols, decode_f32(&data)?).map_err(GgufError::from),
            1 => Matrix::from_f16_bits(rows, cols, decode_u16(&data)?).map_err(GgufError::from),
            6 => Matrix::from_q5_0(rows, cols, data).map_err(GgufError::from),
            8 => Matrix::from_q8_0(rows, cols, data).map_err(GgufError::from),
            12 => Matrix::from_q4_k(rows, cols, data).map_err(GgufError::from),
            30 => Matrix::from_bf16_bits(rows, cols, decode_u16(&data)?).map_err(GgufError::from),
            _ => Err(GgufError::Unsupported(format!("tensor type {kind}"))),
        }
    }

    fn vector(&mut self, name: &str) -> Result<Vec<f32>, GgufError> {
        let (kind, dims, data) = self.tensor(name)?;
        if dims.len() != 1 {
            return Err(GgufError::Invalid(format!("{name} must be a vector")));
        }
        let values = match kind {
            0 => decode_f32(&data)?,
            1 => decode_u16(&data)?
                .into_iter()
                .map(|bits| f16::from_bits(bits).to_f32())
                .collect(),
            30 => decode_u16(&data)?
                .into_iter()
                .map(|bits| bf16::from_bits(bits).to_f32())
                .collect(),
            _ => return Err(GgufError::Unsupported(format!("vector type {kind}"))),
        };
        Ok(values)
    }
}

fn tensor_size(kind: u32, dims: &[usize]) -> Result<u64, GgufError> {
    let elements = dims
        .iter()
        .copied()
        .try_fold(1_usize, usize::checked_mul)
        .ok_or_else(|| GgufError::Invalid("tensor dimensions overflow".into()))?;
    let bytes = match kind {
        0 => elements.checked_mul(4),
        1 | 30 => elements.checked_mul(2),
        6 if dims[0].is_multiple_of(32) => (elements / 32).checked_mul(22),
        8 if dims[0].is_multiple_of(32) => (elements / 32).checked_mul(34),
        12 if dims[0].is_multiple_of(256) => (elements / 256).checked_mul(144),
        6 | 8 | 12 => {
            return Err(GgufError::Invalid(
                "quantized tensor width must divide by block width".into(),
            ))
        }
        _ => return Err(GgufError::Unsupported(format!("tensor type {kind}"))),
    }
    .ok_or_else(|| GgufError::Invalid("tensor size overflow".into()))?;
    u64::try_from(bytes).map_err(|_| GgufError::Invalid("tensor size overflow".into()))
}

fn align_up(value: u64, alignment: u64) -> Result<u64, GgufError> {
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or_else(|| GgufError::Invalid("tensor alignment overflow".into()))
}

fn read_value(
    file: &mut File,
    file_len: u64,
    kind: u32,
    keep: bool,
) -> Result<Option<Value>, GgufError> {
    let value = match kind {
        4 if keep => Some(Value::U32(read_u32(file)?)),
        6 if keep => Some(Value::F32(f32::from_le_bytes(read_bytes::<4>(file)?))),
        7 if keep => Some(Value::Bool(match read_bytes::<1>(file)?[0] {
            0 => false,
            1 => true,
            _ => return Err(GgufError::Invalid("invalid boolean metadata".into())),
        })),
        8 if keep => Some(Value::String(read_string(file)?)),
        8 => {
            skip_string(file, file_len)?;
            None
        }
        9 => {
            let element_type = read_u32(file)?;
            let count = read_u64(file)?;
            if count > MAX_ENTRIES {
                return Err(GgufError::Invalid("metadata array too large".into()));
            }
            if keep && element_type == 8 {
                Some(Value::Strings(
                    (0..count)
                        .map(|_| read_string(file))
                        .collect::<Result<Vec<_>, _>>()?,
                ))
            } else if keep && element_type == 5 {
                Some(Value::I32s(
                    (0..count)
                        .map(|_| Ok(i32::from_le_bytes(read_bytes::<4>(file)?)))
                        .collect::<Result<Vec<_>, GgufError>>()?,
                ))
            } else if element_type == 8 {
                for _ in 0..count {
                    skip_string(file, file_len)?;
                }
                None
            } else {
                let width = primitive_width(element_type)?;
                skip_bytes(
                    file,
                    file_len,
                    count
                        .checked_mul(width)
                        .ok_or_else(|| GgufError::Invalid("array length overflow".into()))?,
                )?;
                None
            }
        }
        _ => {
            skip_bytes(file, file_len, primitive_width(kind)?)?;
            None
        }
    };
    Ok(value)
}

fn primitive_width(kind: u32) -> Result<u64, GgufError> {
    match kind {
        0 | 1 | 7 => Ok(1),
        2 | 3 => Ok(2),
        4..=6 => Ok(4),
        10..=12 => Ok(8),
        _ => Err(GgufError::Unsupported(format!(
            "metadata value type {kind}"
        ))),
    }
}

fn skip_string(file: &mut File, file_len: u64) -> Result<(), GgufError> {
    let length = read_u64(file)?;
    if length > MAX_STRING_BYTES {
        return Err(GgufError::Invalid("string too long".into()));
    }
    skip_bytes(file, file_len, length)
}

fn skip_bytes(file: &mut File, file_len: u64, length: u64) -> Result<(), GgufError> {
    let end = file
        .stream_position()?
        .checked_add(length)
        .ok_or_else(|| GgufError::Invalid("file offset overflow".into()))?;
    if end > file_len {
        return Err(GgufError::Invalid("metadata extends past file end".into()));
    }
    file.seek(SeekFrom::Start(end))?;
    Ok(())
}

fn read_string(file: &mut File) -> Result<String, GgufError> {
    let length = read_u64(file)?;
    if length > MAX_STRING_BYTES {
        return Err(GgufError::Invalid("string too long".into()));
    }
    let mut bytes = vec![0; length as usize];
    file.read_exact(&mut bytes)?;
    String::from_utf8(bytes).map_err(|_| GgufError::Invalid("invalid UTF-8 string".into()))
}

fn read_bytes<const N: usize>(file: &mut File) -> Result<[u8; N], GgufError> {
    let mut bytes = [0; N];
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn read_u32(file: &mut File) -> Result<u32, GgufError> {
    Ok(u32::from_le_bytes(read_bytes(file)?))
}

fn read_u64(file: &mut File) -> Result<u64, GgufError> {
    Ok(u64::from_le_bytes(read_bytes(file)?))
}

fn decode_u16(bytes: &[u8]) -> Result<Vec<u16>, GgufError> {
    let (chunks, remainder) = bytes.as_chunks::<2>();
    if !remainder.is_empty() {
        return Err(GgufError::Invalid("incomplete 16-bit tensor".into()));
    }
    Ok(chunks
        .iter()
        .map(|chunk| u16::from_le_bytes(*chunk))
        .collect())
}

fn decode_f32(bytes: &[u8]) -> Result<Vec<f32>, GgufError> {
    let (chunks, remainder) = bytes.as_chunks::<4>();
    if !remainder.is_empty() {
        return Err(GgufError::Invalid("incomplete f32 tensor".into()));
    }
    Ok(chunks
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::{GgufError, GgufReader};

    static NEXT_FILE: AtomicUsize = AtomicUsize::new(0);

    struct Fixture(PathBuf);

    impl Fixture {
        fn new(bytes: &[u8]) -> Self {
            let path = std::env::temp_dir().join(format!(
                "inference-engine-gguf-{}-{}",
                std::process::id(),
                NEXT_FILE.fetch_add(1, Ordering::Relaxed)
            ));
            fs::write(&path, bytes).unwrap();
            Self(path)
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.0);
        }
    }

    fn write_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    fn archive(offsets: &[u64]) -> Vec<u8> {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&(offsets.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        write_string(&mut bytes, "general.alignment");
        bytes.extend_from_slice(&4_u32.to_le_bytes());
        bytes.extend_from_slice(&32_u32.to_le_bytes());
        for (index, offset) in offsets.iter().enumerate() {
            write_string(&mut bytes, &format!("tensor.{index}"));
            bytes.extend_from_slice(&2_u32.to_le_bytes());
            bytes.extend_from_slice(&32_u64.to_le_bytes());
            bytes.extend_from_slice(&1_u64.to_le_bytes());
            bytes.extend_from_slice(&8_u32.to_le_bytes());
            bytes.extend_from_slice(&offset.to_le_bytes());
        }
        bytes.resize(bytes.len().next_multiple_of(32), 0);
        bytes.extend_from_slice(&[0; 34]);
        bytes
    }

    #[test]
    fn reads_a_q8_0_tensor() {
        let fixture = Fixture::new(&archive(&[0]));
        let mut reader = GgufReader::open(&fixture.0, false).unwrap();
        let (kind, shape, data) = reader.tensor("tensor.0").unwrap();
        assert_eq!(kind, 8);
        assert_eq!(shape, [32, 1]);
        assert_eq!(data.len(), 34);
    }

    #[test]
    fn rejects_overlapping_or_truncated_tensors() {
        let overlap = Fixture::new(&archive(&[0, 0]));
        assert!(matches!(
            GgufReader::open(&overlap.0, false),
            Err(GgufError::Invalid(_))
        ));
        let mut truncated = archive(&[0]);
        truncated.truncate(truncated.len() - 1);
        let truncated = Fixture::new(&truncated);
        assert!(matches!(
            GgufReader::open(&truncated.0, false),
            Err(GgufError::Invalid(_))
        ));
    }
}
