use crate::tensor::{apply_rope, rms_norm, silu, softmax};
use crate::{EngineError, Matrix};
use std::mem::size_of;

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_positions: usize,
    pub rms_norm_epsilon: f32,
    pub rope_theta: f32,
}

#[derive(Clone, Debug)]
pub struct LayerWeights {
    pub attention_norm: Vec<f32>,
    pub query: Matrix,
    pub key: Matrix,
    pub value: Matrix,
    pub attention_output: Matrix,
    pub feed_forward_norm: Vec<f32>,
    pub gate: Matrix,
    pub up: Matrix,
    pub down: Matrix,
}

#[derive(Clone, Debug)]
pub struct ModelWeights {
    pub token_embeddings: Matrix,
    pub layers: Vec<LayerWeights>,
    pub final_norm: Vec<f32>,
    /// None means that the output projection reuses token_embeddings.
    pub output: Option<Matrix>,
}

#[derive(Clone, Debug)]
pub struct Model {
    config: ModelConfig,
    weights: ModelWeights,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct LayerCache {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

#[derive(Clone, Debug)]
pub(crate) struct KvCache {
    layers: Vec<LayerCache>,
    position: usize,
}

impl KvCache {
    pub(crate) fn new(num_layers: usize) -> Self {
        Self {
            layers: vec![LayerCache::default(); num_layers],
            position: 0,
        }
    }

    pub(crate) fn position(&self) -> usize {
        self.position
    }
}

impl Model {
    pub fn new(config: ModelConfig, weights: ModelWeights) -> Result<Self, EngineError> {
        validate_config(&config)?;
        validate_weights(&config, &weights)?;
        Ok(Self { config, weights })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Bytes held by weight values, excluding allocation and model metadata.
    pub fn stored_weight_bytes(&self) -> usize {
        let mut total = self.weights.token_embeddings.storage_bytes()
            + self.weights.final_norm.len() * size_of::<f32>();
        if let Some(output) = &self.weights.output {
            total += output.storage_bytes();
        }
        for layer in &self.weights.layers {
            total += layer.attention_norm.len() * size_of::<f32>();
            total += layer.feed_forward_norm.len() * size_of::<f32>();
            total += layer.query.storage_bytes();
            total += layer.key.storage_bytes();
            total += layer.value.storage_bytes();
            total += layer.attention_output.storage_bytes();
            total += layer.gate.storage_bytes();
            total += layer.up.storage_bytes();
            total += layer.down.storage_bytes();
        }
        total
    }

    pub(crate) fn forward_token(
        &self,
        token_id: usize,
        cache: &mut KvCache,
    ) -> Result<Vec<f32>, EngineError> {
        if token_id >= self.config.vocab_size {
            return Err(EngineError::InvalidToken(token_id));
        }
        if cache.position >= self.config.max_positions {
            return Err(EngineError::ContextFull);
        }
        let position = cache.position;
        let hidden_size = self.config.hidden_size;
        let head_count = self.config.num_attention_heads;
        let kv_head_count = self.config.num_key_value_heads;
        let head_size = hidden_size / head_count;
        let kv_size = kv_head_count * head_size;
        let heads_per_kv_head = head_count / kv_head_count;
        let mut hidden = self.weights.token_embeddings.row(token_id)?;
        let mut new_keys = Vec::with_capacity(self.config.num_layers);
        let mut new_values = Vec::with_capacity(self.config.num_layers);

        for (layer_index, layer) in self.weights.layers.iter().enumerate() {
            let normalized =
                rms_norm(&hidden, &layer.attention_norm, self.config.rms_norm_epsilon)?;
            let mut query = layer.query.mul_vec(&normalized)?;
            let mut key = layer.key.mul_vec(&normalized)?;
            let value = layer.value.mul_vec(&normalized)?;
            debug_assert_eq!(query.len(), hidden_size);
            debug_assert_eq!(key.len(), kv_size);

            for head in query.chunks_exact_mut(head_size) {
                apply_rope(head, position, self.config.rope_theta);
            }
            for head in key.chunks_exact_mut(head_size) {
                apply_rope(head, position, self.config.rope_theta);
            }

            let layer_cache = &cache.layers[layer_index];
            let mut attended = vec![0.0; hidden_size];
            for query_head in 0..head_count {
                let kv_head = query_head / heads_per_kv_head;
                let query_start = query_head * head_size;
                let kv_start = kv_head * head_size;
                let query_slice = &query[query_start..query_start + head_size];
                let scale = (head_size as f32).sqrt().recip();
                let mut scores = Vec::with_capacity(position + 1);
                for previous_key in &layer_cache.keys {
                    scores.push(
                        dot(query_slice, &previous_key[kv_start..kv_start + head_size]) * scale,
                    );
                }
                scores.push(dot(query_slice, &key[kv_start..kv_start + head_size]) * scale);
                let probabilities = softmax(&scores);
                for (step, probability) in probabilities.into_iter().enumerate() {
                    let values = if step == position {
                        &value
                    } else {
                        &layer_cache.values[step]
                    };
                    for offset in 0..head_size {
                        attended[query_start + offset] += probability * values[kv_start + offset];
                    }
                }
            }
            add_in_place(&mut hidden, &layer.attention_output.mul_vec(&attended)?);

            let normalized = rms_norm(
                &hidden,
                &layer.feed_forward_norm,
                self.config.rms_norm_epsilon,
            )?;
            let gate = layer.gate.mul_vec(&normalized)?;
            let up = layer.up.mul_vec(&normalized)?;
            let activated: Vec<f32> = gate
                .into_iter()
                .zip(up)
                .map(|(gate_value, up_value)| silu(gate_value) * up_value)
                .collect();
            add_in_place(&mut hidden, &layer.down.mul_vec(&activated)?);
            new_keys.push(key);
            new_values.push(value);
        }

        let normalized = rms_norm(
            &hidden,
            &self.weights.final_norm,
            self.config.rms_norm_epsilon,
        )?;
        let output = self
            .weights
            .output
            .as_ref()
            .unwrap_or(&self.weights.token_embeddings);
        let logits = output.mul_vec(&normalized)?;
        if !logits.iter().all(|value| value.is_finite()) {
            return Err(EngineError::InvalidValue("next-token scores"));
        }

        for (layer, (key, value)) in cache
            .layers
            .iter_mut()
            .zip(new_keys.into_iter().zip(new_values))
        {
            layer.keys.push(key);
            layer.values.push(value);
        }
        cache.position += 1;
        Ok(logits)
    }
}

fn validate_config(config: &ModelConfig) -> Result<(), EngineError> {
    if config.vocab_size == 0
        || config.hidden_size == 0
        || config.intermediate_size == 0
        || config.num_layers == 0
        || config.num_attention_heads == 0
        || config.num_key_value_heads == 0
        || config.max_positions == 0
    {
        return Err(EngineError::InvalidConfig(
            "sizes and counts must be positive",
        ));
    }
    if !config
        .hidden_size
        .is_multiple_of(config.num_attention_heads)
    {
        return Err(EngineError::InvalidConfig(
            "hidden size must divide evenly into attention heads",
        ));
    }
    if !config
        .num_attention_heads
        .is_multiple_of(config.num_key_value_heads)
    {
        return Err(EngineError::InvalidConfig(
            "attention heads must divide evenly into key/value heads",
        ));
    }
    if !(config.hidden_size / config.num_attention_heads).is_multiple_of(2) {
        return Err(EngineError::InvalidConfig(
            "attention head size must be even for rotary positions",
        ));
    }
    if !config.rms_norm_epsilon.is_finite() || config.rms_norm_epsilon <= 0.0 {
        return Err(EngineError::InvalidConfig(
            "RMS normalization epsilon must be positive",
        ));
    }
    if !config.rope_theta.is_finite() || config.rope_theta <= 0.0 {
        return Err(EngineError::InvalidConfig(
            "rotary position base must be positive",
        ));
    }
    Ok(())
}

fn validate_weights(config: &ModelConfig, weights: &ModelWeights) -> Result<(), EngineError> {
    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    let head_size = hidden / config.num_attention_heads;
    let kv_size = config.num_key_value_heads * head_size;
    check_matrix(
        "token embeddings",
        &weights.token_embeddings,
        config.vocab_size,
        hidden,
    )?;
    check_vector("final normalization", &weights.final_norm, hidden)?;
    if weights.layers.len() != config.num_layers {
        return Err(EngineError::InvalidShape {
            name: "model layers",
            expected: vec![config.num_layers],
            actual: vec![weights.layers.len()],
        });
    }
    if let Some(output) = &weights.output {
        check_matrix("output projection", output, config.vocab_size, hidden)?;
    }
    for layer in &weights.layers {
        check_vector("attention normalization", &layer.attention_norm, hidden)?;
        check_matrix("query projection", &layer.query, hidden, hidden)?;
        check_matrix("key projection", &layer.key, kv_size, hidden)?;
        check_matrix("value projection", &layer.value, kv_size, hidden)?;
        check_matrix(
            "attention output projection",
            &layer.attention_output,
            hidden,
            hidden,
        )?;
        check_vector(
            "feed-forward normalization",
            &layer.feed_forward_norm,
            hidden,
        )?;
        check_matrix("gate projection", &layer.gate, intermediate, hidden)?;
        check_matrix("up projection", &layer.up, intermediate, hidden)?;
        check_matrix("down projection", &layer.down, hidden, intermediate)?;
    }
    Ok(())
}

fn check_matrix(
    name: &'static str,
    matrix: &Matrix,
    rows: usize,
    cols: usize,
) -> Result<(), EngineError> {
    if matrix.rows() != rows || matrix.cols() != cols {
        return Err(EngineError::InvalidShape {
            name,
            expected: vec![rows, cols],
            actual: vec![matrix.rows(), matrix.cols()],
        });
    }
    Ok(())
}

fn check_vector(name: &'static str, values: &[f32], size: usize) -> Result<(), EngineError> {
    if values.len() != size {
        return Err(EngineError::InvalidShape {
            name,
            expected: vec![size],
            actual: vec![values.len()],
        });
    }
    if !values.iter().all(|value| value.is_finite()) {
        return Err(EngineError::InvalidValue(name));
    }
    Ok(())
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn add_in_place(target: &mut [f32], source: &[f32]) {
    for (target_value, source_value) in target.iter_mut().zip(source) {
        *target_value += source_value;
    }
}
