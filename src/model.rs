#[cfg(target_os = "macos")]
use crate::metal_backend::{DecoderStep, MetalBackend, MetalKvCache};
use crate::tensor::{apply_rope, apply_rope_interleaved, rms_norm, silu, softmax};
use crate::{EngineError, Matrix};
use std::mem::size_of;
use std::ops::Range;

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
    pub rope_interleaved: bool,
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

/// One end of a decoder divided at a layer boundary.
#[derive(Clone, Debug)]
pub struct ModelStage {
    config: ModelConfig,
    range: Range<usize>,
    weights: StageWeights,
    model_digest: [u8; 32],
}

#[derive(Clone, Debug)]
pub(crate) struct StageWeights {
    pub(crate) token_embeddings: Option<Matrix>,
    pub(crate) layers: Vec<LayerWeights>,
    pub(crate) final_norm: Option<Vec<f32>>,
    pub(crate) output: Option<Matrix>,
}

/// Key/value state for the layers owned by one model stage.
/// A caller must validate the stage pair before exchanging activations and discard
/// both sessions if either stage fails during a split request.
pub struct StageSession<'a> {
    stage: &'a ModelStage,
    cache: KvCache,
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

struct LayerRun {
    hidden: Vec<f32>,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
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

    pub(crate) fn truncate(&mut self, position: usize) {
        debug_assert!(position <= self.position);
        for layer in &mut self.layers {
            layer.keys.truncate(position);
            layer.values.truncate(position);
            layer.keys.shrink_to_fit();
            layer.values.shrink_to_fit();
        }
        self.position = position;
    }

    pub(crate) fn allocated_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| {
                (layer.keys.capacity() + layer.values.capacity()) * size_of::<Vec<f32>>()
                    + layer
                        .keys
                        .iter()
                        .chain(layer.values.iter())
                        .map(|values| values.capacity() * size_of::<f32>())
                        .sum::<usize>()
            })
            .sum()
    }

    fn commit(&mut self, keys: Vec<Vec<f32>>, values: Vec<Vec<f32>>) {
        debug_assert_eq!(keys.len(), self.layers.len());
        debug_assert_eq!(values.len(), self.layers.len());
        for (layer, (key, value)) in self.layers.iter_mut().zip(keys.into_iter().zip(values)) {
            layer.keys.push(key);
            layer.values.push(value);
        }
        self.position += 1;
    }
}

impl ModelStage {
    pub(crate) fn new(
        config: ModelConfig,
        range: Range<usize>,
        weights: StageWeights,
        model_digest: [u8; 32],
    ) -> Result<Self, EngineError> {
        validate_config(&config)?;
        if range.start >= range.end
            || range.end > config.num_layers
            || (range.start > 0 && range.end < config.num_layers)
            || (range.start == 0 && range.end == config.num_layers)
        {
            return Err(EngineError::InvalidConfig(
                "stage must be a proper prefix or suffix layer range",
            ));
        }
        let prefix = range.start == 0;
        let suffix = range.end == config.num_layers;
        if weights.layers.len() != range.end - range.start
            || weights.token_embeddings.is_some() != prefix
            || weights.final_norm.is_some() != suffix
            || weights.output.is_some() != suffix
        {
            return Err(EngineError::InvalidConfig("stage endpoint weights"));
        }
        if let Some(embeddings) = &weights.token_embeddings {
            check_matrix(
                "token embeddings",
                embeddings,
                config.vocab_size,
                config.hidden_size,
            )?;
        }
        if let Some(norm) = &weights.final_norm {
            check_vector("final normalization", norm, config.hidden_size)?;
        }
        if let Some(output) = &weights.output {
            check_matrix(
                "output projection",
                output,
                config.vocab_size,
                config.hidden_size,
            )?;
        }
        validate_layer_weights(&config, &weights.layers)?;
        Ok(Self {
            config,
            range,
            weights,
            model_digest,
        })
    }

    pub fn layer_range(&self) -> Range<usize> {
        self.range.clone()
    }

    pub(crate) fn is_prefix(&self) -> bool {
        self.range.start == 0
    }

    pub(crate) fn is_suffix(&self) -> bool {
        self.range.end == self.config.num_layers
    }

    pub fn model_digest(&self) -> [u8; 32] {
        self.model_digest
    }

    pub fn validate_pair(prefix: &Self, suffix: &Self) -> Result<(), EngineError> {
        if prefix.range.start != 0
            || suffix.range.end != prefix.config.num_layers
            || prefix.range.end != suffix.range.start
            || prefix.model_digest != suffix.model_digest
        {
            return Err(EngineError::InvalidConfig(
                "model stages must have the same model and adjacent layer ranges",
            ));
        }
        Ok(())
    }

    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    pub fn max_positions(&self) -> usize {
        self.config.max_positions
    }

    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    pub fn stored_weight_bytes(&self) -> usize {
        let mut total = self
            .weights
            .token_embeddings
            .as_ref()
            .map_or(0, Matrix::storage_bytes)
            + self
                .weights
                .output
                .as_ref()
                .map_or(0, Matrix::storage_bytes)
            + self
                .weights
                .final_norm
                .as_ref()
                .map_or(0, |norm| norm.len() * size_of::<f32>());
        for layer in &self.weights.layers {
            total += layer_weight_bytes(layer);
        }
        total
    }
}

impl<'a> StageSession<'a> {
    pub fn new(stage: &'a ModelStage) -> Self {
        Self {
            stage,
            cache: KvCache::new(stage.weights.layers.len()),
        }
    }

    pub fn position(&self) -> usize {
        self.cache.position()
    }

    pub fn allocated_cache_bytes(&self) -> usize {
        self.cache.allocated_bytes()
    }

    /// Restore a completed prompt checkpoint after speculative generation.
    pub fn rewind(&mut self, position: usize) -> Result<(), EngineError> {
        if position == 0 || position > self.cache.position() {
            return Err(EngineError::InvalidConfig(
                "invalid stage checkpoint position",
            ));
        }
        self.cache.truncate(position);
        Ok(())
    }

    /// Run an input token through a prefix stage and return its hidden activation.
    pub fn forward_token(&mut self, token_id: usize) -> Result<Vec<f32>, EngineError> {
        if self.stage.range.start != 0 {
            return Err(EngineError::InvalidConfig(
                "token input requires a prefix stage",
            ));
        }
        if token_id >= self.stage.config.vocab_size {
            return Err(EngineError::InvalidToken(token_id));
        }
        if self.cache.position >= self.stage.config.max_positions {
            return Err(EngineError::ContextFull);
        }
        let hidden = self
            .stage
            .weights
            .token_embeddings
            .as_ref()
            .expect("validated prefix embedding")
            .row(token_id)?;
        let run = execute_layers(
            &self.stage.config,
            &self.stage.weights.layers,
            &self.cache.layers,
            self.cache.position,
            hidden,
            &mut cpu_multiply,
        )?;
        self.cache.commit(run.keys, run.values);
        Ok(run.hidden)
    }

    /// Run a received activation through a suffix stage and return next-token scores.
    pub fn forward_hidden(&mut self, hidden: Vec<f32>) -> Result<Vec<f32>, EngineError> {
        if self.stage.range.end != self.stage.config.num_layers {
            return Err(EngineError::InvalidConfig(
                "score output requires a suffix stage",
            ));
        }
        if self.cache.position >= self.stage.config.max_positions {
            return Err(EngineError::ContextFull);
        }
        let run = execute_layers(
            &self.stage.config,
            &self.stage.weights.layers,
            &self.cache.layers,
            self.cache.position,
            hidden,
            &mut cpu_multiply,
        )?;
        let scores = project_logits(
            &self.stage.config,
            self.stage
                .weights
                .final_norm
                .as_ref()
                .expect("validated suffix final norm"),
            self.stage
                .weights
                .output
                .as_ref()
                .expect("validated suffix output"),
            &run.hidden,
            &mut cpu_multiply,
        )?;
        self.cache.commit(run.keys, run.values);
        Ok(scores)
    }

    /// Validate one versioned activation and execute this suffix at its current position.
    pub fn forward_frame(
        &mut self,
        frame: &[u8],
        request_id: uuid::Uuid,
    ) -> Result<Vec<f32>, crate::ActivationError> {
        let activation = crate::ActivationFrame::decode_for_stage(
            frame,
            self.stage,
            request_id,
            self.cache.position,
        )?;
        self.forward_hidden(activation.into_hidden())
            .map_err(crate::ActivationError::from)
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

    #[cfg(target_os = "macos")]
    pub(crate) fn matrices(&self) -> Vec<&Matrix> {
        let mut matrices = Vec::with_capacity(1 + self.weights.layers.len() * 7);
        matrices.push(
            self.weights
                .output
                .as_ref()
                .unwrap_or(&self.weights.token_embeddings),
        );
        for layer in &self.weights.layers {
            matrices.extend([
                &layer.query,
                &layer.key,
                &layer.value,
                &layer.attention_output,
                &layer.gate,
                &layer.up,
                &layer.down,
            ]);
        }
        matrices
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn layer_norms(&self) -> Vec<(&[f32], &[f32])> {
        self.weights
            .layers
            .iter()
            .map(|layer| {
                (
                    layer.attention_norm.as_slice(),
                    layer.feed_forward_norm.as_slice(),
                )
            })
            .collect()
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn final_norm_weights(&self) -> &[f32] {
        &self.weights.final_norm
    }

    /// Bytes held by weight values, excluding allocation and model metadata.
    pub fn stored_weight_bytes(&self) -> usize {
        let mut total = self.weights.token_embeddings.storage_bytes()
            + self.weights.final_norm.len() * size_of::<f32>();
        if let Some(output) = &self.weights.output {
            total += output.storage_bytes();
        }
        for layer in &self.weights.layers {
            total += layer_weight_bytes(layer);
        }
        total
    }

    pub(crate) fn forward_token(
        &self,
        token_id: usize,
        cache: &mut KvCache,
        multiply: impl FnMut(&[&Matrix], &[f32]) -> Result<Vec<Vec<f32>>, EngineError>,
    ) -> Result<Vec<f32>, EngineError> {
        let all_layers = 0..self.config.num_layers;
        self.forward_token_ranges(token_id, cache, std::slice::from_ref(&all_layers), multiply)
    }

    /// One Metal command runs the decoder and output projection for a token. The
    /// session commits a position only after the complete command succeeds.
    #[cfg(target_os = "macos")]
    pub(crate) fn forward_token_metal(
        &self,
        token_id: usize,
        cache: &mut KvCache,
        metal_cache: &mut MetalKvCache,
        backend: &mut MetalBackend<'_>,
    ) -> Result<Vec<f32>, EngineError> {
        if token_id >= self.config.vocab_size {
            return Err(EngineError::InvalidToken(token_id));
        }
        if cache.position >= self.config.max_positions {
            return Err(EngineError::ContextFull);
        }
        if !cache.layers.is_empty() {
            return Err(EngineError::InvalidConfig("Metal cache has CPU layers"));
        }
        let position = cache.position;
        let hidden = self.embed_token(token_id)?;
        let head_count = self.config.num_attention_heads;
        let kv_head_count = self.config.num_key_value_heads;
        let head_size = self.config.hidden_size / head_count;
        let mut rotations = Vec::with_capacity(head_size);
        for index in 0..head_size / 2 {
            let frequency = self
                .config
                .rope_theta
                .powf(-((2 * index) as f32) / head_size as f32);
            let (sine, cosine) = (position as f32 * frequency).sin_cos();
            rotations.extend([sine, cosine]);
        }
        let output = self
            .weights
            .output
            .as_ref()
            .unwrap_or(&self.weights.token_embeddings);
        let logits = backend.run_decoder(
            metal_cache,
            DecoderStep {
                position,
                head_count,
                kv_head_count,
                layers: &self.weights.layers,
                hidden: &hidden,
                final_norm: &self.weights.final_norm,
                output,
                epsilon: self.config.rms_norm_epsilon,
                rotations: &rotations,
                interleaved: self.config.rope_interleaved,
            },
        )?;
        if !logits.iter().all(|value| value.is_finite()) {
            return Err(EngineError::InvalidValue("next-token scores"));
        }
        cache.commit(Vec::new(), Vec::new());
        Ok(logits)
    }

    fn forward_token_ranges(
        &self,
        token_id: usize,
        cache: &mut KvCache,
        ranges: &[Range<usize>],
        mut multiply: impl FnMut(&[&Matrix], &[f32]) -> Result<Vec<Vec<f32>>, EngineError>,
    ) -> Result<Vec<f32>, EngineError> {
        if token_id >= self.config.vocab_size {
            return Err(EngineError::InvalidToken(token_id));
        }
        if cache.position >= self.config.max_positions {
            return Err(EngineError::ContextFull);
        }
        let position = cache.position;
        let mut hidden = self.embed_token(token_id)?;
        let mut new_keys = Vec::with_capacity(self.config.num_layers);
        let mut new_values = Vec::with_capacity(self.config.num_layers);
        let mut next_layer = 0;
        for range in ranges {
            if range.start != next_layer || range.is_empty() || range.end > self.config.num_layers {
                return Err(EngineError::InvalidConfig("invalid decoder layer ranges"));
            }
            let run = execute_layers(
                &self.config,
                &self.weights.layers[range.clone()],
                &cache.layers[range.clone()],
                position,
                hidden,
                &mut multiply,
            )?;
            hidden = run.hidden;
            new_keys.extend(run.keys);
            new_values.extend(run.values);
            next_layer = range.end;
        }
        if next_layer != self.config.num_layers {
            return Err(EngineError::InvalidConfig(
                "incomplete decoder layer ranges",
            ));
        }

        let logits = self.project_logits(&hidden, &mut multiply)?;

        cache.commit(new_keys, new_values);
        Ok(logits)
    }

    fn embed_token(&self, token_id: usize) -> Result<Vec<f32>, EngineError> {
        self.weights.token_embeddings.row(token_id)
    }

    fn project_logits(
        &self,
        hidden: &[f32],
        multiply: &mut impl FnMut(&[&Matrix], &[f32]) -> Result<Vec<Vec<f32>>, EngineError>,
    ) -> Result<Vec<f32>, EngineError> {
        let output = self
            .weights
            .output
            .as_ref()
            .unwrap_or(&self.weights.token_embeddings);
        project_logits(
            &self.config,
            &self.weights.final_norm,
            output,
            hidden,
            multiply,
        )
    }
}

fn layer_weight_bytes(layer: &LayerWeights) -> usize {
    (layer.attention_norm.len() + layer.feed_forward_norm.len()) * size_of::<f32>()
        + layer.query.storage_bytes()
        + layer.key.storage_bytes()
        + layer.value.storage_bytes()
        + layer.attention_output.storage_bytes()
        + layer.gate.storage_bytes()
        + layer.up.storage_bytes()
        + layer.down.storage_bytes()
}

fn cpu_multiply(matrices: &[&Matrix], input: &[f32]) -> Result<Vec<Vec<f32>>, EngineError> {
    matrices
        .iter()
        .map(|matrix| matrix.mul_vec(input))
        .collect()
}

fn project_logits(
    config: &ModelConfig,
    final_norm: &[f32],
    output: &Matrix,
    hidden: &[f32],
    multiply: &mut impl FnMut(&[&Matrix], &[f32]) -> Result<Vec<Vec<f32>>, EngineError>,
) -> Result<Vec<f32>, EngineError> {
    let normalized = rms_norm(hidden, final_norm, config.rms_norm_epsilon)?;
    let scores = multiply_one(multiply, output, &normalized)?;
    if !scores.iter().all(|value| value.is_finite()) {
        return Err(EngineError::InvalidValue("next-token scores"));
    }
    Ok(scores)
}

fn execute_layers(
    config: &ModelConfig,
    layers: &[LayerWeights],
    caches: &[LayerCache],
    position: usize,
    mut hidden: Vec<f32>,
    multiply: &mut impl FnMut(&[&Matrix], &[f32]) -> Result<Vec<Vec<f32>>, EngineError>,
) -> Result<LayerRun, EngineError> {
    if hidden.len() != config.hidden_size {
        return Err(EngineError::InvalidShape {
            name: "decoder hidden state",
            expected: vec![config.hidden_size],
            actual: vec![hidden.len()],
        });
    }
    if !hidden.iter().all(|value| value.is_finite()) {
        return Err(EngineError::InvalidValue("decoder hidden state"));
    }
    if layers.len() != caches.len() {
        return Err(EngineError::InvalidConfig("decoder cache layer count"));
    }
    let hidden_size = config.hidden_size;
    let head_count = config.num_attention_heads;
    let kv_head_count = config.num_key_value_heads;
    let head_size = hidden_size / head_count;
    let kv_size = kv_head_count * head_size;
    let heads_per_kv_head = head_count / kv_head_count;
    let mut keys = Vec::with_capacity(layers.len());
    let mut values = Vec::with_capacity(layers.len());

    for (layer, layer_cache) in layers.iter().zip(caches) {
        if layer_cache.keys.len() != position || layer_cache.values.len() != position {
            return Err(EngineError::Backend(
                "decoder cache position mismatch".into(),
            ));
        }
        let normalized = rms_norm(&hidden, &layer.attention_norm, config.rms_norm_epsilon)?;
        let [mut query, mut key, value]: [Vec<f32>; 3] =
            multiply(&[&layer.query, &layer.key, &layer.value], &normalized)?
                .try_into()
                .map_err(|_| EngineError::Backend("attention projection count".into()))?;
        debug_assert_eq!(query.len(), hidden_size);
        debug_assert_eq!(key.len(), kv_size);

        for head in query.chunks_exact_mut(head_size) {
            if config.rope_interleaved {
                apply_rope_interleaved(head, position, config.rope_theta);
            } else {
                apply_rope(head, position, config.rope_theta);
            }
        }
        for head in key.chunks_exact_mut(head_size) {
            if config.rope_interleaved {
                apply_rope_interleaved(head, position, config.rope_theta);
            } else {
                apply_rope(head, position, config.rope_theta);
            }
        }

        let mut attended = vec![0.0; hidden_size];
        for query_head in 0..head_count {
            let kv_head = query_head / heads_per_kv_head;
            let query_start = query_head * head_size;
            let kv_start = kv_head * head_size;
            let query_slice = &query[query_start..query_start + head_size];
            let scale = (head_size as f32).sqrt().recip();
            let mut scores = Vec::with_capacity(position + 1);
            for previous_key in &layer_cache.keys {
                scores
                    .push(dot(query_slice, &previous_key[kv_start..kv_start + head_size]) * scale);
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
        add_in_place(
            &mut hidden,
            &multiply_one(multiply, &layer.attention_output, &attended)?,
        );

        let normalized = rms_norm(&hidden, &layer.feed_forward_norm, config.rms_norm_epsilon)?;
        let [gate, up]: [Vec<f32>; 2] = multiply(&[&layer.gate, &layer.up], &normalized)?
            .try_into()
            .map_err(|_| EngineError::Backend("feed-forward projection count".into()))?;
        let activated: Vec<f32> = gate
            .into_iter()
            .zip(up)
            .map(|(gate_value, up_value)| silu(gate_value) * up_value)
            .collect();
        add_in_place(
            &mut hidden,
            &multiply_one(multiply, &layer.down, &activated)?,
        );
        keys.push(key);
        values.push(value);
    }
    Ok(LayerRun {
        hidden,
        keys,
        values,
    })
}

fn multiply_one(
    multiply: &mut impl FnMut(&[&Matrix], &[f32]) -> Result<Vec<Vec<f32>>, EngineError>,
    matrix: &Matrix,
    input: &[f32],
) -> Result<Vec<f32>, EngineError> {
    multiply(&[matrix], input)?
        .into_iter()
        .next()
        .ok_or_else(|| EngineError::Backend("missing matrix result".into()))
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
    validate_layer_weights(config, &weights.layers)
}

fn validate_layer_weights(
    config: &ModelConfig,
    layers: &[LayerWeights],
) -> Result<(), EngineError> {
    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    let head_size = hidden / config.num_attention_heads;
    let kv_size = config.num_key_value_heads * head_size;
    for layer in layers {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_multiply(matrices: &[&Matrix], input: &[f32]) -> Result<Vec<Vec<f32>>, EngineError> {
        matrices
            .iter()
            .map(|matrix| matrix.mul_vec(input))
            .collect()
    }

    fn tiny_model() -> Model {
        fn matrix(seed: usize, rows: usize, columns: usize) -> Matrix {
            Matrix::new(
                rows,
                columns,
                (0..rows * columns)
                    .map(|index| ((index * 17 + seed * 13) % 23) as f32 / 20.0 - 0.55)
                    .collect(),
            )
            .unwrap()
        }
        let config = ModelConfig {
            vocab_size: 8,
            hidden_size: 4,
            intermediate_size: 6,
            num_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            max_positions: 8,
            rms_norm_epsilon: 1e-5,
            rope_theta: 10000.0,
            rope_interleaved: false,
        };
        let layers = (0..2)
            .map(|index| {
                let base = 2 + index * 7;
                LayerWeights {
                    attention_norm: vec![1.0; 4],
                    query: matrix(base, 4, 4),
                    key: matrix(base + 1, 2, 4),
                    value: matrix(base + 2, 2, 4),
                    attention_output: matrix(base + 3, 4, 4),
                    feed_forward_norm: vec![1.0; 4],
                    gate: matrix(base + 4, 6, 4),
                    up: matrix(base + 5, 6, 4),
                    down: matrix(base + 6, 4, 6),
                }
            })
            .collect();
        Model::new(
            config,
            ModelWeights {
                token_embeddings: matrix(1, 8, 4),
                layers,
                final_norm: vec![1.0; 4],
                output: None,
            },
        )
        .unwrap()
    }

    #[test]
    fn two_layer_ranges_match_full_step_and_do_not_commit_a_failed_step() {
        let model = tiny_model();
        let mut full = KvCache::new(2);
        let mut split = KvCache::new(2);
        for token in [1, 2, 3, 4] {
            let expected = model.forward_token(token, &mut full, cpu_multiply).unwrap();
            let actual = model
                .forward_token_ranges(token, &mut split, &[0..1, 1..2], cpu_multiply)
                .unwrap();
            assert_eq!(actual, expected);
            assert_eq!(split.position(), full.position());
            assert_eq!(split.layers[0].keys.len(), split.position());
            assert_eq!(split.layers[1].keys.len(), split.position());
        }

        let position = split.position();
        let error = model.forward_token_ranges(5, &mut split, &[0..1, 1..2], |matrices, input| {
            if input.len() == model.config.hidden_size && matrices.len() == 1 {
                return Err(EngineError::Backend("injected failure".into()));
            }
            cpu_multiply(matrices, input)
        });
        assert_eq!(error, Err(EngineError::Backend("injected failure".into())));
        assert_eq!(split.position(), position);
        assert!(split
            .layers
            .iter()
            .all(|layer| layer.keys.len() == position && layer.values.len() == position));
    }

    #[test]
    #[ignore = "requires SmolLM2-135M-Q4_K_M.gguf in SMOLLM2_DIR"]
    fn real_q4_k_model_matches_at_layer_boundary() {
        let directory = std::env::var("SMOLLM2_DIR").expect("set SMOLLM2_DIR");
        let model =
            crate::load_gguf(std::path::Path::new(&directory).join("SmolLM2-135M-Q4_K_M.gguf"))
                .unwrap();
        let split_at = model.config.num_layers / 2;
        let mut full = KvCache::new(model.config.num_layers);
        let mut split = KvCache::new(model.config.num_layers);
        for token in [1, 2, 3, 30] {
            let expected = model.forward_token(token, &mut full, cpu_multiply).unwrap();
            let actual = model
                .forward_token_ranges(
                    token,
                    &mut split,
                    &[0..split_at, split_at..model.config.num_layers],
                    cpu_multiply,
                )
                .unwrap();
            assert_eq!(actual, expected);
            assert_eq!(split.position(), full.position());
        }
    }
}
