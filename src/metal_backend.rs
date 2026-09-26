use std::collections::HashMap;
use std::mem::size_of;
use std::sync::{Arc, Mutex};

use metal::{
    Buffer, CommandBufferRef, CommandQueue, CompileOptions, ComputeCommandEncoderRef,
    ComputePipelineState, Device, MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
};

use crate::{
    EngineError, GenerationSession, LayerWeights, Matrix, Model, ModelStage, StageSession,
};

#[repr(C)]
struct Params {
    rows: u32,
    cols: u32,
    kind: u32,
    reserved: u32,
}

#[repr(C)]
struct AttentionParams {
    head_count: u32,
    kv_head_count: u32,
    head_size: u32,
    kv_size: u32,
    sequence_length: u32,
}

#[repr(C)]
struct RopeParams {
    head_count: u32,
    kv_head_count: u32,
    head_size: u32,
    kv_size: u32,
    position: u32,
    interleaved: u32,
}

#[repr(C)]
struct NormParams {
    count: u32,
    epsilon: f32,
}

pub(crate) struct DecoderStep<'a> {
    pub(crate) position: usize,
    pub(crate) head_count: usize,
    pub(crate) kv_head_count: usize,
    pub(crate) layers: &'a [LayerWeights],
    pub(crate) hidden: &'a [f32],
    pub(crate) final_norm: Option<&'a [f32]>,
    pub(crate) output: Option<&'a Matrix>,
    pub(crate) epsilon: f32,
    pub(crate) rotations: &'a [f32],
    pub(crate) interleaved: bool,
}

struct LayerBuffers<'a> {
    hidden: &'a Buffer,
    rotations: &'a Buffer,
    retained: &'a mut Vec<Buffer>,
}

#[derive(Default)]
struct MetalLayerCache {
    keys: Option<Buffer>,
    values: Option<Buffer>,
    capacity: usize,
}

/// Per-session GPU buffers. The separate session position is the commit marker.
pub(crate) struct MetalKvCache {
    layers: Vec<MetalLayerCache>,
    kv_size: usize,
    max_positions: usize,
}

impl MetalKvCache {
    pub(crate) fn new(num_layers: usize, kv_size: usize, max_positions: usize) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|_| MetalLayerCache::default())
                .collect(),
            kv_size,
            max_positions,
        }
    }

    pub(crate) fn allocated_bytes(&self) -> usize {
        self.layers.iter().fold(0_usize, |total, layer| {
            total.saturating_add(
                layer
                    .capacity
                    .saturating_mul(self.kv_size)
                    .saturating_mul(2 * size_of::<f32>()),
            )
        })
    }

    fn layer_at(
        &mut self,
        device: &Device,
        layer_index: usize,
        position: usize,
    ) -> Result<&MetalLayerCache, EngineError> {
        if position >= self.max_positions {
            return Err(EngineError::ContextFull);
        }
        let layer = self
            .layers
            .get_mut(layer_index)
            .ok_or(EngineError::InvalidConfig("Metal attention layer"))?;
        if layer.capacity <= position {
            let mut capacity = layer.capacity.max(16).min(self.max_positions);
            while capacity <= position {
                capacity = capacity.saturating_mul(2).min(self.max_positions);
            }
            let bytes = capacity
                .checked_mul(self.kv_size)
                .and_then(|values| values.checked_mul(size_of::<f32>()))
                .ok_or_else(|| EngineError::Backend("Metal cache size overflow".into()))?;
            if bytes as u64 > device.max_buffer_length() {
                return Err(EngineError::Backend("Metal cache buffer limit".into()));
            }
            let keys = device.new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
            let values = device.new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
            let old_bytes = position * self.kv_size * size_of::<f32>();
            if let (Some(old_keys), Some(old_values)) = (&layer.keys, &layer.values) {
                // The session has completed all earlier GPU commands before growth.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        old_keys.contents().cast::<u8>(),
                        keys.contents().cast::<u8>(),
                        old_bytes,
                    );
                    std::ptr::copy_nonoverlapping(
                        old_values.contents().cast::<u8>(),
                        values.contents().cast::<u8>(),
                        old_bytes,
                    );
                }
            }
            layer.keys = Some(keys);
            layer.values = Some(values);
            layer.capacity = capacity;
        }
        Ok(layer)
    }
}

pub(crate) struct MetalBackend {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    rotate_and_store: ComputePipelineState,
    attention_scores: ComputePipelineState,
    attention_reduce: ComputePipelineState,
    silu_multiply: ComputePipelineState,
    rms_scale: ComputePipelineState,
    rms_apply: ComputePipelineState,
    add_vectors: ComputePipelineState,
    weights: HashMap<usize, (Buffer, u32)>,
    norms: Vec<(Buffer, Buffer)>,
    final_norm: Option<Buffer>,
}

/// A prepared Mac GPU runtime. Multiple generation sessions reuse its uploaded weights.
pub struct MetalRuntime<'a> {
    model: &'a Model,
    backend: Arc<Mutex<MetalBackend>>,
}

/// A prepared GPU runtime for one owned range of decoder layers.
pub struct MetalStageRuntime<'a> {
    stage: &'a ModelStage,
    backend: Arc<Mutex<MetalBackend>>,
}

impl<'a> MetalRuntime<'a> {
    pub fn new(model: &'a Model) -> Result<Self, EngineError> {
        Ok(Self {
            model,
            backend: Arc::new(Mutex::new(MetalBackend::new(model)?)),
        })
    }

    pub fn session(&self) -> GenerationSession<'a> {
        GenerationSession::with_metal_backend(self.model, Arc::clone(&self.backend))
    }
}

impl<'a> MetalStageRuntime<'a> {
    pub fn new(stage: &'a ModelStage) -> Result<Self, EngineError> {
        Ok(Self {
            stage,
            backend: Arc::new(Mutex::new(MetalBackend::new_stage(stage)?)),
        })
    }

    pub fn session(&self) -> StageSession<'a> {
        StageSession::with_metal_backend(self.stage, Arc::clone(&self.backend))
    }
}

impl MetalBackend {
    pub(crate) fn new(model: &Model) -> Result<Self, EngineError> {
        Self::with_weights(
            model.matrices(),
            model.layer_norms(),
            Some(model.final_norm_weights()),
        )
    }

    pub(crate) fn new_stage(stage: &ModelStage) -> Result<Self, EngineError> {
        Self::with_weights(
            stage.matrices(),
            stage.layer_norms(),
            stage.final_norm_weights(),
        )
    }

    fn with_weights(
        matrices: Vec<&Matrix>,
        layer_norms: Vec<(&[f32], &[f32])>,
        final_norm_weights: Option<&[f32]>,
    ) -> Result<Self, EngineError> {
        let device = Device::system_default()
            .ok_or_else(|| EngineError::Backend("no Metal device is available".into()))?;
        let library = device
            .new_library_with_source(include_str!("matvec.metal"), &CompileOptions::new())
            .map_err(|error| EngineError::Backend(format!("Metal shader: {error}")))?;
        let function = library
            .get_function("matvec", None)
            .map_err(|error| EngineError::Backend(format!("Metal function: {error}")))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|error| EngineError::Backend(format!("Metal pipeline: {error}")))?;
        let scores_function = library
            .get_function("attention_scores", None)
            .map_err(|error| EngineError::Backend(format!("Metal function: {error}")))?;
        let attention_scores = device
            .new_compute_pipeline_state_with_function(&scores_function)
            .map_err(|error| EngineError::Backend(format!("Metal attention pipeline: {error}")))?;
        let rotate_function = library
            .get_function("rotate_and_store", None)
            .map_err(|error| EngineError::Backend(format!("Metal function: {error}")))?;
        let rotate_and_store = device
            .new_compute_pipeline_state_with_function(&rotate_function)
            .map_err(|error| EngineError::Backend(format!("Metal rotary pipeline: {error}")))?;
        let reduce_function = library
            .get_function("attention_reduce", None)
            .map_err(|error| EngineError::Backend(format!("Metal function: {error}")))?;
        let attention_reduce = device
            .new_compute_pipeline_state_with_function(&reduce_function)
            .map_err(|error| EngineError::Backend(format!("Metal attention pipeline: {error}")))?;
        let silu_function = library
            .get_function("silu_multiply", None)
            .map_err(|error| EngineError::Backend(format!("Metal function: {error}")))?;
        let silu_multiply = device
            .new_compute_pipeline_state_with_function(&silu_function)
            .map_err(|error| EngineError::Backend(format!("Metal activation pipeline: {error}")))?;
        let pipeline_named = |name: &str| {
            let function = library
                .get_function(name, None)
                .map_err(|error| EngineError::Backend(format!("Metal function: {error}")))?;
            device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|error| EngineError::Backend(format!("Metal {name} pipeline: {error}")))
        };
        let rms_scale = pipeline_named("rms_scale")?;
        let rms_apply = pipeline_named("rms_apply")?;
        let add_vectors = pipeline_named("add_vectors")?;
        let queue = device.new_command_queue();
        let mut weights = HashMap::new();
        for matrix in matrices {
            let (pointer, length, kind) = matrix.metal_storage();
            if length as u64 > device.max_buffer_length() {
                return Err(EngineError::Backend(
                    "matrix exceeds Metal buffer limit".into(),
                ));
            }
            let buffer = device.new_buffer_with_data(
                pointer,
                length as u64,
                MTLResourceOptions::StorageModeShared,
            );
            weights.insert(matrix as *const Matrix as usize, (buffer, kind));
        }
        let upload = |values: &[f32]| -> Result<Buffer, EngineError> {
            let bytes = size_of_val(values) as u64;
            if bytes > device.max_buffer_length() {
                return Err(EngineError::Backend("Metal norm buffer limit".into()));
            }
            Ok(device.new_buffer_with_data(
                values.as_ptr().cast(),
                bytes,
                MTLResourceOptions::StorageModeShared,
            ))
        };
        let mut norms = Vec::with_capacity(layer_norms.len());
        for (attention, feed_forward) in layer_norms {
            norms.push((upload(attention)?, upload(feed_forward)?));
        }
        let final_norm = final_norm_weights.map(upload).transpose()?;
        Ok(Self {
            device,
            queue,
            pipeline,
            rotate_and_store,
            attention_scores,
            attention_reduce,
            silu_multiply,
            rms_scale,
            rms_apply,
            add_vectors,
            weights,
            norms,
            final_norm,
        })
    }

    fn encode_matrix(
        &self,
        encoder: &ComputeCommandEncoderRef,
        matrix: &Matrix,
        input: &Buffer,
        output: &Buffer,
    ) -> Result<(), EngineError> {
        let (weight, kind) = self
            .weights
            .get(&(matrix as *const Matrix as usize))
            .ok_or_else(|| EngineError::Backend("matrix was not uploaded to Metal".into()))?;
        let params = Params {
            rows: u32::try_from(matrix.rows())
                .map_err(|_| EngineError::Backend("matrix has too many rows".into()))?,
            cols: u32::try_from(matrix.cols())
                .map_err(|_| EngineError::Backend("matrix has too many columns".into()))?,
            kind: *kind,
            reserved: 0,
        };
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(weight), 0);
        encoder.set_buffer(1, Some(input), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(
            3,
            size_of::<Params>() as u64,
            (&params as *const Params).cast(),
        );
        encoder.dispatch_threads(
            MTLSize::new(u64::from(params.rows), 1, 1),
            MTLSize::new(self.pipeline.thread_execution_width(), 1, 1),
        );
        Ok(())
    }

    fn encode_norm(
        &self,
        command: &CommandBufferRef,
        input: &Buffer,
        weights: &Buffer,
        scale: &Buffer,
        output: &Buffer,
        params: &NormParams,
    ) {
        let scale_encoder = command.new_compute_command_encoder();
        scale_encoder.set_compute_pipeline_state(&self.rms_scale);
        scale_encoder.set_buffer(0, Some(input), 0);
        scale_encoder.set_buffer(1, Some(scale), 0);
        scale_encoder.set_bytes(
            2,
            size_of::<NormParams>() as u64,
            (params as *const NormParams).cast(),
        );
        scale_encoder.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        scale_encoder.end_encoding();

        let apply_encoder = command.new_compute_command_encoder();
        apply_encoder.set_compute_pipeline_state(&self.rms_apply);
        apply_encoder.set_buffer(0, Some(input), 0);
        apply_encoder.set_buffer(1, Some(weights), 0);
        apply_encoder.set_buffer(2, Some(scale), 0);
        apply_encoder.set_buffer(3, Some(output), 0);
        apply_encoder.set_bytes(
            4,
            size_of::<u32>() as u64,
            (&params.count as *const u32).cast(),
        );
        apply_encoder.dispatch_threads(
            MTLSize::new(u64::from(params.count), 1, 1),
            MTLSize::new(self.rms_apply.thread_execution_width(), 1, 1),
        );
        apply_encoder.end_encoding();
    }

    fn encode_add(
        &self,
        command: &CommandBufferRef,
        left: &Buffer,
        right: &Buffer,
        output: &Buffer,
        count: u32,
    ) {
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.add_vectors);
        encoder.set_buffer(0, Some(left), 0);
        encoder.set_buffer(1, Some(right), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(3, size_of::<u32>() as u64, (&count as *const u32).cast());
        encoder.dispatch_threads(
            MTLSize::new(u64::from(count), 1, 1),
            MTLSize::new(self.add_vectors.thread_execution_width(), 1, 1),
        );
        encoder.end_encoding();
    }

    pub(crate) fn run_decoder(
        &mut self,
        cache: &mut MetalKvCache,
        step: DecoderStep<'_>,
    ) -> Result<Vec<f32>, EngineError> {
        if step.layers.is_empty()
            || step.layers.len() != cache.layers.len()
            || step.layers.len() != self.norms.len()
            || step.hidden.is_empty()
            || step.final_norm.is_some() != step.output.is_some()
            || step.final_norm.is_some() != self.final_norm.is_some()
            || step
                .final_norm
                .is_some_and(|norm| norm.len() != step.hidden.len())
            || step
                .output
                .is_some_and(|output| output.cols() != step.hidden.len())
            || !step.hidden.iter().all(|value| value.is_finite())
        {
            return Err(EngineError::InvalidConfig("Metal decoder dimensions"));
        }
        let bytes = |count: usize| {
            count
                .checked_mul(size_of::<f32>())
                .filter(|&size| size as u64 <= self.device.max_buffer_length())
                .map(|size| size as u64)
                .ok_or_else(|| EngineError::Backend("Metal decoder buffer limit".into()))
        };
        let hidden_bytes = bytes(step.hidden.len())?;
        let rotation_bytes = bytes(step.rotations.len())?;
        bytes(1)?;
        let hidden_input = self.device.new_buffer_with_data(
            step.hidden.as_ptr().cast(),
            hidden_bytes,
            MTLResourceOptions::StorageModeShared,
        );
        let rotations = self.device.new_buffer_with_data(
            step.rotations.as_ptr().cast(),
            rotation_bytes,
            MTLResourceOptions::StorageModeShared,
        );
        let command = self.queue.new_command_buffer();
        let mut retained = Vec::with_capacity(step.layers.len() * 16);
        let mut hidden = hidden_input;
        for layer_index in 0..step.layers.len() {
            let mut buffers = LayerBuffers {
                hidden: &hidden,
                rotations: &rotations,
                retained: &mut retained,
            };
            let next = self.encode_layer(command, cache, layer_index, &step, &mut buffers)?;
            retained.push(hidden);
            hidden = next;
        }

        let (result, result_len) =
            if let (Some(output), Some(norm)) = (step.output, self.final_norm.as_ref()) {
                let norm_params = NormParams {
                    count: u32::try_from(step.hidden.len())
                        .map_err(|_| EngineError::Backend("Metal hidden size too large".into()))?,
                    epsilon: step.epsilon,
                };
                let scale = self.device.new_buffer(
                    size_of::<f32>() as u64,
                    MTLResourceOptions::StorageModeShared,
                );
                let normalized = self
                    .device
                    .new_buffer(hidden_bytes, MTLResourceOptions::StorageModeShared);
                let scores = self
                    .device
                    .new_buffer(bytes(output.rows())?, MTLResourceOptions::StorageModeShared);
                self.encode_norm(command, &hidden, norm, &scale, &normalized, &norm_params);
                let projection = command.new_compute_command_encoder();
                self.encode_matrix(projection, output, &normalized, &scores)?;
                projection.end_encoding();
                retained.extend([scale, normalized, hidden]);
                (scores, output.rows())
            } else {
                (hidden, step.hidden.len())
            };

        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(EngineError::Backend(format!(
                "Metal decoder command ended with {:?}",
                command.status()
            )));
        }
        Ok(unsafe {
            std::slice::from_raw_parts(result.contents().cast::<f32>(), result_len).to_vec()
        })
    }

    fn encode_layer(
        &self,
        command: &CommandBufferRef,
        cache: &mut MetalKvCache,
        layer_index: usize,
        step: &DecoderStep<'_>,
        buffers: &mut LayerBuffers<'_>,
    ) -> Result<Buffer, EngineError> {
        let layer_weights = step
            .layers
            .get(layer_index)
            .ok_or(EngineError::InvalidConfig("Metal decoder layer index"))?;
        let (query_matrix, key_matrix, value_matrix) = (
            &layer_weights.query,
            &layer_weights.key,
            &layer_weights.value,
        );
        let hidden_size = step.hidden.len();
        let kv_size = cache.kv_size;
        if hidden_size == 0 || step.head_count == 0 || step.kv_head_count == 0 {
            return Err(EngineError::InvalidConfig("Metal layer dimensions"));
        }
        let head_size = hidden_size / step.head_count;
        let intermediate_size = layer_weights.gate.rows();
        if head_size == 0
            || !head_size.is_multiple_of(2)
            || !step.head_count.is_multiple_of(step.kv_head_count)
            || hidden_size != step.head_count * head_size
            || kv_size != step.kv_head_count * head_size
            || query_matrix.rows() != hidden_size
            || key_matrix.rows() != kv_size
            || value_matrix.rows() != kv_size
            || [query_matrix, key_matrix, value_matrix]
                .iter()
                .any(|matrix| matrix.cols() != hidden_size)
            || layer_weights.attention_output.rows() != hidden_size
            || layer_weights.attention_output.cols() != hidden_size
            || layer_weights.attention_norm.len() != hidden_size
            || layer_weights.feed_forward_norm.len() != hidden_size
            || intermediate_size == 0
            || layer_weights.gate.cols() != hidden_size
            || layer_weights.up.rows() != intermediate_size
            || layer_weights.up.cols() != hidden_size
            || layer_weights.down.rows() != hidden_size
            || layer_weights.down.cols() != intermediate_size
            || step.rotations.len() != head_size
            || !step.epsilon.is_finite()
            || step.epsilon <= 0.0
        {
            return Err(EngineError::InvalidConfig("Metal layer dimensions"));
        }
        let hidden_count = u32::try_from(hidden_size)
            .map_err(|_| EngineError::Backend("Metal hidden size too large".into()))?;
        let intermediate_count = u32::try_from(intermediate_size)
            .map_err(|_| EngineError::Backend("Metal intermediate size too large".into()))?;
        let sequence_length = step
            .position
            .checked_add(1)
            .ok_or_else(|| EngineError::Backend("context too long for Metal".into()))?;
        let params = AttentionParams {
            head_count: u32::try_from(step.head_count)
                .map_err(|_| EngineError::Backend("too many attention heads".into()))?,
            kv_head_count: u32::try_from(step.kv_head_count)
                .map_err(|_| EngineError::Backend("too many key/value heads".into()))?,
            head_size: u32::try_from(head_size)
                .map_err(|_| EngineError::Backend("attention head too wide".into()))?,
            kv_size: u32::try_from(kv_size)
                .map_err(|_| EngineError::Backend("key/value state too wide".into()))?,
            sequence_length: u32::try_from(sequence_length)
                .map_err(|_| EngineError::Backend("context too long for Metal".into()))?,
        };
        let rope_params = RopeParams {
            head_count: params.head_count,
            kv_head_count: params.kv_head_count,
            head_size: params.head_size,
            kv_size: params.kv_size,
            position: u32::try_from(step.position)
                .map_err(|_| EngineError::Backend("context too long for Metal".into()))?,
            interleaved: u32::from(step.interleaved),
        };
        let norm_params = NormParams {
            count: hidden_count,
            epsilon: step.epsilon,
        };
        let score_count = step
            .head_count
            .checked_mul(sequence_length)
            .ok_or_else(|| EngineError::Backend("Metal score size overflow".into()))?;
        u32::try_from(score_count)
            .map_err(|_| EngineError::Backend("too many Metal attention scores".into()))?;
        let rotation_threads = step
            .head_count
            .checked_add(step.kv_head_count)
            .ok_or_else(|| EngineError::Backend("Metal rotary size overflow".into()))?
            .checked_mul(head_size / 2)
            .ok_or_else(|| EngineError::Backend("Metal rotary size overflow".into()))?
            .max(kv_size);
        u32::try_from(rotation_threads)
            .map_err(|_| EngineError::Backend("Metal rotary grid too large".into()))?;
        let f32_bytes = |count: usize| {
            count
                .checked_mul(size_of::<f32>())
                .filter(|&bytes| bytes as u64 <= self.device.max_buffer_length())
                .map(|bytes| bytes as u64)
                .ok_or_else(|| EngineError::Backend("Metal layer buffer limit".into()))
        };
        let hidden_bytes = f32_bytes(hidden_size)?;
        let kv_bytes = f32_bytes(kv_size)?;
        let score_bytes = f32_bytes(score_count)?;
        let intermediate_bytes = f32_bytes(intermediate_size)?;
        f32_bytes(step.rotations.len())?;
        f32_bytes(1)?;
        let (attention_norm, feed_forward_norm) = self
            .norms
            .get(layer_index)
            .ok_or(EngineError::InvalidConfig("Metal layer norm index"))?;
        let layer = cache.layer_at(&self.device, layer_index, step.position)?;
        let keys = layer.keys.as_ref().expect("cache keys allocated");
        let values = layer.values.as_ref().expect("cache values allocated");
        let allocate = |bytes| {
            self.device
                .new_buffer(bytes, MTLResourceOptions::StorageModeShared)
        };
        let attention_scale = allocate(size_of::<f32>() as u64);
        let feed_forward_scale = allocate(size_of::<f32>() as u64);
        let normalized_attention = allocate(hidden_bytes);
        let normalized_feed_forward = allocate(hidden_bytes);
        let query = allocate(hidden_bytes);
        let key = allocate(kv_bytes);
        let value = allocate(kv_bytes);
        let scores = allocate(score_bytes);
        let attended = allocate(hidden_bytes);
        let attention_projection = allocate(hidden_bytes);
        let after_attention = allocate(hidden_bytes);
        let gate = allocate(intermediate_bytes);
        let up = allocate(intermediate_bytes);
        let activated = allocate(intermediate_bytes);
        let down = allocate(hidden_bytes);
        let output = allocate(hidden_bytes);

        self.encode_norm(
            command,
            buffers.hidden,
            attention_norm,
            &attention_scale,
            &normalized_attention,
            &norm_params,
        );
        let projection = command.new_compute_command_encoder();
        for (matrix, target) in [query_matrix, key_matrix, value_matrix]
            .into_iter()
            .zip([&query, &key, &value])
        {
            self.encode_matrix(projection, matrix, &normalized_attention, target)?;
        }
        projection.end_encoding();

        let rotary = command.new_compute_command_encoder();
        rotary.set_compute_pipeline_state(&self.rotate_and_store);
        rotary.set_buffer(0, Some(&query), 0);
        rotary.set_buffer(1, Some(&key), 0);
        rotary.set_buffer(2, Some(&value), 0);
        rotary.set_buffer(3, Some(keys), 0);
        rotary.set_buffer(4, Some(values), 0);
        rotary.set_buffer(5, Some(buffers.rotations), 0);
        rotary.set_bytes(
            6,
            size_of::<RopeParams>() as u64,
            (&rope_params as *const RopeParams).cast(),
        );
        rotary.dispatch_threads(
            MTLSize::new(rotation_threads as u64, 1, 1),
            MTLSize::new(self.rotate_and_store.thread_execution_width(), 1, 1),
        );
        rotary.end_encoding();

        let score_encoder = command.new_compute_command_encoder();
        score_encoder.set_compute_pipeline_state(&self.attention_scores);
        score_encoder.set_buffer(0, Some(&query), 0);
        score_encoder.set_buffer(1, Some(keys), 0);
        score_encoder.set_buffer(2, Some(&scores), 0);
        score_encoder.set_bytes(
            3,
            size_of::<AttentionParams>() as u64,
            (&params as *const AttentionParams).cast(),
        );
        score_encoder.dispatch_threads(
            MTLSize::new(score_count as u64, 1, 1),
            MTLSize::new(self.attention_scores.thread_execution_width(), 1, 1),
        );
        score_encoder.end_encoding();

        let reduce_encoder = command.new_compute_command_encoder();
        reduce_encoder.set_compute_pipeline_state(&self.attention_reduce);
        reduce_encoder.set_buffer(0, Some(&scores), 0);
        reduce_encoder.set_buffer(1, Some(values), 0);
        reduce_encoder.set_buffer(2, Some(&attended), 0);
        reduce_encoder.set_bytes(
            3,
            size_of::<AttentionParams>() as u64,
            (&params as *const AttentionParams).cast(),
        );
        reduce_encoder.dispatch_threads(
            MTLSize::new(hidden_size as u64, 1, 1),
            MTLSize::new(self.attention_reduce.thread_execution_width(), 1, 1),
        );
        reduce_encoder.end_encoding();

        let attention_output_encoder = command.new_compute_command_encoder();
        self.encode_matrix(
            attention_output_encoder,
            &layer_weights.attention_output,
            &attended,
            &attention_projection,
        )?;
        attention_output_encoder.end_encoding();
        self.encode_add(
            command,
            buffers.hidden,
            &attention_projection,
            &after_attention,
            hidden_count,
        );
        self.encode_norm(
            command,
            &after_attention,
            feed_forward_norm,
            &feed_forward_scale,
            &normalized_feed_forward,
            &norm_params,
        );

        let feed_forward_encoder = command.new_compute_command_encoder();
        self.encode_matrix(
            feed_forward_encoder,
            &layer_weights.gate,
            &normalized_feed_forward,
            &gate,
        )?;
        self.encode_matrix(
            feed_forward_encoder,
            &layer_weights.up,
            &normalized_feed_forward,
            &up,
        )?;
        feed_forward_encoder.end_encoding();

        let activation = command.new_compute_command_encoder();
        activation.set_compute_pipeline_state(&self.silu_multiply);
        activation.set_buffer(0, Some(&gate), 0);
        activation.set_buffer(1, Some(&up), 0);
        activation.set_buffer(2, Some(&activated), 0);
        activation.set_bytes(
            3,
            size_of::<u32>() as u64,
            (&intermediate_count as *const u32).cast(),
        );
        activation.dispatch_threads(
            MTLSize::new(u64::from(intermediate_count), 1, 1),
            MTLSize::new(self.silu_multiply.thread_execution_width(), 1, 1),
        );
        activation.end_encoding();

        let down_encoder = command.new_compute_command_encoder();
        self.encode_matrix(down_encoder, &layer_weights.down, &activated, &down)?;
        down_encoder.end_encoding();
        self.encode_add(command, &after_attention, &down, &output, hidden_count);

        buffers.retained.extend([
            attention_scale,
            feed_forward_scale,
            normalized_attention,
            normalized_feed_forward,
            query,
            key,
            value,
            scores,
            attended,
            attention_projection,
            after_attention,
            gate,
            up,
            activated,
            down,
        ]);
        Ok(output)
    }
}
