use std::collections::HashMap;
use std::mem::size_of;
use std::sync::{Arc, Mutex};

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};

use crate::{EngineError, GenerationSession, Matrix, Model};

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
}

pub(crate) struct MetalBackend<'a> {
    _model: &'a Model,
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    attention_scores: ComputePipelineState,
    attention_reduce: ComputePipelineState,
    weights: HashMap<usize, (Buffer, u32)>,
}

/// A prepared Mac GPU runtime. Multiple generation sessions reuse its uploaded weights.
pub struct MetalRuntime<'a> {
    model: &'a Model,
    backend: Arc<Mutex<MetalBackend<'a>>>,
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

impl<'a> MetalBackend<'a> {
    pub(crate) fn new(model: &'a Model) -> Result<Self, EngineError> {
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
        let reduce_function = library
            .get_function("attention_reduce", None)
            .map_err(|error| EngineError::Backend(format!("Metal function: {error}")))?;
        let attention_reduce = device
            .new_compute_pipeline_state_with_function(&reduce_function)
            .map_err(|error| EngineError::Backend(format!("Metal attention pipeline: {error}")))?;
        let queue = device.new_command_queue();
        let mut weights = HashMap::new();
        for matrix in model.matrices() {
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
        Ok(Self {
            _model: model,
            device,
            queue,
            pipeline,
            attention_scores,
            attention_reduce,
            weights,
        })
    }

    pub(crate) fn attend(
        &mut self,
        cache: &mut MetalKvCache,
        layer_index: usize,
        position: usize,
        head_count: usize,
        kv_head_count: usize,
        vectors: (&[f32], &[f32], &[f32]),
    ) -> Result<Vec<f32>, EngineError> {
        let (query, key, value) = vectors;
        if head_count == 0 {
            return Err(EngineError::InvalidConfig("Metal attention dimensions"));
        }
        let head_size = query.len() / head_count;
        if position >= cache.max_positions
            || kv_head_count == 0
            || !head_count.is_multiple_of(kv_head_count)
            || query.len() != head_count * head_size
            || head_size == 0
            || key.len() != cache.kv_size
            || value.len() != cache.kv_size
            || cache.kv_size != kv_head_count * head_size
        {
            return Err(EngineError::InvalidConfig("Metal attention dimensions"));
        }
        let layer = cache
            .layers
            .get_mut(layer_index)
            .ok_or(EngineError::InvalidConfig("Metal attention layer"))?;
        if layer.capacity <= position {
            let mut capacity = layer.capacity.max(16).min(cache.max_positions);
            while capacity <= position {
                capacity = capacity.saturating_mul(2).min(cache.max_positions);
            }
            let bytes = capacity
                .checked_mul(cache.kv_size)
                .and_then(|values| values.checked_mul(size_of::<f32>()))
                .ok_or_else(|| EngineError::Backend("Metal cache size overflow".into()))?;
            if bytes as u64 > self.device.max_buffer_length() {
                return Err(EngineError::Backend("Metal cache buffer limit".into()));
            }
            let keys = self
                .device
                .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
            let values = self
                .device
                .new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
            let old_bytes = position * cache.kv_size * size_of::<f32>();
            if let (Some(old_keys), Some(old_values)) = (&layer.keys, &layer.values) {
                // All prior commands have completed before a cache buffer grows.
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
        let keys = layer.keys.as_ref().expect("cache keys allocated");
        let values = layer.values.as_ref().expect("cache values allocated");
        let slot = position * cache.kv_size;
        // Shared Metal buffers are CPU writable before submitting the command.
        unsafe {
            std::ptr::copy_nonoverlapping(
                key.as_ptr(),
                keys.contents().cast::<f32>().add(slot),
                cache.kv_size,
            );
            std::ptr::copy_nonoverlapping(
                value.as_ptr(),
                values.contents().cast::<f32>().add(slot),
                cache.kv_size,
            );
        }
        let sequence_length = position + 1;
        let params = AttentionParams {
            head_count: u32::try_from(head_count)
                .map_err(|_| EngineError::Backend("too many attention heads".into()))?,
            kv_head_count: u32::try_from(kv_head_count)
                .map_err(|_| EngineError::Backend("too many key/value heads".into()))?,
            head_size: u32::try_from(head_size)
                .map_err(|_| EngineError::Backend("attention head too wide".into()))?,
            kv_size: u32::try_from(cache.kv_size)
                .map_err(|_| EngineError::Backend("key/value state too wide".into()))?,
            sequence_length: u32::try_from(sequence_length)
                .map_err(|_| EngineError::Backend("context too long for Metal".into()))?,
        };
        let score_count = head_count
            .checked_mul(sequence_length)
            .ok_or_else(|| EngineError::Backend("Metal score size overflow".into()))?;
        u32::try_from(score_count)
            .map_err(|_| EngineError::Backend("too many Metal attention scores".into()))?;
        u32::try_from(query.len())
            .map_err(|_| EngineError::Backend("Metal attention output too wide".into()))?;
        let score_bytes = score_count
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| EngineError::Backend("Metal score size overflow".into()))?;
        let output_bytes = size_of_val(query);
        if score_bytes as u64 > self.device.max_buffer_length()
            || output_bytes as u64 > self.device.max_buffer_length()
        {
            return Err(EngineError::Backend("Metal attention buffer limit".into()));
        }
        let query_buffer = self.device.new_buffer_with_data(
            query.as_ptr().cast(),
            output_bytes as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let scores = self
            .device
            .new_buffer(score_bytes as u64, MTLResourceOptions::StorageModeShared);
        let output = self
            .device
            .new_buffer(output_bytes as u64, MTLResourceOptions::StorageModeShared);
        let command = self.queue.new_command_buffer();
        let score_encoder = command.new_compute_command_encoder();
        score_encoder.set_compute_pipeline_state(&self.attention_scores);
        score_encoder.set_buffer(0, Some(&query_buffer), 0);
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
        reduce_encoder.set_buffer(2, Some(&output), 0);
        reduce_encoder.set_bytes(
            3,
            size_of::<AttentionParams>() as u64,
            (&params as *const AttentionParams).cast(),
        );
        reduce_encoder.dispatch_threads(
            MTLSize::new(query.len() as u64, 1, 1),
            MTLSize::new(self.attention_reduce.thread_execution_width(), 1, 1),
        );
        reduce_encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(EngineError::Backend(format!(
                "Metal attention command ended with {:?}",
                command.status()
            )));
        }
        Ok(unsafe {
            std::slice::from_raw_parts(output.contents().cast::<f32>(), query.len()).to_vec()
        })
    }

    pub(crate) fn mul_vec_many(
        &mut self,
        matrices: &[&Matrix],
        input: &[f32],
    ) -> Result<Vec<Vec<f32>>, EngineError> {
        if matrices.is_empty() {
            return Ok(Vec::new());
        }
        let mut dispatches = Vec::with_capacity(matrices.len());
        for &matrix in matrices {
            if input.len() != matrix.cols() {
                return Err(EngineError::InvalidShape {
                    name: "matrix input",
                    expected: vec![matrix.cols()],
                    actual: vec![input.len()],
                });
            }
            let (weight, kind) = self
                .weights
                .get(&(matrix as *const Matrix as usize))
                .ok_or_else(|| EngineError::Backend("matrix was not uploaded to Metal".into()))?;
            let rows = u32::try_from(matrix.rows())
                .map_err(|_| EngineError::Backend("matrix has too many rows for Metal".into()))?;
            let cols = u32::try_from(matrix.cols()).map_err(|_| {
                EngineError::Backend("matrix has too many columns for Metal".into())
            })?;
            let output_bytes =
                matrix.rows().checked_mul(size_of::<f32>()).ok_or_else(|| {
                    EngineError::Backend("Metal output buffer size overflow".into())
                })? as u64;
            if output_bytes > self.device.max_buffer_length() {
                return Err(EngineError::Backend("Metal output buffer limit".into()));
            }
            dispatches.push((
                weight,
                Params {
                    rows,
                    cols,
                    kind: *kind,
                    reserved: 0,
                },
                output_bytes,
            ));
        }
        if size_of_val(input) as u64 > self.device.max_buffer_length() {
            return Err(EngineError::Backend("Metal input buffer limit".into()));
        }
        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr().cast(),
            size_of_val(input) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let command = self.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        let threads = self.pipeline.thread_execution_width();
        let mut outputs = Vec::with_capacity(matrices.len());
        for (weight, params, output_bytes) in dispatches {
            let output_buffer = self
                .device
                .new_buffer(output_bytes, MTLResourceOptions::StorageModeShared);
            encoder.set_buffer(0, Some(weight), 0);
            encoder.set_buffer(2, Some(&output_buffer), 0);
            encoder.set_bytes(
                3,
                size_of::<Params>() as u64,
                (&params as *const Params).cast(),
            );
            encoder.dispatch_threads(
                MTLSize::new(u64::from(params.rows), 1, 1),
                MTLSize::new(threads, 1, 1),
            );
            outputs.push(output_buffer);
        }
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(EngineError::Backend(format!(
                "Metal command ended with {:?}",
                command.status()
            )));
        }
        // Shared output buffers are readable after the command completes.
        Ok(outputs
            .iter()
            .zip(matrices)
            .map(|(buffer, matrix)| unsafe {
                std::slice::from_raw_parts(
                    buffer.contents().cast::<f32>() as *const f32,
                    matrix.rows(),
                )
                .to_vec()
            })
            .collect())
    }
}
