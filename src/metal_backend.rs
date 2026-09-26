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

pub(crate) struct MetalBackend<'a> {
    _model: &'a Model,
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
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
            weights,
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
