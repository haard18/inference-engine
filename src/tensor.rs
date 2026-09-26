use crate::EngineError;
use half::{bf16, f16};
use std::mem::size_of;

#[derive(Clone, Debug)]
enum MatrixData {
    F32(Vec<f32>),
    F16(Vec<u16>),
    Bf16(Vec<u16>),
    Q8_0(Vec<u8>),
}

/// A contiguous row-major tensor with two dimensions.
#[derive(Clone, Debug)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: MatrixData,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Result<Self, EngineError> {
        validate_length(rows, cols, data.len())?;
        if !data.iter().all(|value| value.is_finite()) {
            return Err(EngineError::InvalidValue("matrix"));
        }
        Ok(Self {
            rows,
            cols,
            data: MatrixData::F32(data),
        })
    }

    pub fn from_f16_bits(rows: usize, cols: usize, data: Vec<u16>) -> Result<Self, EngineError> {
        validate_length(rows, cols, data.len())?;
        if !data.iter().all(|&bits| f16::from_bits(bits).is_finite()) {
            return Err(EngineError::InvalidValue("matrix"));
        }
        Ok(Self {
            rows,
            cols,
            data: MatrixData::F16(data),
        })
    }

    pub fn from_bf16_bits(rows: usize, cols: usize, data: Vec<u16>) -> Result<Self, EngineError> {
        validate_length(rows, cols, data.len())?;
        if !data.iter().all(|&bits| bf16::from_bits(bits).is_finite()) {
            return Err(EngineError::InvalidValue("matrix"));
        }
        Ok(Self {
            rows,
            cols,
            data: MatrixData::Bf16(data),
        })
    }

    /// Q8_0 stores 32 signed weights behind one f16 scale in each 34-byte block.
    pub fn from_q8_0(rows: usize, cols: usize, data: Vec<u8>) -> Result<Self, EngineError> {
        if !cols.is_multiple_of(32) {
            return Err(EngineError::InvalidConfig(
                "Q8_0 matrix width must be a multiple of 32",
            ));
        }
        let blocks = rows
            .checked_mul(cols / 32)
            .ok_or(EngineError::InvalidConfig("matrix dimensions overflow"))?;
        let expected = blocks
            .checked_mul(34)
            .ok_or(EngineError::InvalidConfig("Q8_0 matrix size overflow"))?;
        if data.len() != expected {
            return Err(EngineError::InvalidShape {
                name: "Q8_0 matrix",
                expected: vec![expected],
                actual: vec![data.len()],
            });
        }
        if !data
            .as_chunks::<34>()
            .0
            .iter()
            .all(|block| f16::from_bits(u16::from_le_bytes([block[0], block[1]])).is_finite())
        {
            return Err(EngineError::InvalidValue("Q8_0 matrix scale"));
        }
        Ok(Self {
            rows,
            cols,
            data: MatrixData::Q8_0(data),
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Bytes used by the matrix values, excluding Vec capacity and metadata.
    pub fn storage_bytes(&self) -> usize {
        match &self.data {
            MatrixData::F32(values) => values.len() * size_of::<f32>(),
            MatrixData::F16(values) | MatrixData::Bf16(values) => values.len() * size_of::<u16>(),
            MatrixData::Q8_0(values) => values.len(),
        }
    }

    pub fn row(&self, index: usize) -> Result<Vec<f32>, EngineError> {
        if index >= self.rows {
            return Err(EngineError::InvalidToken(index));
        }
        let start = index * self.cols;
        Ok(match &self.data {
            MatrixData::F32(values) => values[start..start + self.cols].to_vec(),
            MatrixData::F16(values) => values[start..start + self.cols]
                .iter()
                .map(|&bits| f16::from_bits(bits).to_f32())
                .collect(),
            MatrixData::Bf16(values) => values[start..start + self.cols]
                .iter()
                .map(|&bits| bf16::from_bits(bits).to_f32())
                .collect(),
            MatrixData::Q8_0(values) => {
                let row_bytes = self.cols / 32 * 34;
                values[index * row_bytes..(index + 1) * row_bytes]
                    .as_chunks::<34>()
                    .0
                    .iter()
                    .flat_map(|block| {
                        let scale =
                            f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
                        block[2..]
                            .iter()
                            .map(move |&quant| scale * (quant as i8 as f32))
                    })
                    .collect()
            }
        })
    }

    pub fn mul_vec(&self, input: &[f32]) -> Result<Vec<f32>, EngineError> {
        if input.len() != self.cols {
            return Err(EngineError::InvalidShape {
                name: "matrix input",
                expected: vec![self.cols],
                actual: vec![input.len()],
            });
        }
        let mut result = vec![0.0; self.rows];
        if self.cols == 0 {
            return Ok(result);
        }
        match &self.data {
            MatrixData::F32(values) => {
                for (output, row) in result.iter_mut().zip(values.chunks_exact(self.cols)) {
                    *output = row
                        .iter()
                        .zip(input)
                        .map(|(weight, value)| weight * value)
                        .sum();
                }
            }
            MatrixData::F16(values) => {
                for (output, row) in result.iter_mut().zip(values.chunks_exact(self.cols)) {
                    *output = row
                        .iter()
                        .zip(input)
                        .map(|(&bits, value)| f16::from_bits(bits).to_f32() * value)
                        .sum();
                }
            }
            MatrixData::Bf16(values) => {
                for (output, row) in result.iter_mut().zip(values.chunks_exact(self.cols)) {
                    *output = row
                        .iter()
                        .zip(input)
                        .map(|(&bits, value)| bf16::from_bits(bits).to_f32() * value)
                        .sum();
                }
            }
            MatrixData::Q8_0(values) => {
                let row_bytes = self.cols / 32 * 34;
                for (output, row) in result.iter_mut().zip(values.chunks_exact(row_bytes)) {
                    for (block, input_block) in row
                        .as_chunks::<34>()
                        .0
                        .iter()
                        .zip(input.as_chunks::<32>().0)
                    {
                        let scale =
                            f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
                        let dot: f32 = block[2..]
                            .iter()
                            .zip(input_block)
                            .map(|(&quant, value)| (quant as i8 as f32) * value)
                            .sum();
                        *output += scale * dot;
                    }
                }
            }
        }
        Ok(result)
    }
}

fn validate_length(rows: usize, cols: usize, actual: usize) -> Result<(), EngineError> {
    let expected = rows
        .checked_mul(cols)
        .ok_or(EngineError::InvalidConfig("matrix dimensions overflow"))?;
    if actual != expected {
        return Err(EngineError::InvalidShape {
            name: "matrix",
            expected: vec![rows, cols],
            actual: vec![actual],
        });
    }
    Ok(())
}

pub(crate) fn rms_norm(
    input: &[f32],
    weights: &[f32],
    epsilon: f32,
) -> Result<Vec<f32>, EngineError> {
    if input.len() != weights.len() {
        return Err(EngineError::InvalidShape {
            name: "RMS normalization weights",
            expected: vec![input.len()],
            actual: vec![weights.len()],
        });
    }
    let mean_square = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let scale = (mean_square + epsilon).sqrt().recip();
    Ok(input
        .iter()
        .zip(weights)
        .map(|(value, weight)| value * scale * weight)
        .collect())
}

pub(crate) fn softmax(input: &[f32]) -> Vec<f32> {
    let maximum = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exponentials: Vec<f32> = input.iter().map(|value| (value - maximum).exp()).collect();
    let total: f32 = exponentials.iter().sum();
    exponentials
        .into_iter()
        .map(|value| value / total)
        .collect()
}

pub(crate) fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

pub(crate) fn apply_rope(head: &mut [f32], position: usize, theta: f32) {
    let half = head.len() / 2;
    for index in 0..half {
        let frequency = theta.powf(-((2 * index) as f32) / head.len() as f32);
        let angle = position as f32 * frequency;
        let (sine, cosine) = angle.sin_cos();
        let first = head[index];
        let second = head[index + half];
        head[index] = first * cosine - second * sine;
        head[index + half] = second * cosine + first * sine;
    }
}

pub(crate) fn apply_rope_interleaved(head: &mut [f32], position: usize, theta: f32) {
    let width = head.len() as f32;
    for (index, pair) in head.as_chunks_mut::<2>().0.iter_mut().enumerate() {
        let frequency = theta.powf(-((2 * index) as f32) / width);
        let (sine, cosine) = (position as f32 * frequency).sin_cos();
        let first = pair[0];
        let second = pair[1];
        pair[0] = first * cosine - second * sine;
        pair[1] = second * cosine + first * sine;
    }
}
