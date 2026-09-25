use crate::EngineError;

/// A contiguous row-major tensor with two dimensions.
#[derive(Clone, Debug)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Result<Self, EngineError> {
        let expected = rows
            .checked_mul(cols)
            .ok_or(EngineError::InvalidConfig("matrix dimensions overflow"))?;
        if data.len() != expected {
            return Err(EngineError::InvalidShape {
                name: "matrix",
                expected: vec![rows, cols],
                actual: vec![data.len()],
            });
        }
        if !data.iter().all(|value| value.is_finite()) {
            return Err(EngineError::InvalidValue("matrix"));
        }
        Ok(Self { rows, cols, data })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn row(&self, index: usize) -> Result<&[f32], EngineError> {
        if index >= self.rows {
            return Err(EngineError::InvalidToken(index));
        }
        let start = index * self.cols;
        Ok(&self.data[start..start + self.cols])
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
        for (row, output) in result.iter_mut().enumerate() {
            let start = row * self.cols;
            *output = self.data[start..start + self.cols]
                .iter()
                .zip(input)
                .map(|(weight, value)| weight * value)
                .sum();
        }
        Ok(result)
    }
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
