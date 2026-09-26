use half::{bf16, f16};
use inference_engine::{
    EngineError, GenerationSession, LayerWeights, Matrix, Model, ModelConfig, ModelWeights,
};

fn matrix(seed: usize, rows: usize, cols: usize) -> Matrix {
    let values = (0..rows * cols)
        .map(|index| ((index * 17 + seed * 13) % 23) as f32 / 20.0 - 11.0 / 20.0)
        .collect();
    Matrix::new(rows, cols, values).unwrap()
}

fn model(max_positions: usize) -> Model {
    let config = ModelConfig {
        vocab_size: 8,
        hidden_size: 4,
        intermediate_size: 6,
        num_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        max_positions,
        rms_norm_epsilon: 1e-5,
        rope_theta: 10000.0,
        rope_interleaved: false,
    };
    let layers = (0..2)
        .map(|layer| {
            let base = 2 + layer * 7;
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
    let weights = ModelWeights {
        token_embeddings: matrix(1, 8, 4),
        layers,
        final_norm: vec![1.0; 4],
        output: None,
    };
    Model::new(config, weights).unwrap()
}

#[test]
fn tiny_model_matches_independent_numpy_reference() {
    let model = model(8);
    let mut session = GenerationSession::new(&model);
    session.prefill(&[1, 2, 3]).unwrap();
    let expected = [
        -0.87051666,
        -1.016_894_8,
        0.58579636,
        0.43941823,
        0.29304004,
        0.146_662,
        0.00028393,
        -0.14609423,
    ];
    for (actual, expected) in session.next_token_scores().unwrap().iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "got {actual}, expected {expected}"
        );
    }
    assert_eq!(session.position(), 3);
    assert_eq!(session.next_token().unwrap(), 2);
    assert_eq!(session.position(), 4);
}

#[test]
fn invalid_prompt_does_not_change_session() {
    let model = model(8);
    let mut session = GenerationSession::new(&model);
    assert_eq!(session.prefill(&[1, 8]), Err(EngineError::InvalidToken(8)));
    assert_eq!(session.position(), 0);
    assert!(session.next_token_scores().is_none());
}

#[test]
fn full_context_returns_clear_error() {
    let model = model(3);
    let mut session = GenerationSession::new(&model);
    session.prefill(&[1, 2, 3]).unwrap();
    assert_eq!(session.next_token(), Err(EngineError::ContextFull));
}

#[test]
fn mismatched_weights_are_rejected() {
    let config = ModelConfig {
        vocab_size: 8,
        hidden_size: 4,
        intermediate_size: 6,
        num_layers: 1,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        max_positions: 8,
        rms_norm_epsilon: 1e-5,
        rope_theta: 10000.0,
        rope_interleaved: false,
    };
    let weights = ModelWeights {
        token_embeddings: matrix(1, 8, 3),
        layers: vec![],
        final_norm: vec![1.0; 4],
        output: None,
    };
    assert!(matches!(
        Model::new(config, weights),
        Err(EngineError::InvalidShape {
            name: "token embeddings",
            ..
        })
    ));
}

#[test]
fn half_precision_matrices_keep_compact_weights_and_compute_in_f32() {
    let values = [1.0, -2.0, 0.5, 3.0];
    let bf16_bits = values
        .iter()
        .map(|&value| bf16::from_f32(value).to_bits())
        .collect();
    let f16_bits = values
        .iter()
        .map(|&value| f16::from_f32(value).to_bits())
        .collect();
    for matrix in [
        Matrix::from_bf16_bits(2, 2, bf16_bits).unwrap(),
        Matrix::from_f16_bits(2, 2, f16_bits).unwrap(),
    ] {
        assert_eq!(matrix.storage_bytes(), 8);
        assert_eq!(matrix.row(1).unwrap(), [0.5, 3.0]);
        assert_eq!(matrix.mul_vec(&[2.0, 1.0]).unwrap(), [0.0, 4.0]);
    }
    assert_eq!(
        Matrix::from_bf16_bits(1, 1, vec![bf16::NAN.to_bits()]).unwrap_err(),
        EngineError::InvalidValue("matrix")
    );
}

#[test]
fn q8_0_blocks_compute_without_expanding_weights() {
    let mut data = Vec::new();
    data.extend_from_slice(&f16::from_f32(0.5).to_bits().to_le_bytes());
    data.extend_from_slice(&[1; 32]);
    data.extend_from_slice(&f16::from_f32(0.25).to_bits().to_le_bytes());
    data.extend_from_slice(&[255; 32]);
    let matrix = Matrix::from_q8_0(2, 32, data).unwrap();
    assert_eq!(matrix.storage_bytes(), 68);
    assert_eq!(matrix.row(0).unwrap(), [0.5; 32]);
    assert_eq!(matrix.row(1).unwrap(), [-0.25; 32]);
    assert_eq!(matrix.mul_vec(&[1.0; 32]).unwrap(), [16.0, -8.0]);
    assert!(Matrix::from_q8_0(1, 31, vec![]).is_err());
    let mut invalid_scale = vec![0; 34];
    invalid_scale[..2].copy_from_slice(&f16::NAN.to_bits().to_le_bytes());
    assert_eq!(
        Matrix::from_q8_0(1, 32, invalid_scale).unwrap_err(),
        EngineError::InvalidValue("Q8_0 matrix scale")
    );
}

#[test]
fn q5_0_blocks_decode_high_bits_and_compute() {
    let mut block = vec![0_u8; 22];
    block[..2].copy_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
    block[2..6].copy_from_slice(&(1_u32 | (1_u32 << 16)).to_le_bytes());
    block[6] = 0xf0;
    let matrix = Matrix::from_q5_0(1, 32, block).unwrap();
    assert_eq!(matrix.storage_bytes(), 22);
    let row = matrix.row(0).unwrap();
    assert_eq!(row[0], 0.0);
    assert_eq!(row[16], 15.0);
    assert_eq!(row[1], -16.0);
    assert_eq!(matrix.mul_vec(&[1.0; 32]).unwrap(), [-465.0]);
    assert!(Matrix::from_q5_0(1, 31, vec![]).is_err());
}

#[test]
fn q4_k_blocks_decode_subscales_and_compute() {
    let mut block = vec![0_u8; 144];
    block[..2].copy_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
    block[4..8].fill(1);
    block[12..16].fill(1);
    block[16..].fill(0xf0);
    let matrix = Matrix::from_q4_k(1, 256, block).unwrap();
    assert_eq!(matrix.storage_bytes(), 144);
    let row = matrix.row(0).unwrap();
    assert_eq!(&row[..32], &[0.0; 32]);
    assert_eq!(&row[32..64], &[15.0; 32]);
    assert_eq!(matrix.mul_vec(&[1.0; 256]).unwrap(), [1920.0]);
    assert!(Matrix::from_q4_k(1, 255, vec![]).is_err());
}
