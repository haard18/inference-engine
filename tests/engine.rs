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
