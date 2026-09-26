use std::env;
use std::path::PathBuf;

use inference_engine::{load_gguf, load_safetensors, ByteBpeTokenizer, GenerationSession};

#[test]
#[ignore = "requires SmolLM2-135M-Q8_0.gguf in SMOLLM2_DIR"]
fn q8_0_smollm2_matches_independent_numpy_reference() {
    let directory = PathBuf::from(env::var("SMOLLM2_DIR").expect("set SMOLLM2_DIR"));
    let model = load_gguf(directory.join("SmolLM2-135M-Q8_0.gguf")).expect("load GGUF");
    assert!(model.stored_weight_bytes() < 160_000_000);
    let mut session = GenerationSession::new(&model);
    session.prefill(&[1, 2, 3]).unwrap();
    let expected = [
        18.387_02,
        1.960_439_2,
        1.950_903_9,
        2.528_184,
        1.833_123_7,
        1.985_401_2,
        6.110_73,
        1.985_502_2,
        1.985_401_2,
        1.985_401_2,
        1.985_401_2,
        1.985_401_2,
    ];
    let scores = session.next_token_scores().unwrap();
    for (index, expected) in expected.into_iter().enumerate() {
        let actual = scores[index];
        assert!(
            (actual - expected).abs() < 1e-3,
            "score {index}: got {actual}, expected {expected}"
        );
    }
    assert_eq!(session.next_token().unwrap(), 30);
}

#[test]
#[ignore = "requires SmolLM2-135M-Q4_K_M.gguf in SMOLLM2_DIR"]
fn q4_k_m_smollm2_matches_independent_numpy_reference() {
    let directory = PathBuf::from(env::var("SMOLLM2_DIR").expect("set SMOLLM2_DIR"));
    let model = load_gguf(directory.join("SmolLM2-135M-Q4_K_M.gguf")).expect("load GGUF");
    assert!(model.stored_weight_bytes() < 105_000_000);
    let mut session = GenerationSession::new(&model);
    session.prefill(&[1, 2, 3]).unwrap();
    let expected = [
        17.496_885,
        1.248_974_8,
        1.396_863,
        1.565_452_1,
        2.047_933,
        1.332_554_3,
        4.086_927,
        1.332_554_3,
        1.332_554_3,
        1.332_554_3,
        1.332_554_3,
        1.332_554_3,
    ];
    let scores = session.next_token_scores().unwrap();
    for (index, expected) in expected.into_iter().enumerate() {
        let actual = scores[index];
        assert!(
            (actual - expected).abs() < 1e-3,
            "score {index}: got {actual}, expected {expected}"
        );
    }
    assert_eq!(session.next_token().unwrap(), 30);
}

#[test]
#[ignore = "requires both SmolLM2 checkpoint formats in SMOLLM2_DIR"]
fn q8_0_and_bf16_generate_same_tokens_for_smoke_prompt() {
    let directory = PathBuf::from(env::var("SMOLLM2_DIR").expect("set SMOLLM2_DIR"));
    let tokenizer = ByteBpeTokenizer::from_file(directory.join("tokenizer.json")).unwrap();
    let prompt: Vec<usize> = tokenizer
        .encode("The capital of France is")
        .unwrap()
        .into_iter()
        .map(|id| id as usize)
        .collect();
    let gguf = load_gguf(directory.join("SmolLM2-135M-Q8_0.gguf")).unwrap();
    let safetensors = load_safetensors(
        directory.join("config.json"),
        directory.join("model.safetensors"),
    )
    .unwrap();
    let mut quantized = GenerationSession::new(&gguf);
    let mut baseline = GenerationSession::new(&safetensors);
    quantized.prefill(&prompt).unwrap();
    baseline.prefill(&prompt).unwrap();
    for step in 0..8 {
        assert_eq!(
            quantized.next_token().unwrap(),
            baseline.next_token().unwrap(),
            "generated token {step}"
        );
    }
}
