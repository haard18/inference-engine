use std::env;
use std::path::PathBuf;

use inference_engine::{
    load_gguf, load_gguf_stage, load_safetensors, ActivationFrame, ByteBpeTokenizer,
    GenerationSession, ModelStage, StageSession,
};
use uuid::Uuid;

#[test]
#[ignore = "requires Q4_K_M and Q8_0 SmolLM2 GGUF files in SMOLLM2_DIR"]
fn q4_k_model_stages_own_partial_weights_and_match_full_scores() {
    let directory = PathBuf::from(env::var("SMOLLM2_DIR").expect("set SMOLLM2_DIR"));
    let path = directory.join("SmolLM2-135M-Q4_K_M.gguf");
    let model = load_gguf(&path).expect("load complete model");
    let full_bytes = model.stored_weight_bytes();
    let prefix = load_gguf_stage(&path, 0..15).expect("load prefix stage");
    let suffix = load_gguf_stage(&path, 15..30).expect("load suffix stage");
    ModelStage::validate_pair(&prefix, &suffix).expect("matching stages");
    assert_eq!(prefix.layer_range(), 0..15);
    assert_eq!(suffix.layer_range(), 15..30);
    assert_eq!(prefix.hidden_size(), 576);
    assert!(prefix.stored_weight_bytes() < full_bytes);
    assert!(suffix.stored_weight_bytes() < full_bytes);
    println!(
        "stored weight bytes: full={full_bytes}, prefix={}, suffix={}",
        prefix.stored_weight_bytes(),
        suffix.stored_weight_bytes()
    );

    let mut whole = GenerationSession::new(&model);
    let mut first = StageSession::new(&prefix);
    let mut second = StageSession::new(&suffix);
    let request_id = Uuid::new_v4();
    assert!(first.forward_hidden(vec![0.0; 576]).is_err());
    assert!(second.forward_token(1).is_err());
    assert_eq!(first.position(), 0);
    assert_eq!(second.position(), 0);
    for token in [1, 2, 3, 30] {
        whole.prefill(&[token]).unwrap();
        let hidden = first.forward_token(token).unwrap();
        assert_eq!(hidden.len(), 576);
        let bytes = ActivationFrame::new(&prefix, request_id, second.position(), hidden)
            .unwrap()
            .encode();
        assert_eq!(bytes.len(), 2_368);
        assert!(ActivationFrame::decode_for_stage(
            &bytes,
            &suffix,
            Uuid::new_v4(),
            second.position()
        )
        .is_err());
        assert!(second.forward_frame(&bytes, Uuid::new_v4()).is_err());
        assert_eq!(second.position() + 1, whole.position());
        let actual = second.forward_frame(&bytes, request_id).unwrap();
        assert_eq!(actual, whole.next_token_scores().unwrap());
        assert_eq!(first.position(), whole.position());
        assert_eq!(second.position(), whole.position());
    }
    assert!(second.forward_hidden(vec![0.0; 575]).is_err());
    assert!(second.forward_hidden(vec![f32::NAN; 576]).is_err());
    assert_eq!(second.position(), 4);
    assert!(first.allocated_cache_bytes() > 0);
    assert!(second.allocated_cache_bytes() > 0);
    assert!(load_gguf_stage(&path, 0..30).is_err());
    assert!(load_gguf_stage(&path, 10..20).is_err());
    let wrong_boundary = load_gguf_stage(&path, 14..30).unwrap();
    assert!(ModelStage::validate_pair(&prefix, &wrong_boundary).is_err());
    let wrong_model = load_gguf_stage(directory.join("SmolLM2-135M-Q8_0.gguf"), 15..30)
        .expect("load a different model variant");
    assert!(ModelStage::validate_pair(&prefix, &wrong_model).is_err());
}

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

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Q8_0 and Q4_K_M SmolLM2 GGUF checkpoints in SMOLLM2_DIR"]
fn real_gguf_metal_matches_cpu() {
    let directory = PathBuf::from(env::var("SMOLLM2_DIR").expect("set SMOLLM2_DIR"));
    for filename in ["SmolLM2-135M-Q8_0.gguf", "SmolLM2-135M-Q4_K_M.gguf"] {
        let model = load_gguf(directory.join(filename)).unwrap();
        let mut cpu = GenerationSession::new(&model);
        let mut metal = GenerationSession::on_metal(&model).expect("create Metal session");
        for (position, token) in [1, 2, 3, 30].into_iter().enumerate() {
            cpu.prefill(&[token]).unwrap();
            metal.prefill(&[token]).unwrap();
            let largest_difference = cpu
                .next_token_scores()
                .unwrap()
                .iter()
                .zip(metal.next_token_scores().unwrap())
                .map(|(left, right)| (left - right).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                largest_difference < 1e-3,
                "{filename}, position {position}: score difference {largest_difference}"
            );
        }
        assert_eq!(cpu.next_token().unwrap(), metal.next_token().unwrap());
    }
}
