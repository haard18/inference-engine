use std::env;
use std::path::PathBuf;

use inference_engine::{load_safetensors, GenerationSession};

#[test]
#[ignore = "requires the SmolLM2-135M checkpoint in SMOLLM2_DIR"]
fn smollm2_matches_independent_numpy_reference() {
    let directory = PathBuf::from(env::var("SMOLLM2_DIR").expect("set SMOLLM2_DIR"));
    let model = load_safetensors(
        directory.join("config.json"),
        directory.join("model.safetensors"),
    )
    .expect("load SmolLM2-135M");
    let mut session = GenerationSession::new(&model);
    session.prefill(&[1, 2, 3]).unwrap();
    let expected = [
        19.008_14,
        3.003_776_6,
        3.109_168,
        3.660_808,
        3.000_863_6,
        3.145_187,
        7.877_74,
        3.145_504,
        3.145_446_8,
        3.145_187,
        3.145_506_9,
        3.145_294_7,
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
