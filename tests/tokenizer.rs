use std::env;
use std::path::PathBuf;

use inference_engine::ByteBpeTokenizer;
use tokenizers::Tokenizer;

#[test]
#[ignore = "requires the SmolLM2-135M tokenizer in SMOLLM2_DIR"]
fn smollm2_tokenizer_matches_reference() {
    let path =
        PathBuf::from(env::var("SMOLLM2_DIR").expect("set SMOLLM2_DIR")).join("tokenizer.json");
    let ours = ByteBpeTokenizer::from_file(&path).expect("load tokenizer");
    let reference = Tokenizer::from_file(&path).expect("load reference tokenizer");
    let prompts = [
        "The capital of France is",
        "Hello, world!",
        "  whitespace\n\tand  two spaces ",
        "abc123def 2026-09-26",
        "é café 東京 😀",
        "<|im_start|>user\nHello<|im_end|>",
        "a<|endoftext|>b",
        "don't we'll I've",
    ];
    for prompt in prompts {
        compare(&ours, &reference, prompt);
    }
    let fragments = [
        "hello",
        " ",
        "  ",
        "\n",
        "42",
        "élan",
        "東京",
        "!",
        "<|im_end|>",
    ];
    for case in 0..100 {
        let prompt = (0..12)
            .map(|index| fragments[(case * 7 + index * 11 + index * case) % fragments.len()])
            .collect::<String>();
        compare(&ours, &reference, &prompt);
    }
}

fn compare(ours: &ByteBpeTokenizer, reference: &Tokenizer, prompt: &str) {
    let actual = ours.encode(prompt).expect("encode prompt");
    let expected = reference
        .encode(prompt, true)
        .expect("reference encode")
        .get_ids()
        .to_vec();
    assert_eq!(actual, expected, "prompt: {prompt:?}");
    assert_eq!(
        ours.decode(&actual, false).expect("decode prompt"),
        reference
            .decode(&expected, false)
            .expect("reference decode"),
        "decoded prompt: {prompt:?}"
    );
}
