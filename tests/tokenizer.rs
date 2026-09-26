use std::env;
use std::path::PathBuf;

use inference_engine::{load_gguf_tokenizer, ByteBpeDecoder, ByteBpeTokenizer};
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

#[test]
#[ignore = "requires SmolLM2-135M-Q8_0.gguf and tokenizer.json in SMOLLM2_DIR"]
fn smollm2_gguf_tokenizer_matches_json_and_reference() {
    let directory = PathBuf::from(env::var("SMOLLM2_DIR").expect("set SMOLLM2_DIR"));
    let json_path = directory.join("tokenizer.json");
    let gguf =
        load_gguf_tokenizer(directory.join("SmolLM2-135M-Q8_0.gguf")).expect("load GGUF tokenizer");
    let json = ByteBpeTokenizer::from_file(&json_path).expect("load JSON tokenizer");
    let reference = Tokenizer::from_file(&json_path).expect("load reference tokenizer");
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
        let expected = json.encode(prompt).expect("encode with JSON");
        assert_eq!(gguf.encode(prompt).expect("encode with GGUF"), expected);
        compare(&gguf, &reference, prompt);
        assert_eq!(
            gguf.decode(&expected, false).unwrap(),
            json.decode(&expected, false).unwrap()
        );
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
        assert_eq!(gguf.encode(&prompt).unwrap(), json.encode(&prompt).unwrap());
        compare(&gguf, &reference, &prompt);
    }
}

#[test]
#[ignore = "requires SmolLM2-135M tokenizer.json in SMOLLM2_DIR"]
fn streaming_decode_waits_for_complete_utf8_and_keeps_user_text_plain() {
    let directory = PathBuf::from(env::var("SMOLLM2_DIR").expect("set SMOLLM2_DIR"));
    let tokenizer = ByteBpeTokenizer::from_file(directory.join("tokenizer.json")).unwrap();
    let marker = "<|im_start|>";
    let special = tokenizer.special_token_id(marker).unwrap();
    assert_eq!(tokenizer.encode(marker).unwrap(), [special]);
    assert!(!tokenizer
        .encode_plain_text(marker)
        .unwrap()
        .contains(&special));
    for text in ["é café 東京 😀", "hello\nworld", "a<|im_start|>b"] {
        let ids = tokenizer.encode_plain_text(text).unwrap();
        let mut decoder = ByteBpeDecoder::new();
        let mut output = String::new();
        for id in ids {
            output.push_str(&decoder.push(&tokenizer, id).unwrap());
            assert!(!output.contains('\u{fffd}'));
        }
        output.push_str(&decoder.finish());
        assert_eq!(output, text);
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
