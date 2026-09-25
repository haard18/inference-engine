use std::env;
use std::error::Error;
use std::process;

use inference_engine::{load_safetensors, ByteBpeTokenizer, GenerationSession};

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error + Send + Sync>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 6 {
        return Err(format!(
            "usage: {} CONFIG_JSON MODEL_SAFETENSORS TOKENIZER_JSON PROMPT GENERATION_COUNT",
            args[0]
        )
        .into());
    }
    let tokenizer = ByteBpeTokenizer::from_file(&args[3])?;
    let prompt_tokens: Vec<usize> = tokenizer
        .encode(&args[4])?
        .into_iter()
        .map(|id| id as usize)
        .collect();
    let model = load_safetensors(&args[1], &args[2])?;
    let count: usize = args[5].parse()?;
    let mut session = GenerationSession::new(&model);
    session.prefill(&prompt_tokens)?;
    let mut generated = Vec::with_capacity(count);
    for _ in 0..count {
        generated.push(session.next_token()? as u32);
    }
    println!("{}", tokenizer.decode(&generated, true)?);
    Ok(())
}
