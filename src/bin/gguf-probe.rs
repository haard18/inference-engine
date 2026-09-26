use std::env;
use std::error::Error;
use std::process;

use inference_engine::{load_gguf, load_gguf_tokenizer, GenerationSession};

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error + Send + Sync>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        return Err(format!("usage: {} MODEL_GGUF PROMPT GENERATION_COUNT", args[0]).into());
    }
    let tokenizer = load_gguf_tokenizer(&args[1])?;
    let prompt: Vec<usize> = tokenizer
        .encode(&args[2])?
        .into_iter()
        .map(|id| id as usize)
        .collect();
    let model = load_gguf(&args[1])?;
    let count: usize = args[3].parse()?;
    let mut session = GenerationSession::new(&model);
    session.prefill(&prompt)?;
    let mut generated = Vec::with_capacity(count);
    for _ in 0..count {
        generated.push(session.next_token()? as u32);
    }
    println!("{}", tokenizer.decode(&generated, true)?);
    Ok(())
}
