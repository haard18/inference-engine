use std::env;
use std::error::Error;
use std::process;
use std::time::Instant;

use inference_engine::{load_gguf, load_gguf_tokenizer, GenerationSession};
use serde_json::json;

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error + Send + Sync>> {
    let args: Vec<String> = env::args().collect();
    let metal = args.get(1).is_some_and(|option| option == "--metal");
    let offset = usize::from(metal);
    let timings = args.last().is_some_and(|option| option == "--timings");
    if args.len() != 4 + offset + usize::from(timings) {
        return Err(format!(
            "usage: {} [--metal] MODEL_GGUF PROMPT GENERATION_COUNT [--timings]",
            args[0]
        )
        .into());
    }
    let tokenizer = load_gguf_tokenizer(&args[1 + offset])?;
    let prompt: Vec<usize> = tokenizer
        .encode(&args[2 + offset])?
        .into_iter()
        .map(|id| id as usize)
        .collect();
    let model = load_gguf(&args[1 + offset])?;
    let count: usize = args[3 + offset].parse()?;
    let mut session = if metal {
        GenerationSession::on_metal(&model)?
    } else {
        GenerationSession::new(&model)
    };
    let prefill_started = Instant::now();
    session.prefill(&prompt)?;
    let prefill_elapsed = prefill_started.elapsed();
    let mut generated = Vec::with_capacity(count);
    let generation_started = Instant::now();
    for _ in 0..count {
        generated.push(session.next_token()? as u32);
    }
    let generation_elapsed = generation_started.elapsed();
    println!("{}", tokenizer.decode(&generated, true)?);
    if timings {
        eprintln!(
            "{}",
            json!({
                "prompt_tokens": prompt.len(),
                "generated_tokens": count,
                "prefill_ms": prefill_elapsed.as_secs_f64() * 1000.0,
                "generation_ms": generation_elapsed.as_secs_f64() * 1000.0
            })
        );
    }
    Ok(())
}
