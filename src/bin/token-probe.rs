use std::env;
use std::error::Error;
use std::process;

use inference_engine::{load_safetensors, GenerationSession};

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 5 {
        return Err(format!(
            "usage: {} CONFIG_JSON MODEL_SAFETENSORS TOKEN_IDS GENERATION_COUNT\n\
             example TOKEN_IDS: 0,12,42",
            args[0]
        )
        .into());
    }
    let model = load_safetensors(&args[1], &args[2])?;
    let tokens: Vec<usize> = args[3]
        .split(',')
        .map(str::parse::<usize>)
        .collect::<Result<_, _>>()?;
    let count: usize = args[4].parse()?;
    let mut session = GenerationSession::new(&model);
    session.prefill(&tokens)?;
    for _ in 0..count {
        let token = session.next_token()?;
        println!("{token}");
    }
    Ok(())
}
