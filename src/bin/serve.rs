use std::env;
use std::error::Error;
use std::net::{Ipv4Addr, SocketAddrV4};
use std::process;

use inference_engine::serving::{self, ServingBackend, ServingConfig};
use inference_engine::{load_gguf, load_gguf_tokenizer};

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("{error}");
        process::exit(1);
    }
}

async fn run() -> Result<(), Box<dyn Error + Send + Sync>> {
    let args: Vec<String> = env::args().collect();
    let metal = args.get(1).is_some_and(|argument| argument == "--metal");
    let offset = usize::from(metal);
    if args.len() < 2 + offset || args.len() > 3 + offset {
        return Err(format!("usage: {} [--metal] MODEL_GGUF [PORT]", args[0]).into());
    }
    let port = args
        .get(2 + offset)
        .map(|value| value.parse::<u16>())
        .transpose()?
        .unwrap_or(8080);
    let path = &args[1 + offset];
    let api_key = env::var("INFERENCE_API_KEY")
        .map_err(|_| "set INFERENCE_API_KEY to a secret with at least 32 visible characters")?;
    let tokenizer = load_gguf_tokenizer(path)?;
    let model = load_gguf(path)?;
    let (router, worker) = serving::start(
        model,
        tokenizer,
        ServingConfig {
            model_id: "local-smollm2".into(),
            api_key,
            queue_capacity: 4,
            max_completion_tokens: 256,
            backend: if metal {
                ServingBackend::Metal
            } else {
                ServingBackend::Cpu
            },
        },
    )?;
    let listener =
        tokio::net::TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, port)).await?;
    println!("Serving local-smollm2 at http://{}", listener.local_addr()?);
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    tokio::task::spawn_blocking(move || worker.join())
        .await?
        .map_err(|_| "inference worker panicked during shutdown")?;
    Ok(())
}
