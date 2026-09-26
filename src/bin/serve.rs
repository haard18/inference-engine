use std::env;
use std::error::Error;
use std::net::{Ipv4Addr, SocketAddrV4};
use std::process;
use std::time::Duration;

use inference_engine::load_gguf_tokenizer;
use inference_engine::serving::{self, ServingBackend, ServingConfig};

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("{error}");
        process::exit(1);
    }
}

async fn run() -> Result<(), Box<dyn Error + Send + Sync>> {
    let args: Vec<String> = env::args().collect();
    if args
        .get(1)
        .is_some_and(|argument| argument == "--internal-worker")
    {
        if args.len() < 3 || args.len() > 4 || (args.len() == 4 && args[3] != "--metal") {
            return Err("invalid worker invocation".into());
        }
        return serving::run_worker_stdio(
            &args[2],
            if args.len() == 4 {
                ServingBackend::Metal
            } else {
                ServingBackend::Cpu
            },
        );
    }
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
    let (router, worker) = serving::start_isolated(
        path,
        tokenizer,
        ServingConfig {
            model_id: "local-smollm2".into(),
            api_key,
            queue_capacity: 4,
            max_completion_tokens: 256,
            request_timeout: Duration::from_secs(120),
            backend: if metal {
                ServingBackend::Metal
            } else {
                ServingBackend::Cpu
            },
        },
        env::current_exe()?,
    )
    .await?;
    let listener =
        tokio::net::TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, port)).await?;
    println!("Serving local-smollm2 at http://{}", listener.local_addr()?);
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    worker
        .await
        .map_err(|_| "inference supervisor panicked during shutdown")?;
    Ok(())
}
