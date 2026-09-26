use std::env;
use std::error::Error;
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpListener};
use std::process;
use std::time::Duration;

use inference_engine::load_gguf_tokenizer;
use inference_engine::pool::{DeviceIdentity, PeerStore};
use inference_engine::serving::{self, ServingBackend, ServingConfig};
use tokio::task::JoinSet;

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
    let mut index = 1;
    let metal = args
        .get(index)
        .is_some_and(|argument| argument == "--metal");
    index += usize::from(metal);
    let peer = if args.get(index).is_some_and(|argument| argument == "--peer") {
        let state_dir = args
            .get(index + 1)
            .ok_or("--peer requires STATE_DIR and IP:PORT")?;
        let address: SocketAddr = args
            .get(index + 2)
            .ok_or("--peer requires STATE_DIR and IP:PORT")?
            .parse()?;
        if address.port() == 0 {
            return Err("peer listener needs a nonzero port".into());
        }
        index += 3;
        Some((state_dir, address))
    } else {
        None
    };
    if args.len() < index + 1 || args.len() > index + 2 {
        return Err(format!(
            "usage: {} [--metal] [--peer STATE_DIR IP:PORT] MODEL_GGUF [LOCAL_PORT]",
            args[0]
        )
        .into());
    }
    let port = args
        .get(index + 1)
        .map(|value| value.parse::<u16>())
        .transpose()?
        .unwrap_or(8080);
    let path = &args[index];
    let api_key = env::var("INFERENCE_API_KEY")
        .map_err(|_| "set INFERENCE_API_KEY to a secret with at least 32 visible characters")?;
    let local_listener =
        tokio::net::TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, port)).await?;
    let paired = if let Some((state_dir, address)) = peer {
        let identity = DeviceIdentity::load(state_dir)?;
        let peers = PeerStore::load(state_dir)?;
        let listener = TcpListener::bind(address)?;
        Some((identity, peers, listener))
    } else {
        None
    };
    let tokenizer = load_gguf_tokenizer(path)?;
    let config = ServingConfig {
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
    };
    let executable = env::current_exe()?;
    let (router, peer_server, worker) = if let Some((ref identity, ref peers, _)) = paired {
        let (router, peer_server, worker) =
            serving::start_isolated_paired(path, tokenizer, config, &executable, identity, peers)
                .await?;
        (router, Some(peer_server), worker)
    } else {
        let (router, worker) =
            serving::start_isolated(path, tokenizer, config, &executable).await?;
        (router, None, worker)
    };
    println!(
        "Serving local-smollm2 at http://{}",
        local_listener.local_addr()?
    );
    let (stop, mut stopped) = tokio::sync::watch::channel(false);
    let mut services = JoinSet::new();
    services.spawn(async move {
        axum::serve(local_listener, router)
            .with_graceful_shutdown(async move {
                let _ = stopped.changed().await;
            })
            .await
    });
    let peer_handle = axum_server::Handle::new();
    if let (Some(server), Some((_, _, listener))) = (peer_server, paired) {
        println!(
            "Serving approved peers at https://{}",
            listener.local_addr()?
        );
        let handle = peer_handle.clone();
        services.spawn(async move { server.serve(listener, handle).await });
    }
    let early_exit = tokio::select! {
        _ = tokio::signal::ctrl_c() => None,
        result = services.join_next() => result,
    };
    let stopped_early = early_exit.is_some();
    let _ = stop.send(true);
    peer_handle.graceful_shutdown(Some(Duration::from_secs(5)));
    if let Some(result) = early_exit {
        result??;
    }
    while let Some(result) = services.join_next().await {
        result??;
    }
    worker
        .await
        .map_err(|_| "inference supervisor panicked during shutdown")?;
    if stopped_early {
        return Err("a serving listener stopped unexpectedly".into());
    }
    Ok(())
}
