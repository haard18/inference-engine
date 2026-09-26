use std::env;
use std::error::Error;
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpListener};
use std::process;
use std::time::Duration;

use inference_engine::load_gguf_tokenizer;
use inference_engine::pool::client::PeerClient;
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
    let mut args: Vec<String> = env::args().collect();
    let stage_metal = args.get(1).is_some_and(|argument| argument == "--metal")
        && args
            .get(2)
            .is_some_and(|argument| argument == "--stage-suffix" || argument == "--split-prefix");
    if stage_metal {
        args.remove(1);
    }
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
    if args
        .get(1)
        .is_some_and(|argument| argument == "--internal-stage-worker")
    {
        if args.len() != 5 && !(args.len() == 6 && args[5] == "--metal") {
            return Err("invalid stage worker invocation".into());
        }
        return serving::run_stage_worker_stdio_with_backend(
            &args[2],
            args[3].parse()?,
            args[4].parse()?,
            if args.len() == 6 {
                ServingBackend::Metal
            } else {
                ServingBackend::Cpu
            },
        );
    }
    if args
        .get(1)
        .is_some_and(|argument| argument == "--stage-suffix")
    {
        if args.len() != 7 {
            return Err(format!(
                "usage: {} [--metal] --stage-suffix STATE_DIR IP:PORT MODEL_GGUF LAYER_START LAYER_END",
                args[0]
            )
            .into());
        }
        let address: SocketAddr = args[3].parse()?;
        if address.port() == 0 {
            return Err("stage peer listener needs a nonzero port".into());
        }
        let identity = DeviceIdentity::load(&args[2])?;
        let peers = PeerStore::load(&args[2])?;
        let listener = TcpListener::bind(address)?;
        let server = serving::start_stage_peer_with_backend(
            &args[4],
            env::current_exe()?,
            args[5].parse()?,
            args[6].parse()?,
            serving::StagePeerOptions {
                queue_capacity: 4,
                backend: if stage_metal {
                    ServingBackend::Metal
                } else {
                    ServingBackend::Cpu
                },
            },
            &identity,
            &peers,
        )
        .await?;
        println!(
            "Serving approved suffix stage at https://{}",
            listener.local_addr()?
        );
        let handle = axum_server::Handle::new();
        let mut running = tokio::spawn(server.serve(listener, handle.clone()));
        tokio::select! {
            result = &mut running => result??,
            _ = tokio::signal::ctrl_c() => {
                handle.graceful_shutdown(Some(Duration::from_secs(5)));
                running.await??;
            }
        }
        return Ok(());
    }
    if args
        .get(1)
        .is_some_and(|argument| argument == "--split-prefix")
    {
        if args.len() < 6 || args.len() > 7 {
            return Err(format!(
                "usage: {} [--metal] --split-prefix STATE_DIR SUFFIX_DEVICE_ID MODEL_GGUF SPLIT_LAYER [LOCAL_PORT]",
                args[0]
            )
            .into());
        }
        let identity = DeviceIdentity::load(&args[2])?;
        let peers = PeerStore::load(&args[2])?;
        let peer = peers
            .peers()
            .iter()
            .find(|peer| peer.device_id == args[3])
            .ok_or("suffix device is not approved")?;
        let client = PeerClient::new(&identity, peer.clone())?;
        let port = args
            .get(6)
            .map(|value| value.parse::<u16>())
            .transpose()?
            .unwrap_or(8080);
        let listener =
            tokio::net::TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, port)).await?;
        let api_key = env::var("INFERENCE_API_KEY")
            .map_err(|_| "set INFERENCE_API_KEY to a secret with at least 32 visible characters")?;
        let config = ServingConfig {
            model_id: "local-smollm2".into(),
            api_key,
            queue_capacity: 4,
            max_completion_tokens: 256,
            request_timeout: Duration::from_secs(120),
            backend: if stage_metal {
                ServingBackend::Metal
            } else {
                ServingBackend::Cpu
            },
        };
        let tokenizer = load_gguf_tokenizer(&args[4])?;
        let (router, worker) = serving::start_split_prefix(
            &args[4],
            tokenizer,
            config,
            env::current_exe()?,
            args[5].parse()?,
            &identity,
            client,
        )
        .await?;
        println!(
            "Serving split local-smollm2 at http://{}",
            listener.local_addr()?
        );
        let mut server = tokio::spawn(async move { axum::serve(listener, router).await });
        tokio::select! {
            result = &mut server => result??,
            _ = tokio::signal::ctrl_c() => server.abort(),
        }
        worker.abort();
        return Ok(());
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
