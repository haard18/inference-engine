use std::env;
use std::error::Error;
use std::fs;
use std::net::SocketAddr;
use std::process;

use inference_engine::pool::{DeviceIdentity, PairingOffer, PeerStore};

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error + Send + Sync>> {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("init") if args.len() == 3 => {
            let identity = DeviceIdentity::load_or_create(&args[2])?;
            print_identity(&identity);
        }
        Some("show") if args.len() == 3 => {
            let identity = DeviceIdentity::load(&args[2])?;
            print_identity(&identity);
        }
        Some("offer") if args.len() == 3 => {
            let identity = DeviceIdentity::load(&args[2])?;
            println!("{}", serde_json::to_string_pretty(&identity.offer())?);
        }
        Some("trust") if args.len() == 6 => {
            let identity = DeviceIdentity::load(&args[2])?;
            let metadata = fs::metadata(&args[3])?;
            if metadata.len() > 16 * 1024 {
                return Err("peer offer exceeds 16 KiB".into());
            }
            let offer: PairingOffer = serde_json::from_slice(&fs::read(&args[3])?)?;
            let address: SocketAddr = args[4].parse()?;
            let mut peers = PeerStore::load(&args[2])?;
            peers.trust(&identity.device_id, &offer, address, &args[5])?;
            println!("Trusted device {} at {}", offer.device_id, address);
        }
        Some("peers") if args.len() == 3 => {
            let peers = PeerStore::load(&args[2])?;
            for peer in peers.peers() {
                println!("{} {} {}", peer.device_id, peer.address, peer.fingerprint);
            }
        }
        Some("remove") if args.len() == 4 => {
            let mut peers = PeerStore::load(&args[2])?;
            if !peers.remove(&args[3])? {
                return Err("peer is not trusted".into());
            }
            println!("Removed device {}", args[3]);
        }
        _ => {
            return Err(format!(
                "usage: {} init|show|offer|peers STATE_DIR | trust STATE_DIR OFFER_FILE IP:PORT FINGERPRINT | remove STATE_DIR DEVICE_ID",
                args[0]
            )
            .into());
        }
    }
    Ok(())
}

fn print_identity(identity: &DeviceIdentity) {
    println!("Device ID: {}", identity.device_id);
    println!("Certificate fingerprint: {}", identity.fingerprint);
}
