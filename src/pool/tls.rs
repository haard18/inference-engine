//! Mutual TLS configurations from explicitly approved device certificates.

use std::sync::Arc;

use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer, ServerName};
use rustls::server::WebPkiClientVerifier;
use rustls::{ClientConfig, RootCertStore, ServerConfig};

use super::{DeviceIdentity, PairingOffer, PeerStore, PoolError, TrustedPeer};

pub fn server_config(
    identity: &DeviceIdentity,
    peers: &PeerStore,
) -> Result<ServerConfig, PoolError> {
    if peers.peers().is_empty() {
        return Err(PoolError::Invalid(
            "at least one approved peer is required for LAN serving",
        ));
    }
    let mut roots = RootCertStore::empty();
    for peer in peers.peers() {
        roots
            .add(peer_certificate(peer)?)
            .map_err(|error| PoolError::Tls(error.to_string()))?;
    }
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let verifier =
        WebPkiClientVerifier::builder_with_provider(Arc::new(roots), Arc::clone(&provider))
            .build()
            .map_err(|error| PoolError::Tls(error.to_string()))?;
    let mut config = ServerConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .map_err(|error| PoolError::Tls(error.to_string()))?
        .with_client_cert_verifier(verifier)
        .with_single_cert(vec![own_certificate(identity)], own_key(identity))
        .map_err(|error| PoolError::Tls(error.to_string()))?;
    config.alpn_protocols = vec![b"http/1.1".to_vec()];
    Ok(config)
}

pub fn client_config(
    identity: &DeviceIdentity,
    peer: &TrustedPeer,
) -> Result<ClientConfig, PoolError> {
    let mut roots = RootCertStore::empty();
    roots
        .add(peer_certificate(peer)?)
        .map_err(|error| PoolError::Tls(error.to_string()))?;
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let mut config = ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .map_err(|error| PoolError::Tls(error.to_string()))?
        .with_root_certificates(roots)
        .with_client_auth_cert(vec![own_certificate(identity)], own_key(identity))
        .map_err(|error| PoolError::Tls(error.to_string()))?;
    config.alpn_protocols = vec![b"http/1.1".to_vec()];
    Ok(config)
}

pub fn server_name(peer: &TrustedPeer) -> Result<ServerName<'static>, PoolError> {
    format!("{}.inference.local", peer.device_id)
        .try_into()
        .map_err(|_| PoolError::Invalid("peer device name is invalid"))
}

fn peer_certificate(peer: &TrustedPeer) -> Result<CertificateDer<'static>, PoolError> {
    PairingOffer {
        device_id: peer.device_id.clone(),
        certificate_der_hex: peer.certificate_der_hex.clone(),
        fingerprint: peer.fingerprint.clone(),
    }
    .validate()
    .map(CertificateDer::from)
}

fn own_certificate(identity: &DeviceIdentity) -> CertificateDer<'static> {
    CertificateDer::from(identity.certificate_der.clone())
}

fn own_key(identity: &DeviceIdentity) -> PrivateKeyDer<'static> {
    PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(
        identity.private_key_der().to_vec(),
    ))
}

#[cfg(test)]
mod tests {
    use std::net::SocketAddr;
    use std::time::Duration;

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio_rustls::{TlsAcceptor, TlsConnector};

    use super::*;

    async fn exchange(
        server: &DeviceIdentity,
        approved_clients: &PeerStore,
        client: &DeviceIdentity,
        trusted_server: &TrustedPeer,
    ) -> (Result<(), String>, Result<(), String>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let acceptor =
            TlsAcceptor::from(Arc::new(server_config(server, approved_clients).unwrap()));
        let server_task = tokio::spawn(async move {
            let (socket, _) = listener.accept().await.map_err(|error| error.to_string())?;
            let mut stream = acceptor
                .accept(socket)
                .await
                .map_err(|error| error.to_string())?;
            let mut input = [0];
            stream
                .read_exact(&mut input)
                .await
                .map_err(|error| error.to_string())?;
            if input != [42] {
                return Err("wrong request byte".into());
            }
            stream
                .write_all(&[43])
                .await
                .map_err(|error| error.to_string())
        });
        let connector =
            TlsConnector::from(Arc::new(client_config(client, trusted_server).unwrap()));
        let client_result = tokio::time::timeout(Duration::from_secs(5), async {
            let socket = tokio::net::TcpStream::connect(address)
                .await
                .map_err(|error| error.to_string())?;
            let mut stream = connector
                .connect(server_name(trusted_server).unwrap(), socket)
                .await
                .map_err(|error| error.to_string())?;
            stream
                .write_all(&[42])
                .await
                .map_err(|error| error.to_string())?;
            let mut reply = [0];
            stream
                .read_exact(&mut reply)
                .await
                .map_err(|error| error.to_string())?;
            if reply == [43] {
                Ok(())
            } else {
                Err("wrong response byte".into())
            }
        })
        .await
        .unwrap_or_else(|_| Err("client timed out".into()));
        let server_result = tokio::time::timeout(Duration::from_secs(5), server_task)
            .await
            .unwrap_or_else(|_| panic!("server timed out"))
            .unwrap();
        (client_result, server_result)
    }

    #[tokio::test]
    async fn only_mutually_approved_devices_can_exchange_data() {
        let a_dir = tempfile::tempdir().unwrap();
        let b_dir = tempfile::tempdir().unwrap();
        let c_dir = tempfile::tempdir().unwrap();
        let a = DeviceIdentity::load_or_create(a_dir.path()).unwrap();
        let b = DeviceIdentity::load_or_create(b_dir.path()).unwrap();
        let c = DeviceIdentity::load_or_create(c_dir.path()).unwrap();
        let address: SocketAddr = "127.0.0.1:18888".parse().unwrap();
        let mut b_peers = PeerStore::load(b_dir.path()).unwrap();
        b_peers
            .trust(&b.device_id, &a.offer(), address, &a.fingerprint)
            .unwrap();
        let mut a_peers = PeerStore::load(a_dir.path()).unwrap();
        a_peers
            .trust(&a.device_id, &b.offer(), address, &b.fingerprint)
            .unwrap();
        let trusted_b = &a_peers.peers()[0];

        let (client, server) = exchange(&b, &b_peers, &a, trusted_b).await;
        assert!(client.is_ok(), "approved client: {client:?}");
        assert!(server.is_ok(), "approved server: {server:?}");

        let (client, server) = exchange(&b, &b_peers, &c, trusted_b).await;
        assert!(client.is_err(), "unapproved client was accepted");
        assert!(server.is_err(), "unapproved client reached server");

        let mut c_peers = PeerStore::load(c_dir.path()).unwrap();
        c_peers
            .trust(&c.device_id, &a.offer(), address, &a.fingerprint)
            .unwrap();
        let (client, server) = exchange(&b, &b_peers, &a, &c_peers.peers()[0]).await;
        assert!(client.is_err(), "client accepted the wrong server identity");
        assert!(
            server.is_err(),
            "wrongly identified connection reached the server"
        );
    }
}
