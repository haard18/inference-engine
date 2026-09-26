//! A peer route can only be served behind mutual TLS.

use std::io;
use std::net::{SocketAddr, TcpListener};
use std::sync::Arc;

use axum::Router;
use rustls::ServerConfig;

use super::{peer_router, AppState};

/// Owns the peer API and its mandatory TLS configuration.
pub struct PeerServer {
    routes: Router,
    tls: ServerConfig,
}

impl PeerServer {
    pub(super) fn new(state: Arc<AppState>, tls: ServerConfig) -> Self {
        Self {
            routes: peer_router(state),
            tls,
        }
    }

    /// Serve only mutually approved devices on a pre-bound TCP listener.
    pub async fn serve(
        self,
        listener: TcpListener,
        handle: axum_server::Handle<SocketAddr>,
    ) -> io::Result<()> {
        listener.set_nonblocking(true)?;
        let config = axum_server::tls_rustls::RustlsConfig::from_config(Arc::new(self.tls));
        axum_server::from_tcp_rustls(listener, config)?
            .handle(handle)
            .serve(self.routes.into_make_service())
            .await
    }
}
