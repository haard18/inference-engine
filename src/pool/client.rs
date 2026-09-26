//! Requests to one explicitly approved peer over mutual TLS.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::body::{to_bytes, Body, Bytes};
use axum::http::header::{CACHE_CONTROL, CONTENT_TYPE};
use axum::http::{Request, Response, StatusCode};
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::client::conn::http1;
use hyper_util::rt::TokioIo;
use rustls::pki_types::ServerName;
use rustls::ClientConfig;
use tokio::net::TcpStream;
use tokio::sync::mpsc;
use tokio_rustls::TlsConnector;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use super::tls::{client_config, server_name};
use super::{DeviceIdentity, PoolError, TrustedPeer};
use crate::serving::CONVERSATION_HEADER;
use crate::serving::{
    CapacitySnapshot, StageCapacitySnapshot, MAX_STAGE_BATCH_FRAMES, MAX_STAGE_LEASE_MS,
};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(1);
const SNAPSHOT_TIMEOUT: Duration = Duration::from_millis(750);
const FAILURE_COOLDOWN: Duration = Duration::from_secs(5);
const MAX_SNAPSHOT_BYTES: usize = 8 * 1024;
const MAX_RESPONSE_BYTES: usize = 2 * 1024 * 1024;
const MAX_STAGE_BYTES: usize = 4 * 1024 * 1024;
const STREAM_ERROR: &[u8] = b"\n\ndata: {\"error\":{\"message\":\"peer stream ended before completion\",\"type\":\"server_error\",\"code\":\"peer_stream_lost\"}}\n\ndata: [DONE]\n\n";
const DONE_MARKER: &[u8] = b"data: [DONE]\n\n";

#[derive(Clone)]
pub struct PeerClient {
    peer: TrustedPeer,
    tls: Arc<ClientConfig>,
    server_name: ServerName<'static>,
    failure_until: Arc<Mutex<Option<Instant>>>,
}

impl PeerClient {
    pub fn new(identity: &DeviceIdentity, peer: TrustedPeer) -> Result<Self, PoolError> {
        let tls = Arc::new(client_config(identity, &peer)?);
        let server_name = server_name(&peer)?;
        Ok(Self {
            peer,
            tls,
            server_name,
            failure_until: Arc::new(Mutex::new(None)),
        })
    }

    pub fn device_id(&self) -> &str {
        &self.peer.device_id
    }

    pub fn cooling_down(&self) -> bool {
        self.failure_until.lock().map_or(true, |until| {
            until.is_some_and(|time| Instant::now() < time)
        })
    }

    pub async fn snapshot(&self) -> Result<CapacitySnapshot, PoolError> {
        if self.cooling_down() {
            return Err(PoolError::Transport("peer is in a failure cooldown".into()));
        }
        let deadline = tokio::time::Instant::now() + SNAPSHOT_TIMEOUT;
        let result = tokio::time::timeout_at(deadline, async {
            let request = Request::builder()
                .uri("/internal/capacity")
                .header("host", "peer")
                .body(Full::new(Bytes::new()))
                .map_err(|error| PoolError::Transport(error.to_string()))?;
            let response = self.send(request, deadline).await?;
            if response.status() != StatusCode::OK {
                return Err(PoolError::Transport(format!(
                    "capacity endpoint returned {}",
                    response.status()
                )));
            }
            let body = to_bytes(Body::new(response.into_body()), MAX_SNAPSHOT_BYTES)
                .await
                .map_err(|error| PoolError::Transport(error.to_string()))?;
            let snapshot: CapacitySnapshot = serde_json::from_slice(&body)?;
            if snapshot.model_id.is_empty()
                || snapshot.model_digest.len() != 64
                || hex::decode(&snapshot.model_digest).is_err()
                || snapshot.queue_capacity == 0
                || snapshot.queue_capacity > 1024
                || snapshot.queue_available > snapshot.queue_capacity
                || snapshot.max_positions == 0
                || snapshot.max_completion_tokens == 0
                || snapshot.max_completion_tokens > 4096
            {
                return Err(PoolError::Invalid("peer capacity snapshot is invalid"));
            }
            Ok(snapshot)
        })
        .await
        .map_err(|_| PoolError::Transport("capacity request timed out".into()));
        match result {
            Ok(Ok(snapshot)) => {
                self.clear_failure();
                Ok(snapshot)
            }
            Ok(Err(error)) | Err(error) => {
                self.mark_failure();
                Err(error)
            }
        }
    }

    pub async fn stage_snapshot(&self) -> Result<StageCapacitySnapshot, PoolError> {
        let deadline = tokio::time::Instant::now() + SNAPSHOT_TIMEOUT;
        let request = Request::builder()
            .uri("/internal/stage/capacity")
            .header("host", "peer")
            .body(Full::new(Bytes::new()))
            .map_err(|error| PoolError::Transport(error.to_string()))?;
        let result = async {
            let response = self.send(request, deadline).await?;
            if response.status() != StatusCode::OK {
                return Err(PoolError::Transport(format!(
                    "stage capacity returned {}",
                    response.status()
                )));
            }
            let bytes = tokio::time::timeout_at(
                deadline,
                to_bytes(Body::new(response.into_body()), MAX_SNAPSHOT_BYTES),
            )
            .await
            .map_err(|_| PoolError::Transport("stage capacity timed out".into()))?
            .map_err(|error| PoolError::Transport(error.to_string()))?;
            let snapshot: StageCapacitySnapshot = serde_json::from_slice(&bytes)?;
            if snapshot.model_digest.len() != 64
                || hex::decode(&snapshot.model_digest).is_err()
                || snapshot.layer_start >= snapshot.layer_end
                || snapshot.hidden_size == 0
                || snapshot.vocab_size == 0
                || snapshot.max_positions == 0
                || snapshot.max_positions > snapshot.model_max_positions
                || snapshot.stored_weight_bytes == 0
                || snapshot.queue_capacity == 0
                || snapshot.queue_capacity > 16
                || snapshot.queue_available > snapshot.queue_capacity
            {
                return Err(PoolError::Invalid("stage capacity snapshot is invalid"));
            }
            Ok(snapshot)
        }
        .await;
        if result.is_err() {
            self.mark_failure();
        } else {
            self.clear_failure();
        }
        result
    }

    pub async fn reserve_stage(
        &self,
        request_id: Uuid,
        max_positions: usize,
        deadline: tokio::time::Instant,
    ) -> Result<usize, PoolError> {
        if max_positions == 0 {
            return Err(PoolError::Invalid("reserved stage context is empty"));
        }
        let remaining = deadline
            .checked_duration_since(tokio::time::Instant::now())
            .ok_or(PoolError::Transport("stage deadline expired".into()))?;
        let request = Request::builder()
            .method("POST")
            .uri("/internal/stage/reserve")
            .header("host", "peer")
            .header("x-inference-request-id", request_id.to_string())
            .header(
                "x-inference-deadline-ms",
                (remaining.as_millis().clamp(1, 120_000) as u64).to_string(),
            )
            .header(
                "x-inference-session-lease-ms",
                (remaining
                    .as_millis()
                    .clamp(1, u128::from(MAX_STAGE_LEASE_MS)) as u64)
                    .to_string(),
            )
            .header("x-inference-max-positions", max_positions.to_string())
            .body(Full::new(Bytes::new()))
            .map_err(|error| PoolError::Transport(error.to_string()))?;
        let response = match self.send(request, deadline).await {
            Ok(response) => response,
            Err(error) => {
                self.mark_failure();
                return Err(error);
            }
        };
        let status = response.status();
        let body = match tokio::time::timeout_at(
            deadline,
            to_bytes(Body::new(response.into_body()), 4096),
        )
        .await
        {
            Ok(Ok(body)) => body,
            Ok(Err(error)) => {
                self.mark_failure();
                return Err(PoolError::Transport(error.to_string()));
            }
            Err(_) => {
                self.mark_failure();
                return Err(PoolError::Transport("stage reserve timed out".into()));
            }
        };
        if status == StatusCode::TOO_MANY_REQUESTS {
            self.clear_failure();
            return Err(PoolError::Overloaded(
                String::from_utf8_lossy(&body).into_owned(),
            ));
        }
        if status != StatusCode::OK {
            if status.is_server_error() {
                self.mark_failure();
            } else {
                self.clear_failure();
            }
            return Err(PoolError::Transport(format!(
                "stage reserve returned HTTP {status}: {}",
                String::from_utf8_lossy(&body)
            )));
        }
        let reply: serde_json::Value = serde_json::from_slice(&body)?;
        let position = reply["position"]
            .as_u64()
            .and_then(|position| usize::try_from(position).ok())
            .filter(|position| *position <= max_positions)
            .ok_or(PoolError::Invalid("stage reserve position is invalid"))?;
        self.clear_failure();
        Ok(position)
    }

    /// Send one activation without replaying it after a connection or stage failure.
    pub async fn forward_stage(
        &self,
        frame: Vec<u8>,
        request_id: Uuid,
        expected_vocab_size: usize,
        deadline: tokio::time::Instant,
    ) -> Result<Vec<f32>, PoolError> {
        self.forward_stage_batch(frame, 1, request_id, expected_vocab_size, deadline)
            .await
    }

    /// Send consecutive activations in one bounded request and return the final scores.
    pub async fn forward_stage_batch(
        &self,
        frames: Vec<u8>,
        frame_count: usize,
        request_id: Uuid,
        expected_vocab_size: usize,
        deadline: tokio::time::Instant,
    ) -> Result<Vec<f32>, PoolError> {
        let expected_bytes = expected_vocab_size
            .checked_mul(4)
            .filter(|bytes| *bytes > 0 && *bytes <= MAX_STAGE_BYTES)
            .ok_or(PoolError::Invalid("stage vocabulary size is invalid"))?;
        if !(1..=MAX_STAGE_BATCH_FRAMES).contains(&frame_count)
            || frames.len() < frame_count * 64
            || frames.len() > MAX_STAGE_BYTES
        {
            return Err(PoolError::Invalid("stage activation size is invalid"));
        }
        let remaining = deadline
            .checked_duration_since(tokio::time::Instant::now())
            .ok_or(PoolError::Transport("stage deadline expired".into()))?;
        let remaining_ms = remaining.as_millis().clamp(1, 120_000) as u64;
        let lease_ms = remaining
            .as_millis()
            .clamp(1, u128::from(MAX_STAGE_LEASE_MS)) as u64;
        let request = Request::builder()
            .method("POST")
            .uri("/internal/stage/activation")
            .header("host", "peer")
            .header(CONTENT_TYPE, "application/octet-stream")
            .header("x-inference-request-id", request_id.to_string())
            .header("x-inference-deadline-ms", remaining_ms.to_string())
            .header("x-inference-session-lease-ms", lease_ms.to_string())
            .header("x-inference-frame-count", frame_count.to_string())
            .body(Full::new(Bytes::from(frames)))
            .map_err(|error| PoolError::Transport(error.to_string()))?;
        let response = match self.send(request, deadline).await {
            Ok(response) => response,
            Err(error) => {
                self.mark_failure();
                return Err(error);
            }
        };
        let status = response.status();
        let limit = if status == StatusCode::OK {
            MAX_STAGE_BYTES
        } else {
            4096
        };
        let body = match tokio::time::timeout_at(
            deadline,
            to_bytes(Body::new(response.into_body()), limit),
        )
        .await
        {
            Ok(Ok(body)) => body,
            Ok(Err(error)) => {
                self.mark_failure();
                return Err(PoolError::Transport(error.to_string()));
            }
            Err(_) => {
                self.mark_failure();
                return Err(PoolError::Transport("stage response timed out".into()));
            }
        };
        if status != StatusCode::OK {
            if status.is_server_error() {
                self.mark_failure();
            }
            if status == StatusCode::TOO_MANY_REQUESTS {
                return Err(PoolError::Overloaded(
                    String::from_utf8_lossy(&body).into_owned(),
                ));
            }
            return Err(PoolError::Transport(format!(
                "stage returned HTTP {status}: {}",
                String::from_utf8_lossy(&body)
            )));
        }
        if body.len() != expected_bytes {
            self.mark_failure();
            return Err(PoolError::Transport("stage score length is invalid".into()));
        }
        let scores: Vec<f32> = body
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
        if !scores.iter().all(|score| score.is_finite()) {
            self.mark_failure();
            return Err(PoolError::Transport("stage scores are not finite".into()));
        }
        self.clear_failure();
        Ok(scores)
    }

    pub async fn close_stage(
        &self,
        request_id: Uuid,
        deadline: tokio::time::Instant,
    ) -> Result<(), PoolError> {
        let remaining = deadline
            .checked_duration_since(tokio::time::Instant::now())
            .ok_or(PoolError::Transport("stage deadline expired".into()))?;
        let remaining_ms = remaining.as_millis().clamp(1, 120_000) as u64;
        let request = Request::builder()
            .method("POST")
            .uri("/internal/stage/close")
            .header("host", "peer")
            .header("x-inference-request-id", request_id.to_string())
            .header("x-inference-deadline-ms", remaining_ms.to_string())
            .body(Full::new(Bytes::new()))
            .map_err(|error| PoolError::Transport(error.to_string()))?;
        let response = self.send(request, deadline).await?;
        if response.status() == StatusCode::NO_CONTENT {
            Ok(())
        } else {
            Err(PoolError::Transport(format!(
                "stage close returned {}",
                response.status()
            )))
        }
    }

    /// Check whether the suffix still owns a checkpoint before reusing its key/value state.
    pub async fn probe_stage(
        &self,
        request_id: Uuid,
        deadline: tokio::time::Instant,
    ) -> Result<Option<usize>, PoolError> {
        let remaining = deadline
            .checked_duration_since(tokio::time::Instant::now())
            .ok_or(PoolError::Transport("stage deadline expired".into()))?;
        let request = Request::builder()
            .method("POST")
            .uri("/internal/stage/probe")
            .header("host", "peer")
            .header("x-inference-request-id", request_id.to_string())
            .header(
                "x-inference-deadline-ms",
                (remaining.as_millis().clamp(1, 120_000) as u64).to_string(),
            )
            .body(Full::new(Bytes::new()))
            .map_err(|error| PoolError::Transport(error.to_string()))?;
        let response = self.send(request, deadline).await?;
        if response.status() != StatusCode::OK {
            return Err(PoolError::Transport(format!(
                "stage probe returned {}",
                response.status()
            )));
        }
        let body =
            tokio::time::timeout_at(deadline, to_bytes(Body::new(response.into_body()), 4096))
                .await
                .map_err(|_| PoolError::Transport("stage probe timed out".into()))?
                .map_err(|error| PoolError::Transport(error.to_string()))?;
        let value: serde_json::Value = serde_json::from_slice(&body)?;
        match value.get("position") {
            Some(serde_json::Value::Null) => Ok(None),
            Some(serde_json::Value::Number(number)) => number
                .as_u64()
                .and_then(|position| usize::try_from(position).ok())
                .map(Some)
                .ok_or(PoolError::Invalid("stage probe position is invalid")),
            _ => Err(PoolError::Invalid("stage probe response is invalid")),
        }
    }

    /// Rewind a suffix session to a prompt checkpoint after generation.
    pub async fn rewind_stage(
        &self,
        request_id: Uuid,
        position: usize,
        deadline: tokio::time::Instant,
    ) -> Result<(), PoolError> {
        let remaining = deadline
            .checked_duration_since(tokio::time::Instant::now())
            .ok_or(PoolError::Transport("stage deadline expired".into()))?;
        let request = Request::builder()
            .method("POST")
            .uri("/internal/stage/rewind")
            .header("host", "peer")
            .header("x-inference-request-id", request_id.to_string())
            .header("x-inference-position", position.to_string())
            .header(
                "x-inference-deadline-ms",
                (remaining.as_millis().clamp(1, 120_000) as u64).to_string(),
            )
            .body(Full::new(Bytes::new()))
            .map_err(|error| PoolError::Transport(error.to_string()))?;
        let response = self.send(request, deadline).await?;
        if response.status() == StatusCode::NO_CONTENT {
            Ok(())
        } else {
            Err(PoolError::Transport(format!(
                "stage rewind returned {}",
                response.status()
            )))
        }
    }

    pub async fn forward_chat(
        &self,
        body: Vec<u8>,
        stream: bool,
        conversation_id: Option<&str>,
        response_timeout: Duration,
    ) -> Result<Response<Body>, PoolError> {
        let deadline = tokio::time::Instant::now() + response_timeout;
        let mut request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("host", "peer")
            .header(CONTENT_TYPE, "application/json");
        if let Some(id) = conversation_id {
            request = request.header(CONVERSATION_HEADER, id);
        }
        let request = request
            .body(Full::new(Bytes::from(body)))
            .map_err(|error| PoolError::Transport(error.to_string()))?;
        let response = match self.send(request, deadline).await {
            Ok(response) => response,
            Err(error) => {
                self.mark_failure();
                return Err(error);
            }
        };
        if response.status() == StatusCode::SERVICE_UNAVAILABLE {
            self.mark_failure();
        }
        let mut forwarded = Response::builder().status(response.status());
        for header in [
            CONTENT_TYPE.as_str(),
            CACHE_CONTROL.as_str(),
            CONVERSATION_HEADER,
        ] {
            if let Some(value) = response.headers().get(header) {
                forwarded = forwarded.header(header, value);
            }
        }
        let body = if stream && response.status() == StatusCode::OK {
            let event_stream = response
                .headers()
                .get(CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .is_some_and(|value| value.starts_with("text/event-stream"));
            if !event_stream {
                self.mark_failure();
                return Err(PoolError::Transport(
                    "peer returned a non-streaming response".into(),
                ));
            }
            self.stream_body(response.into_body(), deadline)
        } else {
            let bytes = tokio::time::timeout_at(
                deadline,
                to_bytes(Body::new(response.into_body()), MAX_RESPONSE_BYTES),
            )
            .await;
            let bytes = match bytes {
                Ok(Ok(bytes)) => bytes,
                Ok(Err(error)) => {
                    self.mark_failure();
                    return Err(PoolError::Transport(error.to_string()));
                }
                Err(_) => {
                    self.mark_failure();
                    return Err(PoolError::Transport("peer body timed out".into()));
                }
            };
            Body::from(bytes)
        };
        forwarded
            .body(body)
            .map_err(|error| PoolError::Transport(error.to_string()))
    }

    fn stream_body(&self, mut incoming: Incoming, deadline: tokio::time::Instant) -> Body {
        let (sender, receiver) = mpsc::channel::<Result<Bytes, std::io::Error>>(8);
        let peer = self.clone();
        tokio::spawn(async move {
            let mut tail = Vec::new();
            let mut completed = false;
            let mut total = 0usize;
            loop {
                let next = tokio::select! {
                    _ = sender.closed() => return,
                    next = tokio::time::timeout_at(deadline, incoming.frame()) => next,
                };
                let data = match next {
                    Ok(Some(Ok(frame))) => match frame.into_data() {
                        Ok(data) => data,
                        Err(_) => continue,
                    },
                    _ => break,
                };
                total = total.saturating_add(data.len());
                if total > MAX_RESPONSE_BYTES {
                    break;
                }
                tail.extend_from_slice(&data);
                if tail
                    .windows(DONE_MARKER.len())
                    .any(|window| window == DONE_MARKER)
                {
                    completed = true;
                }
                if tail.len() > DONE_MARKER.len() {
                    tail.drain(..tail.len() - DONE_MARKER.len());
                }
                if sender.send(Ok(data)).await.is_err() {
                    return;
                }
                if completed {
                    return;
                }
            }
            if !completed {
                peer.mark_failure();
                let _ = sender.send(Ok(Bytes::from_static(STREAM_ERROR))).await;
            }
        });
        Body::from_stream(ReceiverStream::new(receiver))
    }

    async fn send(
        &self,
        request: Request<Full<Bytes>>,
        deadline: tokio::time::Instant,
    ) -> Result<Response<Incoming>, PoolError> {
        let stage_deadline = deadline.min(tokio::time::Instant::now() + CONNECT_TIMEOUT);
        let socket = tokio::time::timeout_at(stage_deadline, TcpStream::connect(self.peer.address))
            .await
            .map_err(|_| PoolError::Transport("peer connection timed out".into()))?
            .map_err(|error| PoolError::Transport(error.to_string()))?;
        let connector = TlsConnector::from(Arc::clone(&self.tls));
        let stage_deadline = deadline.min(tokio::time::Instant::now() + CONNECT_TIMEOUT);
        let stream = tokio::time::timeout_at(
            stage_deadline,
            connector.connect(self.server_name.clone(), socket),
        )
        .await
        .map_err(|_| PoolError::Transport("peer handshake timed out".into()))?
        .map_err(|error| PoolError::Transport(error.to_string()))?;
        let stage_deadline = deadline.min(tokio::time::Instant::now() + CONNECT_TIMEOUT);
        let (mut sender, connection) =
            tokio::time::timeout_at(stage_deadline, http1::handshake(TokioIo::new(stream)))
                .await
                .map_err(|_| PoolError::Transport("peer HTTP handshake timed out".into()))?
                .map_err(|error| PoolError::Transport(error.to_string()))?;
        tokio::spawn(async move {
            let _ = connection.await;
        });
        tokio::time::timeout_at(deadline, sender.send_request(request))
            .await
            .map_err(|_| PoolError::Transport("peer response timed out".into()))?
            .map_err(|error| PoolError::Transport(error.to_string()))
    }

    fn mark_failure(&self) {
        if let Ok(mut until) = self.failure_until.lock() {
            *until = Some(Instant::now() + FAILURE_COOLDOWN);
        }
    }

    fn clear_failure(&self) {
        if let Ok(mut until) = self.failure_until.lock() {
            *until = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::net::SocketAddr;

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio_rustls::TlsAcceptor;

    use super::*;
    use crate::pool::tls::server_config;
    use crate::pool::PeerStore;

    #[tokio::test]
    async fn truncated_peer_response_is_not_reported_as_complete() {
        let a_dir = tempfile::tempdir().unwrap();
        let b_dir = tempfile::tempdir().unwrap();
        let a = DeviceIdentity::load_or_create(a_dir.path()).unwrap();
        let b = DeviceIdentity::load_or_create(b_dir.path()).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address: SocketAddr = listener.local_addr().unwrap();
        let mut a_peers = PeerStore::load(a_dir.path()).unwrap();
        a_peers
            .trust(&a.device_id, &b.offer(), address, &b.fingerprint)
            .unwrap();
        let mut b_peers = PeerStore::load(b_dir.path()).unwrap();
        b_peers
            .trust(&b.device_id, &a.offer(), address, &a.fingerprint)
            .unwrap();
        let acceptor = TlsAcceptor::from(Arc::new(server_config(&b, &b_peers).unwrap()));
        let server = tokio::spawn(async move {
            for _ in 0..2 {
                let (socket, _) = listener.accept().await.unwrap();
                let mut stream = acceptor.accept(socket).await.unwrap();
                let mut request = [0; 2048];
                let _ = stream.read(&mut request).await.unwrap();
                stream
                    .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 999\r\n\r\ndata: {\"partial\":true}\n\n")
                    .await
                    .unwrap();
                stream.flush().await.unwrap();
            }
        });
        let client = PeerClient::new(&a, a_peers.peers()[0].clone()).unwrap();
        let streamed = client
            .forward_chat(b"{}".to_vec(), true, None, Duration::from_secs(2))
            .await
            .unwrap();
        let bytes =
            tokio::time::timeout(Duration::from_secs(3), to_bytes(streamed.into_body(), 4096))
                .await
                .unwrap()
                .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(text.contains("peer_stream_lost"), "{text}");
        assert!(text.contains("data: [DONE]"), "{text}");
        assert!(client.cooling_down());

        let ordinary = client
            .forward_chat(b"{}".to_vec(), false, None, Duration::from_secs(2))
            .await;
        assert!(ordinary.is_err());
        tokio::time::timeout(Duration::from_secs(3), server)
            .await
            .unwrap()
            .unwrap();
    }
}
