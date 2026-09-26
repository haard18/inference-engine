//! Capacity-aware whole-request choice among the local worker and approved peers.

use tokio::task::JoinSet;

use crate::pool::client::PeerClient;
use crate::pool::{DeviceIdentity, PeerStore, PoolError};

use super::CapacitySnapshot;

pub(super) struct Coordinator {
    peers: Vec<PeerClient>,
}

impl Coordinator {
    pub fn new(identity: &DeviceIdentity, peers: &PeerStore) -> Result<Self, PoolError> {
        let clients = peers
            .peers()
            .iter()
            .cloned()
            .map(|peer| PeerClient::new(identity, peer))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { peers: clients })
    }

    pub async fn choose(
        &self,
        local: &CapacitySnapshot,
        model_id: &str,
        model_digest: &str,
        required_positions: usize,
        max_completion_tokens: usize,
    ) -> Option<PeerClient> {
        let local_score = if local.ready && local.queue_available > 0 {
            local.backlog()
        } else {
            usize::MAX
        };
        if local_score == 0 {
            return None;
        }
        let mut tasks = JoinSet::new();
        for peer in &self.peers {
            if peer.cooling_down() {
                continue;
            }
            let peer = peer.clone();
            tasks.spawn(async move {
                let snapshot = peer.snapshot().await.ok()?;
                Some((peer, snapshot))
            });
        }
        let mut selected = None;
        let mut best_score = local_score;
        while let Some(result) = tasks.join_next().await {
            let Ok(Some((peer, snapshot))) = result else {
                continue;
            };
            if peer_can_take(
                &snapshot,
                model_id,
                model_digest,
                required_positions,
                max_completion_tokens,
            ) && snapshot.backlog() < best_score
            {
                best_score = snapshot.backlog();
                selected = Some(peer);
            }
        }
        selected
    }

    pub async fn owner_if_available(
        &self,
        device_id: &str,
        model_id: &str,
        model_digest: &str,
        required_positions: usize,
        max_completion_tokens: usize,
    ) -> Option<PeerClient> {
        let peer = self
            .peers
            .iter()
            .find(|peer| peer.device_id() == device_id)?;
        if peer.cooling_down() {
            return None;
        }
        let snapshot = peer.snapshot().await.ok()?;
        peer_can_take(
            &snapshot,
            model_id,
            model_digest,
            required_positions,
            max_completion_tokens,
        )
        .then(|| peer.clone())
    }
}

fn peer_can_take(
    snapshot: &CapacitySnapshot,
    model_id: &str,
    model_digest: &str,
    required_positions: usize,
    max_completion_tokens: usize,
) -> bool {
    snapshot.ready
        && snapshot.queue_available > 0
        && snapshot.model_id == model_id
        && snapshot.model_digest == model_digest
        && required_positions <= snapshot.max_positions
        && max_completion_tokens <= snapshot.max_completion_tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_compatible_workers_with_queue_space_are_candidates() {
        let mut peer = CapacitySnapshot {
            model_id: "model-a".into(),
            model_digest: "07".repeat(32),
            ready: true,
            active: false,
            queue_available: 2,
            queue_capacity: 2,
            max_positions: 512,
            max_completion_tokens: 64,
        };
        assert!(peer_can_take(&peer, "model-a", &"07".repeat(32), 512, 64));
        assert_eq!(peer.backlog(), 0);
        assert!(!peer_can_take(&peer, "model-b", &"07".repeat(32), 512, 64));
        assert!(!peer_can_take(&peer, "model-a", &"08".repeat(32), 512, 64));
        assert!(!peer_can_take(&peer, "model-a", &"07".repeat(32), 513, 64));
        assert!(!peer_can_take(&peer, "model-a", &"07".repeat(32), 512, 65));
        peer.queue_available = 0;
        assert!(!peer_can_take(&peer, "model-a", &"07".repeat(32), 512, 64));
        peer.queue_available = 1;
        peer.ready = false;
        assert!(!peer_can_take(&peer, "model-a", &"07".repeat(32), 512, 64));
        peer.ready = true;
        peer.active = true;
        assert_eq!(peer.backlog(), 2);
    }
}
