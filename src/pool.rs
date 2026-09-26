//! Device identity and owner-approved peer trust for the future device pool.

use std::fmt;
use std::fs;
use std::io::{self, Write};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};

use rcgen::{CertificateParams, KeyPair};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;
use x509_parser::extensions::GeneralName;
use x509_parser::parse_x509_certificate;

const IDENTITY_FILE: &str = "identity.json";
const PEERS_FILE: &str = "peers.json";
const STORE_VERSION: u8 = 1;

pub mod client;
pub mod tls;

#[derive(Debug)]
pub enum PoolError {
    Io(io::Error),
    Json(serde_json::Error),
    Invalid(&'static str),
    Certificate(String),
    Tls(String),
    Transport(String),
    FingerprintMismatch,
}

impl fmt::Display for PoolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "device state I/O failed: {error}"),
            Self::Json(error) => write!(f, "invalid device state: {error}"),
            Self::Invalid(message) => write!(f, "invalid device state: {message}"),
            Self::Certificate(message) => write!(f, "device certificate failed: {message}"),
            Self::Tls(message) => write!(f, "peer TLS configuration failed: {message}"),
            Self::Transport(message) => write!(f, "peer request failed: {message}"),
            Self::FingerprintMismatch => write!(f, "peer certificate fingerprint does not match"),
        }
    }
}

impl std::error::Error for PoolError {}

impl From<io::Error> for PoolError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for PoolError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct IdentityFile {
    version: u8,
    device_id: String,
    certificate_der_hex: String,
    private_key_der_hex: String,
    fingerprint: String,
    key_digest: String,
}

/// A stable certificate and private key for one owner-controlled device.
pub struct DeviceIdentity {
    pub device_id: String,
    pub fingerprint: String,
    pub certificate_der: Vec<u8>,
    private_key_der: Vec<u8>,
}

impl DeviceIdentity {
    pub fn load_or_create(state_dir: impl AsRef<Path>) -> Result<Self, PoolError> {
        let state_dir = state_dir.as_ref();
        secure_directory(state_dir)?;
        let path = state_dir.join(IDENTITY_FILE);
        if path.exists() {
            return Self::load(state_dir);
        }
        let device_id = Uuid::new_v4().to_string();
        let key = KeyPair::generate().map_err(|error| PoolError::Certificate(error.to_string()))?;
        let params = CertificateParams::new(vec![format!("{device_id}.inference.local")])
            .map_err(|error| PoolError::Certificate(error.to_string()))?;
        let certificate = params
            .self_signed(&key)
            .map_err(|error| PoolError::Certificate(error.to_string()))?;
        let cert_bytes = certificate.der().as_ref().to_vec();
        let key_bytes = key.serialize_der();
        let record = IdentityFile {
            version: STORE_VERSION,
            device_id,
            certificate_der_hex: hex::encode(&cert_bytes),
            private_key_der_hex: hex::encode(&key_bytes),
            fingerprint: fingerprint(&cert_bytes),
            key_digest: fingerprint(&key_bytes),
        };
        let mut temporary = tempfile::NamedTempFile::new_in(state_dir)?;
        restrict_private_file(temporary.path())?;
        serde_json::to_writer_pretty(&mut temporary, &record)?;
        temporary.flush()?;
        temporary.as_file().sync_all()?;
        match temporary.persist_noclobber(&path) {
            Ok(_) => Self::load(state_dir),
            Err(error) if error.error.kind() == io::ErrorKind::AlreadyExists => {
                Self::load(state_dir)
            }
            Err(error) => Err(PoolError::Io(error.error)),
        }
    }

    pub fn load(state_dir: impl AsRef<Path>) -> Result<Self, PoolError> {
        let path = state_dir.as_ref().join(IDENTITY_FILE);
        check_private_file(&path)?;
        let record: IdentityFile = serde_json::from_slice(&fs::read(path)?)?;
        if record.version != STORE_VERSION {
            return Err(PoolError::Invalid("unsupported identity version"));
        }
        Uuid::parse_str(&record.device_id)
            .map_err(|_| PoolError::Invalid("device ID is not a UUID"))?;
        let certificate_der = hex::decode(record.certificate_der_hex)
            .map_err(|_| PoolError::Invalid("certificate encoding is invalid"))?;
        let private_key_der = hex::decode(record.private_key_der_hex)
            .map_err(|_| PoolError::Invalid("private key encoding is invalid"))?;
        if certificate_der.is_empty() || private_key_der.is_empty() {
            return Err(PoolError::Invalid("certificate or private key is empty"));
        }
        if fingerprint(&certificate_der) != record.fingerprint
            || fingerprint(&private_key_der) != record.key_digest
        {
            return Err(PoolError::Invalid("identity content digest does not match"));
        }
        let key = KeyPair::try_from(private_key_der.as_slice())
            .map_err(|error| PoolError::Certificate(error.to_string()))?;
        let public_key = validate_certificate(&certificate_der, &record.device_id)?;
        if key.public_key_raw() != public_key.as_slice() {
            return Err(PoolError::Invalid("private key does not match certificate"));
        }
        Ok(Self {
            device_id: record.device_id,
            fingerprint: record.fingerprint,
            certificate_der,
            private_key_der,
        })
    }

    pub fn private_key_der(&self) -> &[u8] {
        &self.private_key_der
    }

    pub fn offer(&self) -> PairingOffer {
        PairingOffer {
            device_id: self.device_id.clone(),
            certificate_der_hex: hex::encode(&self.certificate_der),
            fingerprint: self.fingerprint.clone(),
        }
    }
}

/// Public information that another owner can inspect before approving a peer.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairingOffer {
    pub device_id: String,
    pub certificate_der_hex: String,
    pub fingerprint: String,
}

impl PairingOffer {
    pub fn validate(&self) -> Result<Vec<u8>, PoolError> {
        Uuid::parse_str(&self.device_id)
            .map_err(|_| PoolError::Invalid("peer device ID is not a UUID"))?;
        let certificate_der = hex::decode(&self.certificate_der_hex)
            .map_err(|_| PoolError::Invalid("peer certificate encoding is invalid"))?;
        if certificate_der.is_empty() || fingerprint(&certificate_der) != self.fingerprint {
            return Err(PoolError::FingerprintMismatch);
        }
        validate_certificate(&certificate_der, &self.device_id)?;
        Ok(certificate_der)
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrustedPeer {
    pub device_id: String,
    pub address: SocketAddr,
    pub fingerprint: String,
    pub certificate_der_hex: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PeersFile {
    version: u8,
    peers: Vec<TrustedPeer>,
}

/// Explicitly approved peer certificates. No peer is trusted by discovery alone.
pub struct PeerStore {
    state_dir: PathBuf,
    peers: Vec<TrustedPeer>,
}

impl PeerStore {
    pub fn load(state_dir: impl AsRef<Path>) -> Result<Self, PoolError> {
        let state_dir = state_dir.as_ref().to_path_buf();
        secure_directory(&state_dir)?;
        let path = state_dir.join(PEERS_FILE);
        if !path.exists() {
            return Ok(Self {
                state_dir,
                peers: Vec::new(),
            });
        }
        check_private_file(&path)?;
        let record: PeersFile = serde_json::from_slice(&fs::read(path)?)?;
        if record.version != STORE_VERSION {
            return Err(PoolError::Invalid("unsupported peers version"));
        }
        for (index, peer) in record.peers.iter().enumerate() {
            validate_peer(peer)?;
            if record.peers[..index]
                .iter()
                .any(|other| other.device_id == peer.device_id)
            {
                return Err(PoolError::Invalid("duplicate peer device ID"));
            }
        }
        Ok(Self {
            state_dir,
            peers: record.peers,
        })
    }

    pub fn peers(&self) -> &[TrustedPeer] {
        &self.peers
    }

    pub fn trust(
        &mut self,
        own_device_id: &str,
        offer: &PairingOffer,
        address: SocketAddr,
        expected_fingerprint: &str,
    ) -> Result<(), PoolError> {
        offer.validate()?;
        if offer.fingerprint != expected_fingerprint {
            return Err(PoolError::FingerprintMismatch);
        }
        if offer.device_id == own_device_id {
            return Err(PoolError::Invalid("a device cannot trust itself as a peer"));
        }
        validate_address(address)?;
        if self
            .peers
            .iter()
            .any(|peer| peer.device_id == offer.device_id)
        {
            return Err(PoolError::Invalid(
                "peer is already trusted; remove it before replacing",
            ));
        }
        if self
            .peers
            .iter()
            .any(|peer| peer.fingerprint == offer.fingerprint)
        {
            return Err(PoolError::Invalid(
                "certificate is already assigned to another peer",
            ));
        }
        let peer = TrustedPeer {
            device_id: offer.device_id.clone(),
            address,
            fingerprint: offer.fingerprint.clone(),
            certificate_der_hex: offer.certificate_der_hex.clone(),
        };
        self.peers.push(peer);
        if let Err(error) = self.save() {
            self.peers.pop();
            return Err(error);
        }
        Ok(())
    }

    pub fn remove(&mut self, device_id: &str) -> Result<bool, PoolError> {
        let Some(index) = self
            .peers
            .iter()
            .position(|peer| peer.device_id == device_id)
        else {
            return Ok(false);
        };
        let peer = self.peers.remove(index);
        if let Err(error) = self.save() {
            self.peers.insert(index, peer);
            return Err(error);
        }
        Ok(true)
    }

    fn save(&self) -> Result<(), PoolError> {
        let path = self.state_dir.join(PEERS_FILE);
        let mut temporary = tempfile::NamedTempFile::new_in(&self.state_dir)?;
        restrict_private_file(temporary.path())?;
        serde_json::to_writer_pretty(
            &mut temporary,
            &PeersFile {
                version: STORE_VERSION,
                peers: self.peers.clone(),
            },
        )?;
        temporary.flush()?;
        temporary.as_file().sync_all()?;
        temporary
            .persist(path)
            .map_err(|error| PoolError::Io(error.error))?;
        Ok(())
    }
}

fn validate_peer(peer: &TrustedPeer) -> Result<(), PoolError> {
    let offer = PairingOffer {
        device_id: peer.device_id.clone(),
        certificate_der_hex: peer.certificate_der_hex.clone(),
        fingerprint: peer.fingerprint.clone(),
    };
    offer.validate()?;
    validate_address(peer.address)
}

fn validate_address(address: SocketAddr) -> Result<(), PoolError> {
    let ip = address.ip();
    if address.port() == 0
        || ip.is_unspecified()
        || ip.is_multicast()
        || matches!(ip, IpAddr::V4(value) if value == Ipv4Addr::BROADCAST)
    {
        return Err(PoolError::Invalid(
            "peer address must be a unicast IP and nonzero port",
        ));
    }
    Ok(())
}

fn fingerprint(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn validate_certificate(bytes: &[u8], device_id: &str) -> Result<Vec<u8>, PoolError> {
    let (remainder, certificate) = parse_x509_certificate(bytes)
        .map_err(|_| PoolError::Invalid("device certificate is malformed"))?;
    if !remainder.is_empty() || certificate.subject() != certificate.issuer() {
        return Err(PoolError::Invalid("device certificate must be self-signed"));
    }
    if certificate.is_ca() || !certificate.validity().is_valid() {
        return Err(PoolError::Invalid(
            "device certificate is a CA or is outside its validity period",
        ));
    }
    certificate
        .verify_signature(None)
        .map_err(|_| PoolError::Invalid("device certificate signature is invalid"))?;
    let expected_name = format!("{device_id}.inference.local");
    let san = certificate
        .subject_alternative_name()
        .map_err(|_| PoolError::Invalid("device certificate name is malformed"))?
        .ok_or(PoolError::Invalid("device certificate has no device name"))?;
    if !san
        .value
        .general_names
        .iter()
        .any(|name| matches!(name, GeneralName::DNSName(value) if *value == expected_name))
    {
        return Err(PoolError::Invalid(
            "device certificate name does not match device ID",
        ));
    }
    Ok(certificate.public_key().subject_public_key.data.to_vec())
}

fn secure_directory(path: &Path) -> Result<(), PoolError> {
    fs::create_dir_all(path)?;
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err(PoolError::Invalid("device state path is not a directory"));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

fn check_private_file(path: &Path) -> Result<(), PoolError> {
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err(PoolError::Invalid("device state file is not regular"));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o077 != 0 {
            return Err(PoolError::Invalid(
                "device state file is accessible to other users",
            ));
        }
    }
    Ok(())
}

fn restrict_private_file(path: &Path) -> Result<(), PoolError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_is_stable_private_and_exports_no_key() {
        let directory = tempfile::tempdir().unwrap();
        let first = DeviceIdentity::load_or_create(directory.path()).unwrap();
        let second = DeviceIdentity::load_or_create(directory.path()).unwrap();
        assert_eq!(first.device_id, second.device_id);
        assert_eq!(first.fingerprint, second.fingerprint);
        assert_eq!(first.private_key_der(), second.private_key_der());
        let offer = first.offer();
        assert!(offer.validate().is_ok());
        let public_json = serde_json::to_string(&offer).unwrap();
        assert!(!public_json.contains(&hex::encode(first.private_key_der())));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(directory.path().join(IDENTITY_FILE))
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o077, 0);
        }
    }

    #[test]
    fn trust_requires_exact_fingerprint_and_persists_removal() {
        let own_dir = tempfile::tempdir().unwrap();
        let peer_dir = tempfile::tempdir().unwrap();
        let own = DeviceIdentity::load_or_create(own_dir.path()).unwrap();
        let peer = DeviceIdentity::load_or_create(peer_dir.path()).unwrap();
        let mut peers = PeerStore::load(own_dir.path()).unwrap();
        let address: SocketAddr = "127.0.0.1:18888".parse().unwrap();
        assert!(matches!(
            peers.trust(&own.device_id, &peer.offer(), address, &own.fingerprint),
            Err(PoolError::FingerprintMismatch)
        ));
        assert!(peers.peers().is_empty());
        peers
            .trust(&own.device_id, &peer.offer(), address, &peer.fingerprint)
            .unwrap();
        assert_eq!(PeerStore::load(own_dir.path()).unwrap().peers().len(), 1);
        assert!(matches!(
            peers.trust(&own.device_id, &peer.offer(), address, &peer.fingerprint),
            Err(PoolError::Invalid(_))
        ));
        assert!(peers.remove(&peer.device_id).unwrap());
        assert!(PeerStore::load(own_dir.path()).unwrap().peers().is_empty());
    }

    #[test]
    fn corrupted_identity_and_offer_are_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let other_directory = tempfile::tempdir().unwrap();
        let identity = DeviceIdentity::load_or_create(directory.path()).unwrap();
        let other = DeviceIdentity::load_or_create(other_directory.path()).unwrap();
        let mut offer = identity.offer();
        offer.certificate_der_hex.push_str("00");
        assert!(matches!(
            offer.validate(),
            Err(PoolError::FingerprintMismatch)
        ));
        let malformed = PairingOffer {
            device_id: identity.device_id.clone(),
            certificate_der_hex: "00".into(),
            fingerprint: fingerprint(&[0]),
        };
        assert!(malformed.validate().is_err());
        let path = directory.path().join(IDENTITY_FILE);
        let mut record: IdentityFile = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        record.private_key_der_hex.push_str("00");
        fs::write(&path, serde_json::to_vec(&record).unwrap()).unwrap();
        assert!(DeviceIdentity::load(directory.path()).is_err());

        record.private_key_der_hex = hex::encode(other.private_key_der());
        record.key_digest = fingerprint(other.private_key_der());
        fs::write(&path, serde_json::to_vec(&record).unwrap()).unwrap();
        assert!(matches!(
            DeviceIdentity::load(directory.path()),
            Err(PoolError::Invalid("private key does not match certificate"))
        ));
    }
}
