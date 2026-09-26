use std::fs;
use std::process::Command;

use inference_engine::pool::PairingOffer;

#[test]
fn owner_explicitly_trusts_and_removes_a_second_device() {
    let directory = tempfile::tempdir().unwrap();
    let first = directory.path().join("first");
    let second = directory.path().join("second");
    let executable = env!("CARGO_BIN_EXE_device");

    for path in [&first, &second] {
        let result = Command::new(executable)
            .arg("init")
            .arg(path)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
    let result = Command::new(executable)
        .arg("offer")
        .arg(&second)
        .output()
        .unwrap();
    assert!(result.status.success());
    let offer: PairingOffer = serde_json::from_slice(&result.stdout).unwrap();
    let offer_path = directory.path().join("offer.json");
    fs::write(&offer_path, &result.stdout).unwrap();

    let rejected = Command::new(executable)
        .args([
            "trust",
            first.to_str().unwrap(),
            offer_path.to_str().unwrap(),
            "127.0.0.1:18888",
            "wrong-fingerprint",
        ])
        .output()
        .unwrap();
    assert!(!rejected.status.success());
    let approved = Command::new(executable)
        .args([
            "trust",
            first.to_str().unwrap(),
            offer_path.to_str().unwrap(),
            "127.0.0.1:18888",
            &offer.fingerprint,
        ])
        .output()
        .unwrap();
    assert!(
        approved.status.success(),
        "{}",
        String::from_utf8_lossy(&approved.stderr)
    );

    let listed = Command::new(executable)
        .arg("peers")
        .arg(&first)
        .output()
        .unwrap();
    assert!(listed.status.success());
    assert!(String::from_utf8(listed.stdout)
        .unwrap()
        .contains(&offer.device_id));
    let removed = Command::new(executable)
        .arg("remove")
        .arg(&first)
        .arg(&offer.device_id)
        .output()
        .unwrap();
    assert!(removed.status.success());
    let listed = Command::new(executable)
        .arg("peers")
        .arg(&first)
        .output()
        .unwrap();
    assert!(listed.status.success());
    assert!(listed.stdout.is_empty());
}
