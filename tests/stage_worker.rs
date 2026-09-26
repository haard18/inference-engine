use std::env;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Duration;

use inference_engine::{load_gguf, GenerationSession};
use serde_json::{json, Value};
use uuid::Uuid;

struct StageProcess {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
    ready: Value,
}

impl StageProcess {
    fn start(path: &Path, start: usize, end: usize) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_serve"))
            .arg("--internal-stage-worker")
            .arg(path)
            .arg(start.to_string())
            .arg(end.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("start stage worker");
        let input = child.stdin.take().unwrap();
        let mut output = BufReader::new(child.stdout.take().unwrap());
        let mut line = String::new();
        output.read_line(&mut line).unwrap();
        let ready: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(ready["kind"], "ready");
        Self {
            child,
            input,
            output,
            ready,
        }
    }

    fn request(&mut self, command: Value, payload: &[u8]) -> (Value, Vec<u8>) {
        writeln!(self.input, "{command}").unwrap();
        self.input.write_all(payload).unwrap();
        self.input.flush().unwrap();
        let mut line = String::new();
        self.output.read_line(&mut line).unwrap();
        let response: Value = serde_json::from_str(&line).unwrap();
        let length = match response["kind"].as_str().unwrap() {
            "activation" => response["payload_bytes"].as_u64().unwrap() as usize,
            "scores" => response["count"].as_u64().unwrap() as usize * 4,
            _ => 0,
        };
        let mut payload = vec![0; length];
        self.output.read_exact(&mut payload).unwrap();
        (response, payload)
    }

    fn reserve(&mut self, request_id: Uuid, max_positions: usize, lease_ms: u64) -> Value {
        self.request(
            json!({"kind":"reserve","request_id":request_id.to_string(),"max_positions":max_positions,"lease_ms":lease_ms}),
            &[],
        )
        .0
    }
}

impl Drop for StageProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
#[ignore = "requires SmolLM2-135M-Q4_K_M.gguf in SMOLLM2_DIR"]
fn two_stage_processes_match_the_complete_real_model() {
    let directory = env::var("SMOLLM2_DIR").expect("set SMOLLM2_DIR");
    let path = Path::new(&directory).join("SmolLM2-135M-Q4_K_M.gguf");
    let model = load_gguf(&path).unwrap();
    let full_bytes = model.stored_weight_bytes() as u64;
    let mut prefix = StageProcess::start(&path, 0, 15);
    let mut suffix = StageProcess::start(&path, 15, 30);
    assert_eq!(prefix.ready["model_digest"], suffix.ready["model_digest"]);
    assert_eq!(prefix.ready["layer_start"], 0);
    assert_eq!(prefix.ready["layer_end"], 15);
    assert_eq!(suffix.ready["layer_start"], 15);
    assert_eq!(suffix.ready["layer_end"], 30);
    assert_eq!(prefix.ready["max_positions"], 4096);
    assert_eq!(prefix.ready["model_max_positions"], 8192);
    assert_eq!(suffix.ready["vocab_size"], 49152);
    assert!(prefix.ready["stored_weight_bytes"].as_u64().unwrap() < full_bytes);
    assert!(suffix.ready["stored_weight_bytes"].as_u64().unwrap() < full_bytes);

    let id = Uuid::new_v4();
    assert_eq!(prefix.reserve(id, 4, 3000)["kind"], "reserved");
    assert_eq!(suffix.reserve(id, 4, 3000)["kind"], "reserved");
    let mut whole = GenerationSession::new(&model);
    for (step, token) in [1, 2, 3, 30].into_iter().enumerate() {
        whole.prefill(&[token]).unwrap();
        let (kind, activation) = prefix.request(
            json!({"kind":"token","request_id":id.to_string(),"token_id":token}),
            &[],
        );
        assert_eq!(kind["kind"], "activation");
        assert_eq!(activation.len(), 2_368);
        if step == 0 {
            let (bad, _) = suffix.request(
                json!({"kind":"activation","request_id":Uuid::new_v4().to_string(),"payload_bytes":activation.len()}),
                &activation,
            );
            assert_eq!(bad["kind"], "failed");
        }
        let (kind, bytes) = suffix.request(
            json!({"kind":"activation","request_id":id.to_string(),"payload_bytes":activation.len()}),
            &activation,
        );
        assert_eq!(kind["kind"], "scores");
        let scores: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes))
            .collect();
        assert_eq!(scores, whole.next_token_scores().unwrap());
    }
    assert_eq!(
        prefix
            .request(json!({"kind":"close","request_id":id.to_string()}), &[])
            .0["kind"],
        "closed"
    );
    assert_eq!(
        suffix
            .request(json!({"kind":"close","request_id":id.to_string()}), &[])
            .0["kind"],
        "closed"
    );

    let ids: Vec<Uuid> = (0..8).map(|_| Uuid::new_v4()).collect();
    for id in &ids {
        assert_eq!(prefix.reserve(*id, 1, 3000)["kind"], "reserved");
        let (kind, _) = prefix.request(
            json!({"kind":"token","request_id":id.to_string(),"token_id":1}),
            &[],
        );
        assert_eq!(kind["kind"], "activation");
    }
    let extra = Uuid::new_v4();
    let full = prefix.reserve(extra, 1, 3000);
    assert_eq!(full["kind"], "failed");
    assert_eq!(
        prefix
            .request(json!({"kind":"close","request_id":ids[0].to_string()}), &[])
            .0["kind"],
        "closed"
    );
    assert_eq!(prefix.reserve(extra, 1, 3000)["kind"], "reserved");
    let (admitted, _) = prefix.request(
        json!({"kind":"token","request_id":extra.to_string(),"token_id":1}),
        &[],
    );
    assert_eq!(admitted["kind"], "activation");
}

#[test]
#[ignore = "requires SmolLM2-135M-Q4_K_M.gguf in SMOLLM2_DIR"]
fn expired_active_stage_session_is_removed_but_checkpoint_remains() {
    let directory = env::var("SMOLLM2_DIR").expect("set SMOLLM2_DIR");
    let path = Path::new(&directory).join("SmolLM2-135M-Q4_K_M.gguf");
    let mut prefix = StageProcess::start(&path, 0, 15);
    let active = Uuid::new_v4();
    assert_eq!(prefix.reserve(active, 1, 3000)["kind"], "reserved");
    let (event, _) = prefix.request(
        json!({"kind":"token","request_id":active.to_string(),"token_id":1,"lease_ms":3000}),
        &[],
    );
    assert_eq!(event["kind"], "activation");
    let checkpoint = Uuid::new_v4();
    assert_eq!(prefix.reserve(checkpoint, 1, 3000)["kind"], "reserved");
    let (event, _) = prefix.request(
        json!({"kind":"token","request_id":checkpoint.to_string(),"token_id":1,"lease_ms":3000}),
        &[],
    );
    assert_eq!(event["kind"], "activation");
    let (rewound, _) = prefix.request(
        json!({"kind":"rewind","request_id":checkpoint.to_string(),"position":1}),
        &[],
    );
    assert_eq!(rewound["kind"], "rewound");
    std::thread::sleep(Duration::from_millis(3100));
    let (expired, _) = prefix.request(json!({"kind":"probe","request_id":active.to_string()}), &[]);
    assert_eq!(expired["position"], Value::Null);
    let (position, _) = prefix.request(
        json!({"kind":"probe","request_id":checkpoint.to_string()}),
        &[],
    );
    assert_eq!(position["kind"], "position");
    assert_eq!(position["position"], 1);
    let invalid = Uuid::new_v4();
    let (failed, _) = prefix.request(
        json!({"kind":"token","request_id":invalid.to_string(),"token_id":1,"lease_ms":0}),
        &[],
    );
    assert_eq!(failed["kind"], "failed");
    let (missing, _) = prefix.request(
        json!({"kind":"probe","request_id":invalid.to_string()}),
        &[],
    );
    assert_eq!(missing["position"], Value::Null);
}
