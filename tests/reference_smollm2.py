"""Independent NumPy reference for a local SmolLM2-135M Safetensors checkpoint.

Run with: python3 tests/reference_smollm2.py CONFIG_JSON MODEL_SAFETENSORS
"""

import json
import mmap
import sys
from pathlib import Path

import numpy as np

config = json.loads(Path(sys.argv[1]).read_text())
model_file = Path(sys.argv[2]).open("rb")
model_map = mmap.mmap(model_file.fileno(), 0, access=mmap.ACCESS_READ)
header_size = int.from_bytes(model_map[:8], "little")
header = json.loads(model_map[8 : 8 + header_size])
data_start = 8 + header_size
hidden_size = config["hidden_size"]
head_count = config["num_attention_heads"]
kv_head_count = config["num_key_value_heads"]
head_size = hidden_size // head_count
heads_per_kv_head = head_count // kv_head_count
rope_theta = np.float32(config["rope_theta"])
epsilon = np.float32(config["rms_norm_eps"])


def tensor(name: str) -> np.ndarray:
    info = header[name]
    start, end = info["data_offsets"]
    shape = info["shape"]
    count = int(np.prod(shape))
    if info["dtype"] != "BF16" or end - start != count * 2:
        raise ValueError(f"Unexpected tensor type or size: {name}")
    raw = np.frombuffer(model_map, dtype="<u2", count=count, offset=data_start + start)
    return (raw.astype(np.uint32) << 16).view(np.float32).reshape(shape)


def norm(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    variance = np.mean(values * values, dtype=np.float32)
    return values * (np.float32(1) / np.sqrt(variance + epsilon)) * weights


def rope(values: np.ndarray, position: int) -> np.ndarray:
    heads = values.reshape(-1, head_size).copy()
    half = head_size // 2
    indices = np.arange(half, dtype=np.float32)
    frequencies = np.power(rope_theta, -np.float32(2) * indices / head_size)
    angles = np.float32(position) * frequencies
    cosine = np.cos(angles)
    sine = np.sin(angles)
    first = heads[:, :half].copy()
    second = heads[:, half:].copy()
    heads[:, :half] = first * cosine - second * sine
    heads[:, half:] = second * cosine + first * sine
    return heads


def silu(values: np.ndarray) -> np.ndarray:
    return values / (np.float32(1) + np.exp(-values))


embeddings = tensor("model.embed_tokens.weight")
caches = [{"keys": [], "values": []} for _ in range(config["num_hidden_layers"])]
for position, token in enumerate([1, 2, 3]):
    hidden = embeddings[token].copy()
    for layer_index in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{layer_index}"
        normalized = norm(hidden, tensor(f"{prefix}.input_layernorm.weight"))
        query = rope(tensor(f"{prefix}.self_attn.q_proj.weight") @ normalized, position)
        key = rope(tensor(f"{prefix}.self_attn.k_proj.weight") @ normalized, position)
        value = (tensor(f"{prefix}.self_attn.v_proj.weight") @ normalized).reshape(
            kv_head_count, head_size
        )
        caches[layer_index]["keys"].append(key)
        caches[layer_index]["values"].append(value)
        keys = np.stack(caches[layer_index]["keys"])
        values = np.stack(caches[layer_index]["values"])
        attended = np.zeros((head_count, head_size), dtype=np.float32)
        for head in range(head_count):
            kv_head = head // heads_per_kv_head
            scores = keys[:, kv_head, :] @ query[head] / np.sqrt(np.float32(head_size))
            probabilities = np.exp(scores - np.max(scores))
            probabilities /= np.sum(probabilities)
            attended[head] = probabilities @ values[:, kv_head, :]
        hidden += tensor(f"{prefix}.self_attn.o_proj.weight") @ attended.reshape(-1)
        normalized = norm(hidden, tensor(f"{prefix}.post_attention_layernorm.weight"))
        gate = tensor(f"{prefix}.mlp.gate_proj.weight") @ normalized
        up = tensor(f"{prefix}.mlp.up_proj.weight") @ normalized
        hidden += tensor(f"{prefix}.mlp.down_proj.weight") @ (silu(gate) * up)
        if not np.isfinite(hidden).all():
            raise ValueError(f"Non-finite values after layer {layer_index}")
logits = embeddings @ norm(hidden, tensor("model.norm.weight"))
print("finite logits:", int(np.isfinite(logits).sum()), "of", len(logits))
print("first logits:", [float(value) for value in logits[:12]])
print("greedy token:", int(np.argmax(logits)))
