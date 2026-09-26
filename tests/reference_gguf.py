"""Independent NumPy reference for a local SmolLM2-135M GGUF file.

Run with: python3 tests/reference_gguf.py MODEL_GGUF
"""

import mmap
import struct
import sys
from pathlib import Path

import numpy as np

model_file = Path(sys.argv[1]).open("rb")
model_map = mmap.mmap(model_file.fileno(), 0, access=mmap.ACCESS_READ)
position = 0


def read(size: int) -> bytes:
    global position
    data = model_map[position : position + size]
    position += size
    return data


def uint32() -> int:
    return struct.unpack("<I", read(4))[0]


def uint64() -> int:
    return struct.unpack("<Q", read(8))[0]


def string() -> str:
    return read(uint64()).decode("utf-8")


def value(kind: int):
    if kind == 8:
        return string()
    if kind == 9:
        item_kind = uint32()
        return [value(item_kind) for _ in range(uint64())]
    formats = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    code = formats[kind]
    return struct.unpack("<" + code, read(struct.calcsize(code)))[0]


assert read(4) == b"GGUF"
assert uint32() == 3
tensor_count = uint64()
metadata_count = uint64()
metadata = {}
for _ in range(metadata_count):
    key = string()
    metadata[key] = value(uint32())

tensor_info = {}
for _ in range(tensor_count):
    name = string()
    dimensions = [uint64() for _ in range(uint32())]
    kind = uint32()
    offset = uint64()
    tensor_info[name] = (dimensions, kind, offset)

alignment = metadata.get("general.alignment", 32)
data_start = (position + alignment - 1) // alignment * alignment
hidden_size = metadata["llama.embedding_length"]
head_count = metadata["llama.attention.head_count"]
kv_head_count = metadata["llama.attention.head_count_kv"]
head_size = hidden_size // head_count
heads_per_kv_head = head_count // kv_head_count
rope_theta = np.float32(metadata["llama.rope.freq_base"])
epsilon = np.float32(metadata["llama.attention.layer_norm_rms_epsilon"])


def tensor(name: str) -> np.ndarray:
    dimensions, kind, offset = tensor_info[name]
    shape = tuple(reversed(dimensions))
    count = int(np.prod(shape))
    if kind == 0:
        return np.frombuffer(model_map, dtype="<f4", count=count, offset=data_start + offset).reshape(shape)
    if kind == 6 and dimensions[0] % 32 == 0:
        blocks = np.frombuffer(model_map, dtype=np.uint8, count=count // 32 * 22, offset=data_start + offset).reshape(-1, 22)
        scales = np.frombuffer(blocks[:, :2].copy().tobytes(), dtype="<f2").astype(np.float32)
        high = np.frombuffer(blocks[:, 2:6].copy().tobytes(), dtype="<u4")[:, None]
        packed = blocks[:, 6:].astype(np.int32)
        positions = np.arange(16, dtype=np.uint32)
        low = (packed & 15) | (((high >> positions) & 1) << 4)
        upper = (packed >> 4) | (((high >> (positions + 16)) & 1) << 4)
        values = np.concatenate([low, upper], axis=1).astype(np.float32) - 16
        return (values * scales[:, None]).reshape(shape)
    if kind == 12 and dimensions[0] % 256 == 0:
        blocks = np.frombuffer(model_map, dtype=np.uint8, count=count // 256 * 144, offset=data_start + offset).reshape(-1, 144)
        scales = blocks[:, 4:16]
        q = blocks[:, 16:]
        d = np.frombuffer(blocks[:, :2].copy().tobytes(), dtype="<f2").astype(np.float32)
        dmin = np.frombuffer(blocks[:, 2:4].copy().tobytes(), dtype="<f2").astype(np.float32)
        output = np.empty((len(blocks), 256), dtype=np.float32)

        def scale_min(index: int):
            if index < 4:
                return scales[:, index] & 63, scales[:, index + 4] & 63
            return (scales[:, index + 4] & 15) | ((scales[:, index - 4] >> 6) << 4), \
                (scales[:, index + 4] >> 4) | ((scales[:, index] >> 6) << 4)

        for group in range(4):
            lower_scale, lower_min = scale_min(group * 2)
            upper_scale, upper_min = scale_min(group * 2 + 1)
            packed = q[:, group * 32:(group + 1) * 32]
            output[:, group * 64:group * 64 + 32] = \
                (d * lower_scale)[:, None] * (packed & 15) - (dmin * lower_min)[:, None]
            output[:, group * 64 + 32:group * 64 + 64] = \
                (d * upper_scale)[:, None] * (packed >> 4) - (dmin * upper_min)[:, None]
        return output.reshape(shape)
    if kind != 8 or dimensions[0] % 32:
        raise ValueError(f"Unexpected tensor type: {name}, {kind}")
    blocks = np.frombuffer(model_map, dtype=np.uint8, count=count // 32 * 34, offset=data_start + offset).reshape(-1, 34)
    scales = np.frombuffer(blocks[:, :2].copy().tobytes(), dtype="<f2").astype(np.float32)
    quants = blocks[:, 2:].copy().view(np.int8).astype(np.float32)
    return (scales[:, None] * quants).reshape(shape)


def norm(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    variance = np.mean(values * values, dtype=np.float32)
    return values * (np.float32(1) / np.sqrt(variance + epsilon)) * weights


def rope(values: np.ndarray, step: int) -> np.ndarray:
    pairs = values.reshape(-1, head_size // 2, 2).copy()
    index = np.arange(head_size // 2, dtype=np.float32)
    angles = np.float32(step) * np.power(rope_theta, -np.float32(2) * index / head_size)
    cosine, sine = np.cos(angles), np.sin(angles)
    first = pairs[:, :, 0].copy()
    second = pairs[:, :, 1].copy()
    pairs[:, :, 0] = first * cosine - second * sine
    pairs[:, :, 1] = second * cosine + first * sine
    return pairs.reshape(-1, head_size)


def silu(values: np.ndarray) -> np.ndarray:
    return values / (np.float32(1) + np.exp(-values))


embeddings = tensor("token_embd.weight")
caches = [{"keys": [], "values": []} for _ in range(metadata["llama.block_count"])]
for step, token in enumerate([1, 2, 3]):
    hidden = embeddings[token].copy()
    for layer_index in range(metadata["llama.block_count"]):
        prefix = f"blk.{layer_index}"
        normalized = norm(hidden, tensor(f"{prefix}.attn_norm.weight"))
        query = rope(tensor(f"{prefix}.attn_q.weight") @ normalized, step)
        key = rope(tensor(f"{prefix}.attn_k.weight") @ normalized, step)
        value_vector = (tensor(f"{prefix}.attn_v.weight") @ normalized).reshape(kv_head_count, head_size)
        caches[layer_index]["keys"].append(key)
        caches[layer_index]["values"].append(value_vector)
        keys = np.stack(caches[layer_index]["keys"])
        values = np.stack(caches[layer_index]["values"])
        attended = np.zeros((head_count, head_size), dtype=np.float32)
        for head in range(head_count):
            kv_head = head // heads_per_kv_head
            scores = keys[:, kv_head, :] @ query[head] / np.sqrt(np.float32(head_size))
            probabilities = np.exp(scores - np.max(scores))
            probabilities /= np.sum(probabilities)
            attended[head] = probabilities @ values[:, kv_head, :]
        hidden += tensor(f"{prefix}.attn_output.weight") @ attended.reshape(-1)
        normalized = norm(hidden, tensor(f"{prefix}.ffn_norm.weight"))
        gate = tensor(f"{prefix}.ffn_gate.weight") @ normalized
        up = tensor(f"{prefix}.ffn_up.weight") @ normalized
        hidden += tensor(f"{prefix}.ffn_down.weight") @ (silu(gate) * up)
        if not np.isfinite(hidden).all():
            raise ValueError(f"Non-finite values after layer {layer_index}")

logits = embeddings @ norm(hidden, tensor("output_norm.weight"))
print("finite logits:", int(np.isfinite(logits).sum()), "of", len(logits))
print("first logits:", [float(item) for item in logits[:12]])
print("greedy token:", int(np.argmax(logits)))
