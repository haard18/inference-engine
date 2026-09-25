"""Independent NumPy reference for the tiny Llama-style test model.

Run with: python3 tests/reference.py
"""

import numpy as np

HIDDEN = 4
INTERMEDIATE = 6
VOCAB = 8
HEADS = 2
KV_HEADS = 1
HEAD_SIZE = HIDDEN // HEADS
EPSILON = np.float32(1e-5)
THETA = np.float32(10000.0)


def matrix(seed: int, rows: int, cols: int) -> np.ndarray:
    values = [((index * 17 + seed * 13) % 23 - 11) / 20 for index in range(rows * cols)]
    return np.array(values, dtype=np.float32).reshape(rows, cols)


def rms_norm(values: np.ndarray) -> np.ndarray:
    scale = np.float32(1.0) / np.sqrt(np.mean(values * values) + EPSILON)
    return values * scale


def rope(values: np.ndarray, position: int) -> np.ndarray:
    result = values.copy().reshape(-1, HEAD_SIZE)
    for head in result:
        half = HEAD_SIZE // 2
        for index in range(half):
            frequency = THETA ** np.float32(-2 * index / HEAD_SIZE)
            angle = np.float32(position) * frequency
            sine = np.sin(angle)
            cosine = np.cos(angle)
            first, second = head[index], head[index + half]
            head[index] = first * cosine - second * sine
            head[index + half] = second * cosine + first * sine
    return result


def silu(values: np.ndarray) -> np.ndarray:
    return values / (np.float32(1.0) + np.exp(-values))


embeddings = matrix(1, VOCAB, HIDDEN)
weights = []
for layer in range(2):
    base = 2 + layer * 7
    weights.append(
        {
            "query": matrix(base, HIDDEN, HIDDEN),
            "key": matrix(base + 1, KV_HEADS * HEAD_SIZE, HIDDEN),
            "value": matrix(base + 2, KV_HEADS * HEAD_SIZE, HIDDEN),
            "attention_output": matrix(base + 3, HIDDEN, HIDDEN),
            "gate": matrix(base + 4, INTERMEDIATE, HIDDEN),
            "up": matrix(base + 5, INTERMEDIATE, HIDDEN),
            "down": matrix(base + 6, HIDDEN, INTERMEDIATE),
        }
    )

cache = [{"keys": [], "values": []} for _ in weights]
logits = None
for position, token in enumerate([1, 2, 3]):
    hidden = embeddings[token].copy()
    for layer_index, layer in enumerate(weights):
        normalized = rms_norm(hidden)
        query = rope(layer["query"] @ normalized, position)
        key = rope(layer["key"] @ normalized, position)
        value = (layer["value"] @ normalized).reshape(KV_HEADS, HEAD_SIZE)
        cache[layer_index]["keys"].append(key)
        cache[layer_index]["values"].append(value)
        attended = np.zeros(HIDDEN, dtype=np.float32)
        for head in range(HEADS):
            kv_head = head // (HEADS // KV_HEADS)
            old_keys = np.stack(cache[layer_index]["keys"])
            scores = old_keys[:, kv_head, :] @ query[head] / np.sqrt(np.float32(HEAD_SIZE))
            probabilities = np.exp(scores - np.max(scores))
            probabilities /= np.sum(probabilities)
            old_values = np.stack(cache[layer_index]["values"])
            attended[head * HEAD_SIZE : (head + 1) * HEAD_SIZE] = (
                probabilities @ old_values[:, kv_head, :]
            )
        hidden += layer["attention_output"] @ attended
        normalized = rms_norm(hidden)
        hidden += layer["down"] @ (silu(layer["gate"] @ normalized) * (layer["up"] @ normalized))
    logits = embeddings @ rms_norm(hidden)

assert logits is not None
print("logits:", [float(value) for value in logits])
print("greedy token:", int(np.argmax(logits)))
