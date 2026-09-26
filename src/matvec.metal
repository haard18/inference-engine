#include <metal_stdlib>
using namespace metal;

struct Params {
    uint rows;
    uint cols;
    uint kind;
    uint batch_count;
};

struct AttentionParams {
    uint head_count;
    uint kv_head_count;
    uint head_size;
    uint kv_size;
    uint sequence_length;
};

struct RopeParams {
    uint head_count;
    uint kv_head_count;
    uint head_size;
    uint kv_size;
    uint position;
    uint interleaved;
};

struct NormParams {
    uint count;
    float epsilon;
};

struct BatchNormParams {
    uint width;
    uint batch_count;
    float epsilon;
};

struct BatchAttentionParams {
    uint head_count;
    uint kv_head_count;
    uint head_size;
    uint kv_size;
    uint first_position;
    uint batch_count;
    uint max_sequence_length;
    uint interleaved;
};

static float half_at(device const uchar *bytes, uint offset) {
    ushort bits = ushort(bytes[offset]) | (ushort(bytes[offset + 1]) << 8);
    return float(as_type<half>(bits));
}

static void q4_scale_min(device const uchar *scales, uint index, thread uint &scale, thread uint &minimum) {
    if (index < 4) {
        scale = uint(scales[index] & 63);
        minimum = uint(scales[index + 4] & 63);
    } else {
        scale = uint(scales[index + 4] & 15) | (uint(scales[index - 4] >> 6) << 4);
        minimum = uint(scales[index + 4] >> 4) | (uint(scales[index] >> 6) << 4);
    }
}

kernel void matvec(
    device const uchar *weights [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant Params &params [[buffer(3)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= params.rows) return;
    float sum = 0.0f;
    uint cols = params.cols;
    if (params.kind == 0) {
        device const float *values = reinterpret_cast<device const float *>(weights);
        for (uint i = 0; i < cols; ++i) sum += values[row * cols + i] * input[i];
    } else if (params.kind == 1) {
        device const half *values = reinterpret_cast<device const half *>(weights);
        for (uint i = 0; i < cols; ++i) sum += float(values[row * cols + i]) * input[i];
    } else if (params.kind == 2) {
        device const ushort *values = reinterpret_cast<device const ushort *>(weights);
        for (uint i = 0; i < cols; ++i) {
            uint bits = uint(values[row * cols + i]) << 16;
            sum += as_type<float>(bits) * input[i];
        }
    } else if (params.kind == 3) {
        device const uchar *row_bytes = weights + row * (cols / 32) * 34;
        for (uint block_index = 0; block_index < cols / 32; ++block_index) {
            device const uchar *block = row_bytes + block_index * 34;
            float scale = half_at(block, 0);
            float dot = 0.0f;
            for (uint i = 0; i < 32; ++i) {
                int quant = int(block[2 + i]);
                if (quant >= 128) quant -= 256;
                dot += float(quant) * input[block_index * 32 + i];
            }
            sum += scale * dot;
        }
    } else if (params.kind == 4) {
        device const uchar *row_bytes = weights + row * (cols / 32) * 22;
        for (uint block_index = 0; block_index < cols / 32; ++block_index) {
            device const uchar *block = row_bytes + block_index * 22;
            float scale = half_at(block, 0);
            uint high_bits = uint(block[2]) | (uint(block[3]) << 8) |
                (uint(block[4]) << 16) | (uint(block[5]) << 24);
            for (uint i = 0; i < 16; ++i) {
                uchar packed = block[6 + i];
                int low = int(packed & 15) + int((high_bits >> i) & 1) * 16 - 16;
                int high = int(packed >> 4) + int((high_bits >> (i + 16)) & 1) * 16 - 16;
                sum += (scale * float(low)) * input[block_index * 32 + i];
                sum += (scale * float(high)) * input[block_index * 32 + i + 16];
            }
        }
    } else if (params.kind == 5) {
        device const uchar *row_bytes = weights + row * (cols / 256) * 144;
        for (uint block_index = 0; block_index < cols / 256; ++block_index) {
            device const uchar *block = row_bytes + block_index * 144;
            float scale = half_at(block, 0);
            float minimum = half_at(block, 2);
            device const uchar *scales = block + 4;
            for (uint group = 0; group < 4; ++group) {
                uint low_scale, low_minimum, high_scale, high_minimum;
                q4_scale_min(scales, group * 2, low_scale, low_minimum);
                q4_scale_min(scales, group * 2 + 1, high_scale, high_minimum);
                for (uint i = 0; i < 32; ++i) {
                    uchar packed = block[16 + group * 32 + i];
                    float low = (scale * float(low_scale)) * float(packed & 15) - minimum * float(low_minimum);
                    float high = (scale * float(high_scale)) * float(packed >> 4) - minimum * float(high_minimum);
                    sum += low * input[block_index * 256 + group * 64 + i];
                    sum += high * input[block_index * 256 + group * 64 + 32 + i];
                }
            }
        }
    } else if (params.kind == 6) {
        device const uchar *row_bytes = weights + row * (cols / 256) * 210;
        for (uint block_index = 0; block_index < cols / 256; ++block_index) {
            device const uchar *block = row_bytes + block_index * 210;
            float scale = half_at(block, 208);
            for (uint segment = 0; segment < 2; ++segment) {
                for (uint i = 0; i < 32; ++i) {
                    uchar low_first = block[segment * 64 + i];
                    uchar low_second = block[segment * 64 + i + 32];
                    uchar high = block[128 + segment * 32 + i];
                    uint scale_index = 192 + segment * 8 + i / 16;
                    uint quants[4] = {
                        uint(low_first & 15) | (uint(high & 3) << 4),
                        uint(low_second & 15) | (uint((high >> 2) & 3) << 4),
                        uint(low_first >> 4) | (uint((high >> 4) & 3) << 4),
                        uint(low_second >> 4) | (uint(high >> 6) << 4),
                    };
                    for (uint group = 0; group < 4; ++group) {
                        int group_scale = int(block[scale_index + group * 2]);
                        if (group_scale >= 128) group_scale -= 256;
                        uint input_index = block_index * 256 + segment * 128 + group * 32 + i;
                        sum += (scale * float(group_scale) * (float(quants[group]) - 32.0f)) * input[input_index];
                    }
                }
            }
        }
    }
    output[row] = sum;
}

// Process known prompt-token vectors together. Adjacent token threads read
// the same row of weights, allowing the GPU cache to share those reads.
kernel void matvec_batch(
    device const uchar *weights [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant Params &params [[buffer(3)]],
    uint2 cell [[thread_position_in_grid]]) {
    uint row = cell.x;
    uint token = cell.y;
    if (row >= params.rows || token >= params.batch_count) return;
    device const float *token_input = input + token * params.cols;
    float sum = 0.0f;
    uint cols = params.cols;
    if (params.kind == 0) {
        device const float *values = reinterpret_cast<device const float *>(weights);
        for (uint i = 0; i < cols; ++i) sum += values[row * cols + i] * token_input[i];
    } else if (params.kind == 1) {
        device const half *values = reinterpret_cast<device const half *>(weights);
        for (uint i = 0; i < cols; ++i) sum += float(values[row * cols + i]) * token_input[i];
    } else if (params.kind == 2) {
        device const ushort *values = reinterpret_cast<device const ushort *>(weights);
        for (uint i = 0; i < cols; ++i) {
            uint bits = uint(values[row * cols + i]) << 16;
            sum += as_type<float>(bits) * token_input[i];
        }
    } else if (params.kind == 3) {
        device const uchar *row_bytes = weights + row * (cols / 32) * 34;
        for (uint block_index = 0; block_index < cols / 32; ++block_index) {
            device const uchar *block = row_bytes + block_index * 34;
            float scale = half_at(block, 0);
            float dot = 0.0f;
            for (uint i = 0; i < 32; ++i) {
                int quant = int(block[2 + i]);
                if (quant >= 128) quant -= 256;
                dot += float(quant) * token_input[block_index * 32 + i];
            }
            sum += scale * dot;
        }
    } else if (params.kind == 4) {
        device const uchar *row_bytes = weights + row * (cols / 32) * 22;
        for (uint block_index = 0; block_index < cols / 32; ++block_index) {
            device const uchar *block = row_bytes + block_index * 22;
            float scale = half_at(block, 0);
            uint high_bits = uint(block[2]) | (uint(block[3]) << 8) |
                (uint(block[4]) << 16) | (uint(block[5]) << 24);
            for (uint i = 0; i < 16; ++i) {
                uchar packed = block[6 + i];
                int low = int(packed & 15) + int((high_bits >> i) & 1) * 16 - 16;
                int high = int(packed >> 4) + int((high_bits >> (i + 16)) & 1) * 16 - 16;
                sum += (scale * float(low)) * token_input[block_index * 32 + i];
                sum += (scale * float(high)) * token_input[block_index * 32 + i + 16];
            }
        }
    } else if (params.kind == 5) {
        device const uchar *row_bytes = weights + row * (cols / 256) * 144;
        for (uint block_index = 0; block_index < cols / 256; ++block_index) {
            device const uchar *block = row_bytes + block_index * 144;
            float scale = half_at(block, 0);
            float minimum = half_at(block, 2);
            device const uchar *scales = block + 4;
            for (uint group = 0; group < 4; ++group) {
                uint low_scale, low_minimum, high_scale, high_minimum;
                q4_scale_min(scales, group * 2, low_scale, low_minimum);
                q4_scale_min(scales, group * 2 + 1, high_scale, high_minimum);
                for (uint i = 0; i < 32; ++i) {
                    uchar packed = block[16 + group * 32 + i];
                    float low = (scale * float(low_scale)) * float(packed & 15) - minimum * float(low_minimum);
                    float high = (scale * float(high_scale)) * float(packed >> 4) - minimum * float(high_minimum);
                    sum += low * token_input[block_index * 256 + group * 64 + i];
                    sum += high * token_input[block_index * 256 + group * 64 + 32 + i];
                }
            }
        }
    } else if (params.kind == 6) {
        device const uchar *row_bytes = weights + row * (cols / 256) * 210;
        for (uint block_index = 0; block_index < cols / 256; ++block_index) {
            device const uchar *block = row_bytes + block_index * 210;
            float scale = half_at(block, 208);
            for (uint segment = 0; segment < 2; ++segment) {
                for (uint i = 0; i < 32; ++i) {
                    uchar low_first = block[segment * 64 + i];
                    uchar low_second = block[segment * 64 + i + 32];
                    uchar high = block[128 + segment * 32 + i];
                    uint scale_index = 192 + segment * 8 + i / 16;
                    uint quants[4] = {
                        uint(low_first & 15) | (uint(high & 3) << 4),
                        uint(low_second & 15) | (uint((high >> 2) & 3) << 4),
                        uint(low_first >> 4) | (uint((high >> 4) & 3) << 4),
                        uint(low_second >> 4) | (uint(high >> 6) << 4),
                    };
                    for (uint group = 0; group < 4; ++group) {
                        int group_scale = int(block[scale_index + group * 2]);
                        if (group_scale >= 128) group_scale -= 256;
                        uint input_index = block_index * 256 + segment * 128 + group * 32 + i;
                        sum += (scale * float(group_scale) * (float(quants[group]) - 32.0f)) * token_input[input_index];
                    }
                }
            }
        }
    }
    output[token * params.rows + row] = sum;
}

kernel void silu_multiply(
    device const float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    float value = gate[index];
    output[index] = (value / (1.0f + exp(-value))) * up[index];
}

kernel void rms_scale(
    device const float *input [[buffer(0)]],
    device float *scale [[buffer(1)]],
    constant NormParams &params [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index != 0) return;
    float sum = 0.0f;
    for (uint i = 0; i < params.count; ++i) sum += input[i] * input[i];
    scale[0] = 1.0f / sqrt(sum / float(params.count) + params.epsilon);
}

kernel void rms_apply(
    device const float *input [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device const float *scale [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &count [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    output[index] = input[index] * scale[0] * weights[index];
}

kernel void add_vectors(
    device const float *left [[buffer(0)]],
    device const float *right [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    output[index] = left[index] + right[index];
}

kernel void rotate_and_store(
    device float *query [[buffer(0)]],
    device const float *key [[buffer(1)]],
    device const float *value [[buffer(2)]],
    device float *cached_keys [[buffer(3)]],
    device float *cached_values [[buffer(4)]],
    device const float *rotations [[buffer(5)]],
    constant RopeParams &params [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    uint half_width = params.head_size / 2;
    uint query_pairs = params.head_count * half_width;
    uint key_pairs = params.kv_head_count * half_width;
    uint slot = params.position * params.kv_size;
    if (index < query_pairs) {
        uint head = index / half_width;
        uint pair = index % half_width;
        uint base = head * params.head_size;
        uint first_index = base + (params.interleaved != 0 ? pair * 2 : pair);
        uint second_index = base + (params.interleaved != 0 ? pair * 2 + 1 : pair + half_width);
        float first = query[first_index];
        float second = query[second_index];
        float sine = rotations[pair * 2];
        float cosine = rotations[pair * 2 + 1];
        query[first_index] = first * cosine - second * sine;
        query[second_index] = second * cosine + first * sine;
    } else if (index < query_pairs + key_pairs) {
        uint key_pair = index - query_pairs;
        uint head = key_pair / half_width;
        uint pair = key_pair % half_width;
        uint base = head * params.head_size;
        uint first_index = base + (params.interleaved != 0 ? pair * 2 : pair);
        uint second_index = base + (params.interleaved != 0 ? pair * 2 + 1 : pair + half_width);
        float first = key[first_index];
        float second = key[second_index];
        float sine = rotations[pair * 2];
        float cosine = rotations[pair * 2 + 1];
        cached_keys[slot + first_index] = first * cosine - second * sine;
        cached_keys[slot + second_index] = second * cosine + first * sine;
    }
    if (index < params.kv_size) {
        cached_values[slot + index] = value[index];
    }
}

kernel void attention_scores(
    device const float *query [[buffer(0)]],
    device const float *keys [[buffer(1)]],
    device float *scores [[buffer(2)]],
    constant AttentionParams &params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    uint total = params.head_count * params.sequence_length;
    if (index >= total) return;
    uint head = index / params.sequence_length;
    uint step = index % params.sequence_length;
    uint kv_head = head / (params.head_count / params.kv_head_count);
    uint query_start = head * params.head_size;
    uint key_start = step * params.kv_size + kv_head * params.head_size;
    float dot = 0.0f;
    for (uint i = 0; i < params.head_size; ++i) {
        dot += query[query_start + i] * keys[key_start + i];
    }
    scores[index] = dot * rsqrt(float(params.head_size));
}

kernel void attention_reduce(
    device const float *scores [[buffer(0)]],
    device const float *values [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant AttentionParams &params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    uint hidden_size = params.head_count * params.head_size;
    if (index >= hidden_size) return;
    uint head = index / params.head_size;
    uint offset = index % params.head_size;
    uint kv_head = head / (params.head_count / params.kv_head_count);
    uint value_offset = kv_head * params.head_size + offset;
    uint score_start = head * params.sequence_length;
    float maximum = -INFINITY;
    for (uint step = 0; step < params.sequence_length; ++step) {
        maximum = max(maximum, scores[score_start + step]);
    }
    float denominator = 0.0f;
    float numerator = 0.0f;
    for (uint step = 0; step < params.sequence_length; ++step) {
        float weight = exp(scores[score_start + step] - maximum);
        denominator += weight;
        numerator += weight * values[step * params.kv_size + value_offset];
    }
    output[index] = numerator / denominator;
}

kernel void rms_scale_batch(
    device const float *input [[buffer(0)]],
    device float *scale [[buffer(1)]],
    constant BatchNormParams &params [[buffer(2)]],
    uint token [[thread_position_in_grid]]) {
    if (token >= params.batch_count) return;
    float sum = 0.0f;
    for (uint i = 0; i < params.width; ++i) {
        float value = input[token * params.width + i];
        sum += value * value;
    }
    scale[token] = 1.0f / sqrt(sum / float(params.width) + params.epsilon);
}

kernel void rms_apply_batch(
    device const float *input [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device const float *scale [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant BatchNormParams &params [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= params.batch_count * params.width) return;
    uint token = index / params.width;
    uint column = index % params.width;
    output[index] = input[index] * scale[token] * weights[column];
}

kernel void rotate_and_store_batch(
    device float *query [[buffer(0)]],
    device const float *key [[buffer(1)]],
    device const float *value [[buffer(2)]],
    device float *cached_keys [[buffer(3)]],
    device float *cached_values [[buffer(4)]],
    device const float *rotations [[buffer(5)]],
    constant BatchAttentionParams &params [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    uint half_width = params.head_size / 2;
    uint query_pairs = params.head_count * half_width;
    uint key_pairs = params.kv_head_count * half_width;
    uint span = max(query_pairs + key_pairs, params.kv_size);
    uint token = index / span;
    uint local = index % span;
    if (token >= params.batch_count) return;
    uint position = params.first_position + token;
    uint cache_slot = position * params.kv_size;
    uint query_offset = token * params.head_count * params.head_size;
    uint kv_offset = token * params.kv_size;
    uint rotation_offset = token * params.head_size;
    if (local < query_pairs) {
        uint head = local / half_width;
        uint pair = local % half_width;
        uint base = query_offset + head * params.head_size;
        uint first = base + (params.interleaved != 0 ? pair * 2 : pair);
        uint second = base + (params.interleaved != 0 ? pair * 2 + 1 : pair + half_width);
        float left = query[first];
        float right = query[second];
        float sine = rotations[rotation_offset + pair * 2];
        float cosine = rotations[rotation_offset + pair * 2 + 1];
        query[first] = left * cosine - right * sine;
        query[second] = right * cosine + left * sine;
    } else if (local < query_pairs + key_pairs) {
        uint key_pair = local - query_pairs;
        uint head = key_pair / half_width;
        uint pair = key_pair % half_width;
        uint base = kv_offset + head * params.head_size;
        uint first = base + (params.interleaved != 0 ? pair * 2 : pair);
        uint second = base + (params.interleaved != 0 ? pair * 2 + 1 : pair + half_width);
        float left = key[first];
        float right = key[second];
        float sine = rotations[rotation_offset + pair * 2];
        float cosine = rotations[rotation_offset + pair * 2 + 1];
        cached_keys[cache_slot + head * params.head_size +
            (params.interleaved != 0 ? pair * 2 : pair)] = left * cosine - right * sine;
        cached_keys[cache_slot + head * params.head_size +
            (params.interleaved != 0 ? pair * 2 + 1 : pair + half_width)] =
            right * cosine + left * sine;
    }
    if (local < params.kv_size) {
        cached_values[cache_slot + local] = value[kv_offset + local];
    }
}

kernel void attention_scores_batch(
    device const float *query [[buffer(0)]],
    device const float *keys [[buffer(1)]],
    device float *scores [[buffer(2)]],
    constant BatchAttentionParams &params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    uint per_token = params.head_count * params.max_sequence_length;
    uint token = index / per_token;
    uint within = index % per_token;
    if (token >= params.batch_count) return;
    uint head = within / params.max_sequence_length;
    uint step = within % params.max_sequence_length;
    uint sequence_length = params.first_position + token + 1;
    if (step >= sequence_length) return;
    uint kv_head = head / (params.head_count / params.kv_head_count);
    uint query_start = token * params.head_count * params.head_size + head * params.head_size;
    uint key_start = step * params.kv_size + kv_head * params.head_size;
    float dot = 0.0f;
    for (uint i = 0; i < params.head_size; ++i) {
        dot += query[query_start + i] * keys[key_start + i];
    }
    scores[index] = dot * rsqrt(float(params.head_size));
}

kernel void attention_reduce_batch(
    device const float *scores [[buffer(0)]],
    device const float *values [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant BatchAttentionParams &params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    uint hidden_size = params.head_count * params.head_size;
    uint token = index / hidden_size;
    uint within = index % hidden_size;
    if (token >= params.batch_count) return;
    uint head = within / params.head_size;
    uint offset = within % params.head_size;
    uint kv_head = head / (params.head_count / params.kv_head_count);
    uint value_offset = kv_head * params.head_size + offset;
    uint score_start = (token * params.head_count + head) * params.max_sequence_length;
    uint sequence_length = params.first_position + token + 1;
    float maximum = -INFINITY;
    for (uint step = 0; step < sequence_length; ++step) {
        maximum = max(maximum, scores[score_start + step]);
    }
    float denominator = 0.0f;
    float numerator = 0.0f;
    for (uint step = 0; step < sequence_length; ++step) {
        float weight = exp(scores[score_start + step] - maximum);
        denominator += weight;
        numerator += weight * values[step * params.kv_size + value_offset];
    }
    output[index] = numerator / denominator;
}
