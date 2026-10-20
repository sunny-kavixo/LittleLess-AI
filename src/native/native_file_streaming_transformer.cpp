#include "native_file_streaming_transformer.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <chrono>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <arm_neon.h>

// ---------------------------------------------------------------------------
// CRC32 IEEE 802.3
// ---------------------------------------------------------------------------
static uint32_t crc32_table[256];
static bool crc32_table_init = false;

static void init_crc32_table() {
    if (crc32_table_init) return;
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int j = 0; j < 8; j++) {
            c = (c & 1) ? (0xEDB88320L ^ (c >> 1)) : (c >> 1);
        }
        crc32_table[i] = c;
    }
    crc32_table_init = true;
}

static uint32_t compute_crc32(const uint8_t* buf, size_t len) {
    init_crc32_table();
    uint32_t c = 0xFFFFFFFF;
    for (size_t i = 0; i < len; i++) {
        c = crc32_table[(c ^ buf[i]) & 0xFF] ^ (c >> 8);
    }
    return c ^ 0xFFFFFFFF;
}

// ---------------------------------------------------------------------------
// 256-Entry Lookup Tables for NEON / Radix-3 GEMV
// ---------------------------------------------------------------------------
alignas(64) static float TRIT_2BIT_FLOAT_LUT[256][4];
static bool lut_2bit_initialized = false;

static void init_2bit_trit_lut() {
    if (lut_2bit_initialized) return;
    for (int b = 0; b < 256; ++b) {
        int c0 = (((b >> 0) & 1) << 1) | ((b >> 1) & 1);
        int c1 = (((b >> 2) & 1) << 1) | ((b >> 3) & 1);
        int c2 = (((b >> 4) & 1) << 1) | ((b >> 5) & 1);
        int c3 = (((b >> 6) & 1) << 1) | ((b >> 7) & 1);
        TRIT_2BIT_FLOAT_LUT[b][0] = static_cast<float>(c0 - 1);
        TRIT_2BIT_FLOAT_LUT[b][1] = static_cast<float>(c1 - 1);
        TRIT_2BIT_FLOAT_LUT[b][2] = static_cast<float>(c2 - 1);
        TRIT_2BIT_FLOAT_LUT[b][3] = static_cast<float>(c3 - 1);
    }
    lut_2bit_initialized = true;
}

alignas(64) static float RADIX3_FLOAT_LUT[256][5];
static bool lut_radix3_initialized = false;

static void init_radix3_lut() {
    if (lut_radix3_initialized) return;
    for (int b = 0; b < 256; ++b) {
        int temp = b;
        int c0 = temp % 3; temp /= 3;
        int c1 = temp % 3; temp /= 3;
        int c2 = temp % 3; temp /= 3;
        int c3 = temp % 3; temp /= 3;
        int c4 = temp % 3;
        RADIX3_FLOAT_LUT[b][0] = static_cast<float>(c0 - 1);
        RADIX3_FLOAT_LUT[b][1] = static_cast<float>(c1 - 1);
        RADIX3_FLOAT_LUT[b][2] = static_cast<float>(c2 - 1);
        RADIX3_FLOAT_LUT[b][3] = static_cast<float>(c3 - 1);
        RADIX3_FLOAT_LUT[b][4] = static_cast<float>(c4 - 1);
    }
    lut_radix3_initialized = true;
}

// ---------------------------------------------------------------------------
// ARM NEON Vectorized Direct 2-Bit Packed GEMV
// ---------------------------------------------------------------------------
static inline void direct_packed_gemv_neon(
    const uint8_t* __restrict__ p1,
    const uint8_t* __restrict__ p2,
    const float* __restrict__ s1,
    const float* __restrict__ s2,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int N
) {
    const int bytes_per_row = N / 4;

    for (int i = 0; i < M; ++i) {
        const uint8_t* __restrict__ row_p1 = p1 + i * bytes_per_row;
        const uint8_t* __restrict__ row_p2 = p2 + i * bytes_per_row;

        float32x4_t vacc1 = vdupq_n_f32(0.0f);
        float32x4_t vacc2 = vdupq_n_f32(0.0f);

        int jb = 0;
        for (; jb + 2 <= bytes_per_row; jb += 2) {
            int j = jb * 4;
            float32x4_t vx0 = vld1q_f32(x + j);
            float32x4_t vt1_0 = vld1q_f32(TRIT_2BIT_FLOAT_LUT[row_p1[jb]]);
            float32x4_t vt2_0 = vld1q_f32(TRIT_2BIT_FLOAT_LUT[row_p2[jb]]);
            vacc1 = vmlaq_f32(vacc1, vx0, vt1_0);
            vacc2 = vmlaq_f32(vacc2, vx0, vt2_0);

            float32x4_t vx1 = vld1q_f32(x + j + 4);
            float32x4_t vt1_1 = vld1q_f32(TRIT_2BIT_FLOAT_LUT[row_p1[jb + 1]]);
            float32x4_t vt2_1 = vld1q_f32(TRIT_2BIT_FLOAT_LUT[row_p2[jb + 1]]);
            vacc1 = vmlaq_f32(vacc1, vx1, vt1_1);
            vacc2 = vmlaq_f32(vacc2, vx1, vt2_1);
        }

        for (; jb < bytes_per_row; ++jb) {
            int j = jb * 4;
            float32x4_t vx0 = vld1q_f32(x + j);
            float32x4_t vt1_0 = vld1q_f32(TRIT_2BIT_FLOAT_LUT[row_p1[jb]]);
            float32x4_t vt2_0 = vld1q_f32(TRIT_2BIT_FLOAT_LUT[row_p2[jb]]);
            vacc1 = vmlaq_f32(vacc1, vx0, vt1_0);
            vacc2 = vmlaq_f32(vacc2, vx0, vt2_0);
        }

        float sum1 = vaddvq_f32(vacc1);
        float sum2 = vaddvq_f32(vacc2);
        y[i] = s1[i] * sum1 + s2[i] * sum2;
    }
}

// ---------------------------------------------------------------------------
// Row-Aligned Direct Radix-3 GEMV
// ---------------------------------------------------------------------------
static inline void direct_radix3_gemv(
    const uint8_t* __restrict__ p1,
    const uint8_t* __restrict__ p2,
    const float* __restrict__ s1,
    const float* __restrict__ s2,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int N
) {
    const int bytes_per_row = (N + 4) / 5;

    for (int i = 0; i < M; ++i) {
        const uint8_t* __restrict__ row_p1 = p1 + i * bytes_per_row;
        const uint8_t* __restrict__ row_p2 = p2 + i * bytes_per_row;

        float sum1 = 0.0f;
        float sum2 = 0.0f;

        for (int jb = 0; jb < bytes_per_row; ++jb) {
            int j = jb * 5;
            const float* t1 = RADIX3_FLOAT_LUT[row_p1[jb]];
            const float* t2 = RADIX3_FLOAT_LUT[row_p2[jb]];

            sum1 += t1[0] * x[j + 0] + t1[1] * x[j + 1] + t1[2] * x[j + 2] + t1[3] * x[j + 3] + (j + 4 < N ? t1[4] * x[j + 4] : 0.0f);
            sum2 += t2[0] * x[j + 0] + t2[1] * x[j + 1] + t2[2] * x[j + 2] + t2[3] * x[j + 3] + (j + 4 < N ? t2[4] * x[j + 4] : 0.0f);
        }

        y[i] = s1[i] * sum1 + s2[i] * sum2;
    }
}

// ---------------------------------------------------------------------------
// LayerNorm & GELU
// ---------------------------------------------------------------------------
static inline void layer_norm(
    float* __restrict__ out,
    const float* __restrict__ in,
    const float* __restrict__ w,
    const float* __restrict__ b,
    int C,
    float eps = 1e-5f
) {
    float sum = 0.0f;
    for (int i = 0; i < C; ++i) sum += in[i];
    float mean = sum / (float)C;

    float var_sum = 0.0f;
    for (int i = 0; i < C; ++i) {
        float diff = in[i] - mean;
        var_sum += diff * diff;
    }
    float inv_std = 1.0f / std::sqrt(var_sum / (float)C + eps);

    for (int i = 0; i < C; ++i) {
        out[i] = (in[i] - mean) * inv_std * w[i] + b[i];
    }
}

static inline void gelu_inplace(float* __restrict__ x, int size) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    for (int i = 0; i < size; ++i) {
        float val = x[i];
        float inner = sqrt_2_over_pi * (val + 0.044715f * val * val * val);
        x[i] = 0.5f * val * (1.0f + std::tanh(inner));
    }
}

// ---------------------------------------------------------------------------
// Robust POSIX pread with Short-Read Retry & Error Handling
// ---------------------------------------------------------------------------
static bool robust_pread(int fd, void* dest, size_t count, off_t offset) {
    uint8_t* ptr = static_cast<uint8_t*>(dest);
    size_t total_read = 0;
    while (total_read < count) {
        ssize_t n = ::pread(fd, ptr + total_read, count - total_read, offset + total_read);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) return false; // Unexpected EOF
        total_read += n;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Streamed Layer Index Metadata
// ---------------------------------------------------------------------------
struct StreamedLayerIndex {
    int32_t block_idx = 0; // 0..3, or -1 for head
    int32_t layer_type = 0; // 0=q, 1=k, 2=v, 3=out, 4=fc1, 5=fc2, 6=head
    uint32_t M = 0;
    uint32_t N = 0;
    uint64_t p1_offset = 0;
    uint32_t p1_bytes = 0;
    uint32_t p1_crc32 = 0;
    uint64_t p2_offset = 0;
    uint32_t p2_bytes = 0;
    uint32_t p2_crc32 = 0;
    std::vector<float> s1;
    std::vector<float> s2;
};

struct FileStreamingBlockMeta {
    std::vector<float> ln1_w, ln1_b;
    std::vector<float> ln2_w, ln2_b;
    std::vector<float> fc1_b, fc2_b;

    // Persistent KV Cache in RAM (Context=128, Heads=5, Dim=32)
    alignas(64) float k_cache[5][128][32];
    alignas(64) float v_cache[5][128][32];

    void clear_cache() {
        std::memset(k_cache, 0, sizeof(k_cache));
        std::memset(v_cache, 0, sizeof(v_cache));
    }
};

struct NativeFileStreamingModel {
    int fd = -1; // Owned file descriptor to .llstream container
    int pack_mode = STREAM_PACK_RADIX3;
    int stream_level = STREAM_LEVEL_LAYER; // 0=Block, 1=Layer

    int d_model = 160;
    int n_heads = 5;
    int d_head = 32;
    int n_blocks = 4;
    int max_context = 128;
    int vocab_size = 256;
    int num_streamed_layers = 25;

    // Persistent Parameters in Native C++ RAM
    std::vector<int8_t> wte_q;
    float wte_s = 0.0f;
    std::vector<int8_t> wpe_q;
    float wpe_s = 0.0f;

    std::vector<FileStreamingBlockMeta> blocks;
    std::vector<float> ln_f_w, ln_f_b;

    // Streamed Weight Index (25 layers: 24 block projections + 1 head)
    std::vector<StreamedLayerIndex> layer_indices;

    // TRANSIENT REUSABLE SCRATCHPAD BUFFERS
    // In Layer mode: holds max projection mlp_fc1 (512x160):
    //   Radix-3: 16,480 B per plane = 32,960 B total
    //   2-Bit:   20,480 B per plane = 40,960 B total
    // In Block mode: holds all 6 projections of 1 block:
    //   Radix-3: 53,344 B per plane = 106,688 B total
    //   2-Bit:   66,560 B per plane = 133,120 B total
    alignas(64) std::vector<uint8_t> transient_p1_scratch;
    alignas(64) std::vector<uint8_t> transient_p2_scratch;
    size_t scratchpad_allocated_bytes = 0;

    // I/O Telemetry Counters
    uint64_t pread_calls = 0;
    uint64_t bytes_streamed = 0;

    // Active Inference Activation Buffers
    alignas(64) float x[160];
    alignas(64) float x_norm1[160];
    alignas(64) float q_buf[160];
    alignas(64) float k_buf[160];
    alignas(64) float v_buf[160];
    alignas(64) float att_scores[128];
    alignas(64) float y_attn[160];
    alignas(64) float x_norm2[160];
    alignas(64) float h_mlp[512];
    alignas(64) float out_mlp[160];
    alignas(64) float x_f[160];

    NativeFileStreamingModel(int p_mode, int s_level) : pack_mode(p_mode), stream_level(s_level) {
        blocks.resize(n_blocks);
        init_2bit_trit_lut();
        init_radix3_lut();
        init_crc32_table();

        if (stream_level == STREAM_LEVEL_BLOCK) {
            size_t per_plane = (pack_mode == STREAM_PACK_RADIX3) ? 53344 : 66560;
            transient_p1_scratch.resize(per_plane);
            transient_p2_scratch.resize(per_plane);
            scratchpad_allocated_bytes = per_plane * 2;
        } else {
            size_t per_plane = (pack_mode == STREAM_PACK_RADIX3) ? 16480 : 20480;
            transient_p1_scratch.resize(per_plane);
            transient_p2_scratch.resize(per_plane);
            scratchpad_allocated_bytes = per_plane * 2;
        }
    }

    ~NativeFileStreamingModel() {
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
    }

    void clear_kv_caches() {
        for (auto& b : blocks) b.clear_cache();
    }

    // Streams a single linear projection Just-In-Time from file descriptor into transient scratchpad
    bool execute_streaming_linear(
        int layer_idx,
        const float* in_vec,
        float* out_vec,
        const float* opt_bias = nullptr
    ) {
        const auto& meta = layer_indices[layer_idx];

        // 1. Read p1 and p2 planes directly from file into transient scratchpad
        if (!robust_pread(fd, transient_p1_scratch.data(), meta.p1_bytes, meta.p1_offset)) {
            return false;
        }
        if (!robust_pread(fd, transient_p2_scratch.data(), meta.p2_bytes, meta.p2_offset)) {
            return false;
        }

        pread_calls += 2;
        bytes_streamed += (meta.p1_bytes + meta.p2_bytes);

        // 2. Perform GEMV directly from scratchpad
        if (pack_mode == STREAM_PACK_RADIX3) {
            direct_radix3_gemv(
                transient_p1_scratch.data(),
                transient_p2_scratch.data(),
                meta.s1.data(),
                meta.s2.data(),
                in_vec,
                out_vec,
                meta.M,
                meta.N
            );
        } else {
            direct_packed_gemv_neon(
                transient_p1_scratch.data(),
                transient_p2_scratch.data(),
                meta.s1.data(),
                meta.s2.data(),
                in_vec,
                out_vec,
                meta.M,
                meta.N
            );
        }

        if (opt_bias != nullptr) {
            for (int i = 0; i < meta.M; ++i) out_vec[i] += opt_bias[i];
        }
        return true;
    }

    // Evaluates single token forward pass
    bool forward_token(int token_id, int pos, float* out_logits) {
        // 1. Embeddings (Persistent native parameters)
        const int8_t* wte_row = wte_q.data() + token_id * d_model;
        const int8_t* wpe_row = wpe_q.data() + pos * d_model;
        for (int i = 0; i < d_model; ++i) {
            x[i] = (float)wte_row[i] * wte_s + (float)wpe_row[i] * wpe_s;
        }

        const float scale_factor = 1.0f / std::sqrt((float)d_head);

        // 2. Sequential Blocks with True File-Backed JIT Streaming
        for (int bi = 0; bi < n_blocks; ++bi) {
            auto& b = blocks[bi];
            int base_layer = bi * 6;

            // Attention LN1
            layer_norm(x_norm1, x, b.ln1_w.data(), b.ln1_b.data(), d_model);

            // Stream and evaluate Q, K, V directly from disk container
            if (!execute_streaming_linear(base_layer + 0, x_norm1, q_buf)) return false;
            if (!execute_streaming_linear(base_layer + 1, x_norm1, k_buf)) return false;
            if (!execute_streaming_linear(base_layer + 2, x_norm1, v_buf)) return false;

            // Update persistent KV Cache
            for (int h = 0; h < n_heads; ++h) {
                std::memcpy(&b.k_cache[h][pos][0], k_buf + (h * d_head), sizeof(float) * d_head);
                std::memcpy(&b.v_cache[h][pos][0], v_buf + (h * d_head), sizeof(float) * d_head);
            }

            // Multi-head self-attention
            for (int h = 0; h < n_heads; ++h) {
                const float* q_h = q_buf + h * d_head;
                float* y_h = y_attn + h * d_head;

                float max_score = -1e9f;
                for (int t = 0; t <= pos; ++t) {
                    const float* k_t = &b.k_cache[h][t][0];
                    float dot = 0.0f;
                    for (int d = 0; d < d_head; ++d) {
                        dot += q_h[d] * k_t[d];
                    }
                    float sc = dot * scale_factor;
                    att_scores[t] = sc;
                    if (sc > max_score) max_score = sc;
                }

                float exp_sum = 0.0f;
                for (int t = 0; t <= pos; ++t) {
                    float exp_val = std::exp(att_scores[t] - max_score);
                    att_scores[t] = exp_val;
                    exp_sum += exp_val;
                }
                float inv_exp_sum = 1.0f / (exp_sum + 1e-12f);
                for (int t = 0; t <= pos; ++t) {
                    att_scores[t] *= inv_exp_sum;
                }

                std::memset(y_h, 0, sizeof(float) * d_head);
                for (int t = 0; t <= pos; ++t) {
                    float a = att_scores[t];
                    const float* v_t = &b.v_cache[h][t][0];
                    for (int d = 0; d < d_head; ++d) {
                        y_h[d] += a * v_t[d];
                    }
                }
            }

            // Stream Out Proj (Layer 3)
            if (!execute_streaming_linear(base_layer + 3, y_attn, x_norm2)) return false;
            for (int i = 0; i < d_model; ++i) x[i] += x_norm2[i];

            // MLP LN2
            layer_norm(x_norm2, x, b.ln2_w.data(), b.ln2_b.data(), d_model);

            // Stream FC1 (Layer 4) with bias
            if (!execute_streaming_linear(base_layer + 4, x_norm2, h_mlp, b.fc1_b.data())) return false;
            gelu_inplace(h_mlp, 512);

            // Stream FC2 (Layer 5) with bias
            if (!execute_streaming_linear(base_layer + 5, h_mlp, out_mlp, b.fc2_b.data())) return false;
            for (int i = 0; i < d_model; ++i) x[i] += out_mlp[i];
        }

        // 3. Final LN & LM Head (Streamed from disk)
        layer_norm(x_f, x, ln_f_w.data(), ln_f_b.data(), d_model);
        if (!execute_streaming_linear(24, x_f, out_logits)) return false; // Layer 24 = LM Head

        return true;
    }
};

extern "C" {

NativeFileStreamingModel* create_file_streaming_model(
    const char* container_path,
    int stream_level,
    int* out_status
) {
    if (!container_path) {
        if (out_status) *out_status = STREAM_ERR_PARAM;
        return nullptr;
    }

    int fd = ::open(container_path, O_RDONLY);
    if (fd < 0) {
        if (out_status) *out_status = STREAM_ERR_OPEN;
        return nullptr;
    }

    // 1. Read and validate Header (64 bytes)
    char magic[8];
    if (!robust_pread(fd, magic, 8, 0) || std::memcmp(magic, "LLSTRM01", 8) != 0) {
        ::close(fd);
        if (out_status) *out_status = STREAM_ERR_MAGIC;
        return nullptr;
    }

    uint32_t header_vals[8]; // version, pack_mode, d_model, n_heads, d_head, n_blocks, max_context, vocab_size
    if (!robust_pread(fd, header_vals, sizeof(header_vals), 8)) {
        ::close(fd);
        if (out_status) *out_status = STREAM_ERR_READ;
        return nullptr;
    }

    uint32_t version = header_vals[0];
    if (version != 1) {
        ::close(fd);
        if (out_status) *out_status = STREAM_ERR_VERSION;
        return nullptr;
    }

    int pack_mode = header_vals[1];
    NativeFileStreamingModel* model = new NativeFileStreamingModel(pack_mode, stream_level);
    model->fd = fd;
    model->d_model = header_vals[2];
    model->n_heads = header_vals[3];
    model->d_head = header_vals[4];
    model->n_blocks = header_vals[5];
    model->max_context = header_vals[6];
    model->vocab_size = header_vals[7];

    off_t curr_off = 8 + sizeof(header_vals); // 40 bytes
    uint32_t wte_bytes = 0;
    uint32_t wpe_bytes = 0;
    uint32_t num_layers = 0;

    // 2. Read Embeddings (Persistent)
    if (!robust_pread(fd, &model->wte_s, sizeof(float), curr_off)) goto read_err;
    curr_off += sizeof(float);

    if (!robust_pread(fd, &wte_bytes, sizeof(uint32_t), curr_off)) goto read_err;
    curr_off += sizeof(uint32_t);

    model->wte_q.resize(wte_bytes);
    if (!robust_pread(fd, model->wte_q.data(), wte_bytes, curr_off)) goto read_err;
    curr_off += wte_bytes;

    if (!robust_pread(fd, &model->wpe_s, sizeof(float), curr_off)) goto read_err;
    curr_off += sizeof(float);

    if (!robust_pread(fd, &wpe_bytes, sizeof(uint32_t), curr_off)) goto read_err;
    curr_off += sizeof(uint32_t);

    model->wpe_q.resize(wpe_bytes);
    if (!robust_pread(fd, model->wpe_q.data(), wpe_bytes, curr_off)) goto read_err;
    curr_off += wpe_bytes;

    // 3. Read Persistent Block Parameters (Norms, Biases, Scales)
    for (int bi = 0; bi < model->n_blocks; ++bi) {
        auto& b = model->blocks[bi];
        b.ln1_w.resize(model->d_model);
        b.ln1_b.resize(model->d_model);
        b.ln2_w.resize(model->d_model);
        b.ln2_b.resize(model->d_model);
        b.fc1_b.resize(512);
        b.fc2_b.resize(model->d_model);

        if (!robust_pread(fd, b.ln1_w.data(), model->d_model * sizeof(float), curr_off)) goto read_err;
        curr_off += model->d_model * sizeof(float);
        if (!robust_pread(fd, b.ln1_b.data(), model->d_model * sizeof(float), curr_off)) goto read_err;
        curr_off += model->d_model * sizeof(float);
        if (!robust_pread(fd, b.ln2_w.data(), model->d_model * sizeof(float), curr_off)) goto read_err;
        curr_off += model->d_model * sizeof(float);
        if (!robust_pread(fd, b.ln2_b.data(), model->d_model * sizeof(float), curr_off)) goto read_err;
        curr_off += model->d_model * sizeof(float);
        if (!robust_pread(fd, b.fc1_b.data(), 512 * sizeof(float), curr_off)) goto read_err;
        curr_off += 512 * sizeof(float);
        if (!robust_pread(fd, b.fc2_b.data(), model->d_model * sizeof(float), curr_off)) goto read_err;
        curr_off += model->d_model * sizeof(float);
    }

    // 4. Read Final Head Norms
    model->ln_f_w.resize(model->d_model);
    model->ln_f_b.resize(model->d_model);
    if (!robust_pread(fd, model->ln_f_w.data(), model->d_model * sizeof(float), curr_off)) goto read_err;
    curr_off += model->d_model * sizeof(float);
    if (!robust_pread(fd, model->ln_f_b.data(), model->d_model * sizeof(float), curr_off)) goto read_err;
    curr_off += model->d_model * sizeof(float);

    // 5. Read Streamed Layer Index Table (25 layers)
    if (!robust_pread(fd, &num_layers, sizeof(uint32_t), curr_off)) goto read_err;
    curr_off += sizeof(uint32_t);
    model->num_streamed_layers = num_layers;
    model->layer_indices.resize(num_layers);

    for (uint32_t li = 0; li < num_layers; ++li) {
        auto& meta = model->layer_indices[li];
        uint32_t meta_ints[4]; // block_idx, layer_type, M, N
        if (!robust_pread(fd, meta_ints, sizeof(meta_ints), curr_off)) goto read_err;
        curr_off += sizeof(meta_ints);

        meta.block_idx = (int32_t)meta_ints[0];
        meta.layer_type = (int32_t)meta_ints[1];
        meta.M = meta_ints[2];
        meta.N = meta_ints[3];

        uint64_t off_data[2]; // p1_offset, p2_offset
        uint32_t size_crc[4]; // p1_bytes, p1_crc, p2_bytes, p2_crc
        if (!robust_pread(fd, off_data, sizeof(off_data), curr_off)) goto read_err;
        curr_off += sizeof(off_data);
        if (!robust_pread(fd, size_crc, sizeof(size_crc), curr_off)) goto read_err;
        curr_off += sizeof(size_crc);

        meta.p1_offset = off_data[0];
        meta.p2_offset = off_data[1];
        meta.p1_bytes = size_crc[0];
        meta.p1_crc32 = size_crc[1];
        meta.p2_bytes = size_crc[2];
        meta.p2_crc32 = size_crc[3];

        // Read scales (M floats for s1, M floats for s2)
        meta.s1.resize(meta.M);
        meta.s2.resize(meta.M);
        if (!robust_pread(fd, meta.s1.data(), meta.M * sizeof(float), curr_off)) goto read_err;
        curr_off += meta.M * sizeof(float);
        if (!robust_pread(fd, meta.s2.data(), meta.M * sizeof(float), curr_off)) goto read_err;
        curr_off += meta.M * sizeof(float);
    }

    if (out_status) *out_status = STREAM_SUCCESS;
    return model;

read_err:
    delete model;
    if (out_status) *out_status = STREAM_ERR_READ;
    return nullptr;
}

void free_file_streaming_model(NativeFileStreamingModel* model) {
    if (model) delete model;
}

size_t file_streaming_get_active_weight_ram(NativeFileStreamingModel* model) {
    if (!model) return 0;
    // Exactly persistent model weight bytes + transient scratchpad buffer
    size_t embs = model->wte_q.size() * sizeof(int8_t) + model->wpe_q.size() * sizeof(int8_t);

    size_t scales = 0;
    for (const auto& meta : model->layer_indices) {
        scales += meta.s1.size() * sizeof(float) + meta.s2.size() * sizeof(float);
    }

    size_t norms_biases = model->ln_f_w.size() * sizeof(float) + model->ln_f_b.size() * sizeof(float);
    for (int bi = 0; bi < model->n_blocks; ++bi) {
        norms_biases += model->blocks[bi].ln1_w.size() * sizeof(float) + model->blocks[bi].ln1_b.size() * sizeof(float);
        norms_biases += model->blocks[bi].ln2_w.size() * sizeof(float) + model->blocks[bi].ln2_b.size() * sizeof(float);
        norms_biases += model->blocks[bi].fc1_b.size() * sizeof(float) + model->blocks[bi].fc2_b.size() * sizeof(float);
    }

    size_t scratchpad = model->scratchpad_allocated_bytes;
    return embs + scales + norms_biases + scratchpad;
}

size_t file_streaming_get_total_model_ram(NativeFileStreamingModel* model) {
    if (!model) return 0;
    size_t weights = file_streaming_get_active_weight_ram(model);
    // KV Cache: 4 blocks * 5 heads * 128 context * 32 head_dim * 4 bytes * 2 (K+V) = 655,360 B
    size_t kv_cache = model->n_blocks * sizeof(model->blocks[0].k_cache) * 2;
    // Activation vectors
    size_t activations = sizeof(model->x) + sizeof(model->x_norm1) + sizeof(model->q_buf) +
                         sizeof(model->k_buf) + sizeof(model->v_buf) + sizeof(model->att_scores) +
                         sizeof(model->y_attn) + sizeof(model->x_norm2) + sizeof(model->h_mlp) +
                         sizeof(model->out_mlp) + sizeof(model->x_f);
    return weights + kv_cache + activations;
}

void file_streaming_get_io_stats(
    NativeFileStreamingModel* model,
    uint64_t* out_pread_calls,
    uint64_t* out_bytes_read
) {
    if (!model) return;
    if (out_pread_calls) *out_pread_calls = model->pread_calls;
    if (out_bytes_read) *out_bytes_read = model->bytes_streamed;
}

void file_streaming_reset_io_stats(NativeFileStreamingModel* model) {
    if (!model) return;
    model->pread_calls = 0;
    model->bytes_streamed = 0;
}

int file_streaming_forward_step(
    NativeFileStreamingModel* model,
    int token_id,
    int pos,
    float* logits_out
) {
    if (!model || !logits_out) return STREAM_ERR_PARAM;
    if (!model->forward_token(token_id, pos, logits_out)) return STREAM_ERR_READ;
    return STREAM_SUCCESS;
}

int file_streaming_generate_tokens(
    NativeFileStreamingModel* model,
    const int32_t* prompt,
    int prompt_len,
    int max_new_tokens,
    int32_t* out_tokens,
    double* out_stats
) {
    if (!model || !prompt || !out_tokens) return STREAM_ERR_PARAM;
    if (prompt_len <= 0 || prompt_len + max_new_tokens > model->max_context) return -1;

    model->clear_kv_caches();
    alignas(64) float logits[256];

    // Prefill Phase
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int p = 0; p < prompt_len; ++p) {
        if (!model->forward_token(prompt[p], p, logits)) return STREAM_ERR_READ;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Decode Phase
    int curr_pos = prompt_len;
    int32_t next_token = 0;

    float max_logit = -1e9f;
    for (int v = 0; v < model->vocab_size; ++v) {
        if (logits[v] > max_logit) {
            max_logit = logits[v];
            next_token = v;
        }
    }
    out_tokens[0] = next_token;

    auto t_dec_start = std::chrono::high_resolution_clock::now();
    for (int gen = 1; gen < max_new_tokens; ++gen) {
        if (!model->forward_token(next_token, curr_pos, logits)) return STREAM_ERR_READ;
        curr_pos++;

        max_logit = -1e9f;
        for (int v = 0; v < model->vocab_size; ++v) {
            if (logits[v] > max_logit) {
                max_logit = logits[v];
                next_token = v;
            }
        }
        out_tokens[gen] = next_token;
    }
    auto t_dec_end = std::chrono::high_resolution_clock::now();
    double decode_ms = std::chrono::duration<double, std::milli>(t_dec_end - t_dec_start).count();

    double total_ms = prefill_ms + decode_ms;
    double tok_per_sec = (max_new_tokens > 1) ? ((max_new_tokens - 1) / (decode_ms * 1e-3)) : (1.0 / (decode_ms * 1e-3));

    if (out_stats) {
        out_stats[0] = prefill_ms;
        out_stats[1] = decode_ms;
        out_stats[2] = total_ms;
        out_stats[3] = tok_per_sec;
    }
    return max_new_tokens;
}

int file_streaming_verify_integrity(NativeFileStreamingModel* model) {
    if (!model || model->fd < 0) return STREAM_ERR_PARAM;

    std::vector<uint8_t> check_buf;
    for (const auto& meta : model->layer_indices) {
        if (check_buf.size() < std::max(meta.p1_bytes, meta.p2_bytes)) {
            check_buf.resize(std::max(meta.p1_bytes, meta.p2_bytes));
        }

        // Verify p1 CRC
        if (!robust_pread(model->fd, check_buf.data(), meta.p1_bytes, meta.p1_offset)) {
            return STREAM_ERR_READ;
        }
        if (compute_crc32(check_buf.data(), meta.p1_bytes) != meta.p1_crc32) {
            return STREAM_ERR_CRC;
        }

        // Verify p2 CRC
        if (!robust_pread(model->fd, check_buf.data(), meta.p2_bytes, meta.p2_offset)) {
            return STREAM_ERR_READ;
        }
        if (compute_crc32(check_buf.data(), meta.p2_bytes) != meta.p2_crc32) {
            return STREAM_ERR_CRC;
        }
    }
    return STREAM_SUCCESS;
}

} // extern "C"
