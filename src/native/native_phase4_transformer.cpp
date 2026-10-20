#include "native_phase4_transformer.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <chrono>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <arm_neon.h>
#include <zlib.h>

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
// Direct GEMV Kernels
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
    const int full_blocks = N / 5;
    const int rem = N % 5;
    const int bytes_per_row = (N + 4) / 5;

    for (int i = 0; i < M; ++i) {
        const uint8_t* __restrict__ row_p1 = p1 + i * bytes_per_row;
        const uint8_t* __restrict__ row_p2 = p2 + i * bytes_per_row;

        float sum1 = 0.0f;
        float sum2 = 0.0f;

        for (int jb = 0; jb < full_blocks; ++jb) {
            int j = jb * 5;
            const float* t1 = RADIX3_FLOAT_LUT[row_p1[jb]];
            const float* t2 = RADIX3_FLOAT_LUT[row_p2[jb]];

            sum1 += t1[0] * x[j + 0] + t1[1] * x[j + 1] + t1[2] * x[j + 2] + t1[3] * x[j + 3] + t1[4] * x[j + 4];
            sum2 += t2[0] * x[j + 0] + t2[1] * x[j + 1] + t2[2] * x[j + 2] + t2[3] * x[j + 3] + t2[4] * x[j + 4];
        }

        if (rem > 0) {
            int j = full_blocks * 5;
            const float* t1 = RADIX3_FLOAT_LUT[row_p1[full_blocks]];
            const float* t2 = RADIX3_FLOAT_LUT[row_p2[full_blocks]];
            for (int r = 0; r < rem; ++r) {
                sum1 += t1[r] * x[j + r];
                sum2 += t2[r] * x[j + r];
            }
        }

        y[i] = s1[i] * sum1 + s2[i] * sum2;
    }
}

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

static bool robust_pread(int fd, void* dest, size_t count, off_t offset) {
    uint8_t* ptr = static_cast<uint8_t*>(dest);
    size_t total_read = 0;
    while (total_read < count) {
        ssize_t n = ::pread(fd, ptr + total_read, count - total_read, offset + total_read);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) return false;
        total_read += n;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Phase 4.0 Structs
// ---------------------------------------------------------------------------
struct Phase4LayerIndex {
    int32_t block_idx = 0;
    int32_t layer_type = 0;
    uint32_t M = 0;
    uint32_t N = 0;
    uint64_t p1_offset = 0;
    uint64_t p2_offset = 0;
    uint32_t p1_comp_bytes = 0;
    uint32_t p1_raw_bytes = 0;
    uint32_t p1_crc32 = 0;
    uint32_t p2_comp_bytes = 0;
    uint32_t p2_raw_bytes = 0;
    uint32_t p2_crc32 = 0;
    std::vector<float> s1;
    std::vector<float> s2;
};

struct Phase4BlockMeta {
    std::vector<float> ln1_w, ln1_b;
    std::vector<float> ln2_w, ln2_b;
    std::vector<float> fc1_b, fc2_b;

    // Mode 0: Persistent FP32 KV Cache (81,920 B per block for K + 81,920 B for V = 163,840 B)
    alignas(64) float k_cache_fp32[5][128][32];
    alignas(64) float v_cache_fp32[5][128][32];

    // Mode 1: Dynamic Int8 Quantized KV Cache (23,040 B per block for K + 23,040 B for V = 46,080 B)
    alignas(64) int8_t k_cache_i8[5][128][32];
    alignas(64) float  k_cache_scale[5][128];
    alignas(64) int8_t v_cache_i8[5][128][32];
    alignas(64) float  v_cache_scale[5][128];

    void clear_cache() {
        std::memset(k_cache_fp32, 0, sizeof(k_cache_fp32));
        std::memset(v_cache_fp32, 0, sizeof(v_cache_fp32));
        std::memset(k_cache_i8, 0, sizeof(k_cache_i8));
        std::memset(k_cache_scale, 0, sizeof(k_cache_scale));
        std::memset(v_cache_i8, 0, sizeof(v_cache_i8));
        std::memset(v_cache_scale, 0, sizeof(v_cache_scale));
    }
};

// ---------------------------------------------------------------------------
// Ultra-Fast Sparse-Trit Run-Length Decompressor (STRE)
// ---------------------------------------------------------------------------
static inline void fast_stre_decompress(
    const uint8_t* __restrict__ src,
    size_t src_len,
    uint8_t* __restrict__ dest,
    size_t raw_len
) {
    const uint8_t* src_end = src + src_len;
    while (src < src_end) {
        uint8_t b = *src++;
        if (b == 243) { // Run of byte 121 (all-zero trits)
            uint8_t count = *src++;
            std::memset(dest, 121, count);
            dest += count;
        } else if (b == 244) { // Run of arbitrary byte
            uint8_t val = *src++;
            uint8_t count = *src++;
            std::memset(dest, val, count);
            dest += count;
        } else {
            *dest++ = b;
        }
    }
}

struct NativePhase4Model {
    int fd = -1;
    int container_format = 0; // 0 = .llstream (raw), 1 = .llchunk (zlib), 2 = .llstre (fast STRE)
    int is_chunk_compressed = 0;
    int pack_mode = PHASE4_PACK_RADIX3;
    int stream_level = PHASE4_STREAM_LAYER;
    int kv_mode = PHASE4_KV_FP32; // 0 = FP32, 1 = Int8 Dynamic

    int d_model = 160;
    int n_heads = 5;
    int d_head = 32;
    int n_blocks = 4;
    int max_context = 128;
    int vocab_size = 256;
    int num_streamed_layers = 25;

    // Persistent Non-Block Parameters
    std::vector<int8_t> wte_q;
    float wte_s = 0.0f;
    std::vector<int8_t> wpe_q;
    float wpe_s = 0.0f;

    std::vector<Phase4BlockMeta> blocks;
    std::vector<float> ln_f_w, ln_f_b;

    std::vector<Phase4LayerIndex> layer_indices;

    // Reusable Transient Scratchpads
    alignas(64) std::vector<uint8_t> transient_p1_scratch;
    alignas(64) std::vector<uint8_t> transient_p2_scratch;
    alignas(64) std::vector<uint8_t> transient_comp_scratch; // For reading compressed chunk before decompressing
    size_t scratchpad_allocated_bytes = 0;

    // Telemetry
    uint64_t pread_calls = 0;
    uint64_t bytes_streamed = 0;
    uint64_t decomp_calls = 0;
    uint64_t decomp_bytes = 0;

    // Activations
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

    // Reusable Persistent Decompression Stream
    z_stream strm;
    bool strm_initialized = false;

    NativePhase4Model(int p_mode, int s_level, int k_mode, int c_format)
        : pack_mode(p_mode), stream_level(s_level), kv_mode(k_mode), container_format(c_format) {
        is_chunk_compressed = (container_format != 0);
        blocks.resize(n_blocks);
        init_2bit_trit_lut();
        init_radix3_lut();
        init_crc32_table();

        size_t per_plane = (pack_mode == PHASE4_PACK_RADIX3) ? 16480 : 20480;
        transient_p1_scratch.resize(per_plane);
        transient_p2_scratch.resize(per_plane);
        scratchpad_allocated_bytes = per_plane * 2;

        if (is_chunk_compressed) {
            transient_comp_scratch.resize(per_plane * 2);
            scratchpad_allocated_bytes += per_plane * 2;

            if (container_format == 1) {
                strm.zalloc = Z_NULL;
                strm.zfree = Z_NULL;
                strm.opaque = Z_NULL;
                strm.avail_in = 0;
                strm.next_in = Z_NULL;
                if (inflateInit(&strm) == Z_OK) {
                    strm_initialized = true;
                }
            }
        }
    }

    ~NativePhase4Model() {
        if (strm_initialized) {
            inflateEnd(&strm);
            strm_initialized = false;
        }
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
    }

    void clear_kv_caches() {
        for (auto& b : blocks) b.clear_cache();
    }

    bool execute_streaming_linear(int layer_idx, const float* in_vec, float* out_vec, const float* opt_bias = nullptr) {
        const auto& meta = layer_indices[layer_idx];

        if (container_format == 1) {
            // Coalesced Dual-Plane Read (1 pread per projection instead of 2)
            size_t total_comp = meta.p1_comp_bytes + meta.p2_comp_bytes;
            if (!robust_pread(fd, transient_comp_scratch.data(), total_comp, meta.p1_offset)) return false;

            // 1. Decompress p1
            inflateReset(&strm);
            strm.avail_in = meta.p1_comp_bytes;
            strm.next_in = transient_comp_scratch.data();
            strm.avail_out = meta.p1_raw_bytes;
            strm.next_out = transient_p1_scratch.data();
            if (inflate(&strm, Z_FINISH) != Z_STREAM_END) return false;

            // 2. Decompress p2
            inflateReset(&strm);
            strm.avail_in = meta.p2_comp_bytes;
            strm.next_in = transient_comp_scratch.data() + meta.p1_comp_bytes;
            strm.avail_out = meta.p2_raw_bytes;
            strm.next_out = transient_p2_scratch.data();
            if (inflate(&strm, Z_FINISH) != Z_STREAM_END) return false;

            pread_calls += 1;
            bytes_streamed += total_comp;
            decomp_calls += 2;
            decomp_bytes += (meta.p1_raw_bytes + meta.p2_raw_bytes);
        } else if (container_format == 2) {
            // STRE Coalesced Dual-Plane Read (1 pread per projection, ~10 GB/s decompression)
            size_t total_comp = meta.p1_comp_bytes + meta.p2_comp_bytes;
            if (!robust_pread(fd, transient_comp_scratch.data(), total_comp, meta.p1_offset)) return false;

            fast_stre_decompress(transient_comp_scratch.data(), meta.p1_comp_bytes, transient_p1_scratch.data(), meta.p1_raw_bytes);
            fast_stre_decompress(transient_comp_scratch.data() + meta.p1_comp_bytes, meta.p2_comp_bytes, transient_p2_scratch.data(), meta.p2_raw_bytes);

            pread_calls += 1;
            bytes_streamed += total_comp;
            decomp_calls += 2;
            decomp_bytes += (meta.p1_raw_bytes + meta.p2_raw_bytes);
        } else {
            // Raw pread streaming directly into GEMV scratchpads
            if (!robust_pread(fd, transient_p1_scratch.data(), meta.p1_raw_bytes, meta.p1_offset)) return false;
            if (!robust_pread(fd, transient_p2_scratch.data(), meta.p2_raw_bytes, meta.p2_offset)) return false;

            pread_calls += 2;
            bytes_streamed += (meta.p1_raw_bytes + meta.p2_raw_bytes);
        }

        // Direct GEMV execution
        if (pack_mode == PHASE4_PACK_RADIX3) {
            direct_radix3_gemv(
                transient_p1_scratch.data(), transient_p2_scratch.data(),
                meta.s1.data(), meta.s2.data(),
                in_vec, out_vec, meta.M, meta.N
            );
        } else {
            direct_packed_gemv_neon(
                transient_p1_scratch.data(), transient_p2_scratch.data(),
                meta.s1.data(), meta.s2.data(),
                in_vec, out_vec, meta.M, meta.N
            );
        }

        if (opt_bias) {
            for (int i = 0; i < meta.M; ++i) out_vec[i] += opt_bias[i];
        }
        return true;
    }

    bool forward_token(int token_id, int pos, float* out_logits) {
        // Embeddings
        const int8_t* wte_row = wte_q.data() + token_id * d_model;
        const int8_t* wpe_row = wpe_q.data() + pos * d_model;
        for (int i = 0; i < d_model; ++i) {
            x[i] = (float)wte_row[i] * wte_s + (float)wpe_row[i] * wpe_s;
        }

        const float scale_factor = 1.0f / std::sqrt((float)d_head);

        for (int bi = 0; bi < n_blocks; ++bi) {
            auto& b = blocks[bi];
            int base_layer = bi * 6;

            layer_norm(x_norm1, x, b.ln1_w.data(), b.ln1_b.data(), d_model);

            if (!execute_streaming_linear(base_layer + 0, x_norm1, q_buf)) return false;
            if (!execute_streaming_linear(base_layer + 1, x_norm1, k_buf)) return false;
            if (!execute_streaming_linear(base_layer + 2, x_norm1, v_buf)) return false;

            if (kv_mode == PHASE4_KV_INT8) {
                // Update Dynamic Int8 Quantized KV Cache
                for (int h = 0; h < n_heads; ++h) {
                    const float* k_h = k_buf + (h * d_head);
                    const float* v_h = v_buf + (h * d_head);

                    // Dynamic scale for K
                    float max_k = 0.0f;
                    for (int d = 0; d < d_head; ++d) {
                        float a = std::abs(k_h[d]);
                        if (a > max_k) max_k = a;
                    }
                    float sk = (max_k > 1e-8f) ? (max_k / 127.0f) : 1e-8f;
                    float inv_sk = 1.0f / sk;
                    b.k_cache_scale[h][pos] = sk;
                    for (int d = 0; d < d_head; ++d) {
                        int val = std::round(k_h[d] * inv_sk);
                        b.k_cache_i8[h][pos][d] = (int8_t)std::clamp(val, -127, 127);
                    }

                    // Dynamic scale for V
                    float max_v = 0.0f;
                    for (int d = 0; d < d_head; ++d) {
                        float a = std::abs(v_h[d]);
                        if (a > max_v) max_v = a;
                    }
                    float sv = (max_v > 1e-8f) ? (max_v / 127.0f) : 1e-8f;
                    float inv_sv = 1.0f / sv;
                    b.v_cache_scale[h][pos] = sv;
                    for (int d = 0; d < d_head; ++d) {
                        int val = std::round(v_h[d] * inv_sv);
                        b.v_cache_i8[h][pos][d] = (int8_t)std::clamp(val, -127, 127);
                    }
                }

                // Attention evaluation with Int8 KV cache
                for (int h = 0; h < n_heads; ++h) {
                    const float* q_h = q_buf + h * d_head;
                    float* y_h = y_attn + h * d_head;

                    float max_score = -1e9f;
                    for (int t = 0; t <= pos; ++t) {
                        const int8_t* k_t = &b.k_cache_i8[h][t][0];
                        float dot = 0.0f;
                        for (int d = 0; d < d_head; ++d) {
                            dot += q_h[d] * (float)k_t[d];
                        }
                        float sc = (dot * b.k_cache_scale[h][t]) * scale_factor;
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
                        float a = att_scores[t] * b.v_cache_scale[h][t];
                        const int8_t* v_t = &b.v_cache_i8[h][t][0];
                        for (int d = 0; d < d_head; ++d) {
                            y_h[d] += a * (float)v_t[d];
                        }
                    }
                }
            } else {
                // FP32 KV Cache
                for (int h = 0; h < n_heads; ++h) {
                    std::memcpy(&b.k_cache_fp32[h][pos][0], k_buf + (h * d_head), sizeof(float) * d_head);
                    std::memcpy(&b.v_cache_fp32[h][pos][0], v_buf + (h * d_head), sizeof(float) * d_head);
                }

                for (int h = 0; h < n_heads; ++h) {
                    const float* q_h = q_buf + h * d_head;
                    float* y_h = y_attn + h * d_head;

                    float max_score = -1e9f;
                    for (int t = 0; t <= pos; ++t) {
                        const float* k_t = &b.k_cache_fp32[h][t][0];
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
                        const float* v_t = &b.v_cache_fp32[h][t][0];
                        for (int d = 0; d < d_head; ++d) {
                            y_h[d] += a * v_t[d];
                        }
                    }
                }
            }

            if (!execute_streaming_linear(base_layer + 3, y_attn, x_norm2)) return false;
            for (int i = 0; i < d_model; ++i) x[i] += x_norm2[i];

            layer_norm(x_norm2, x, b.ln2_w.data(), b.ln2_b.data(), d_model);

            if (!execute_streaming_linear(base_layer + 4, x_norm2, h_mlp, b.fc1_b.data())) return false;
            gelu_inplace(h_mlp, 512);

            if (!execute_streaming_linear(base_layer + 5, h_mlp, out_mlp, b.fc2_b.data())) return false;
            for (int i = 0; i < d_model; ++i) x[i] += out_mlp[i];
        }

        layer_norm(x_f, x, ln_f_w.data(), ln_f_b.data(), d_model);
        if (!execute_streaming_linear(24, x_f, out_logits)) return false;

        return true;
    }
};

extern "C" {

NativePhase4Model* create_phase4_model(
    const char* container_path,
    int stream_level,
    int kv_mode,
    int* out_status
) {
    if (!container_path) {
        if (out_status) *out_status = PHASE4_ERR_PARAM;
        return nullptr;
    }

    int fd = ::open(container_path, O_RDONLY);
    if (fd < 0) {
        if (out_status) *out_status = PHASE4_ERR_OPEN;
        return nullptr;
    }

    char magic[8];
    if (!robust_pread(fd, magic, 8, 0)) {
        ::close(fd);
        if (out_status) *out_status = PHASE4_ERR_READ;
        return nullptr;
    }

    int is_chunk = 0;
    if (std::memcmp(magic, "LLSTRM01", 8) == 0) {
        is_chunk = 0;
    } else if (std::memcmp(magic, "LLCHNK01", 8) == 0) {
        is_chunk = 1;
    } else if (std::memcmp(magic, "LLSTRE01", 8) == 0) {
        is_chunk = 2;
    } else {
        ::close(fd);
        if (out_status) *out_status = PHASE4_ERR_MAGIC;
        return nullptr;
    }

    uint32_t header_vals[8];
    if (!robust_pread(fd, header_vals, sizeof(header_vals), 8)) {
        ::close(fd);
        if (out_status) *out_status = PHASE4_ERR_READ;
        return nullptr;
    }

    uint32_t version = header_vals[0];
    if (version != 1) {
        ::close(fd);
        if (out_status) *out_status = PHASE4_ERR_VERSION;
        return nullptr;
    }

    int pack_mode = header_vals[1];
    NativePhase4Model* model = new NativePhase4Model(pack_mode, stream_level, kv_mode, is_chunk);
    model->fd = fd;
    model->d_model = header_vals[2];
    model->n_heads = header_vals[3];
    model->d_head = header_vals[4];
    model->n_blocks = header_vals[5];
    model->max_context = header_vals[6];
    model->vocab_size = header_vals[7];

    off_t curr_off = 8 + sizeof(header_vals);
    uint32_t wte_bytes = 0;
    uint32_t wpe_bytes = 0;
    uint32_t num_layers = 0;

    // Embeddings
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

    // Block Norms & Biases
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

    // Final Head Norms
    model->ln_f_w.resize(model->d_model);
    model->ln_f_b.resize(model->d_model);
    if (!robust_pread(fd, model->ln_f_w.data(), model->d_model * sizeof(float), curr_off)) goto read_err;
    curr_off += model->d_model * sizeof(float);
    if (!robust_pread(fd, model->ln_f_b.data(), model->d_model * sizeof(float), curr_off)) goto read_err;
    curr_off += model->d_model * sizeof(float);

    // Streamed Layer Index Table
    if (!robust_pread(fd, &num_layers, sizeof(uint32_t), curr_off)) goto read_err;
    curr_off += sizeof(uint32_t);
    model->num_streamed_layers = num_layers;
    model->layer_indices.resize(num_layers);

    for (uint32_t li = 0; li < num_layers; ++li) {
        auto& meta = model->layer_indices[li];
        uint32_t meta_ints[4];
        if (!robust_pread(fd, meta_ints, sizeof(meta_ints), curr_off)) goto read_err;
        curr_off += sizeof(meta_ints);
        meta.block_idx = (int32_t)meta_ints[0];
        meta.layer_type = (int32_t)meta_ints[1];
        meta.M = meta_ints[2];
        meta.N = meta_ints[3];

        uint64_t off_data[2];
        if (!robust_pread(fd, off_data, sizeof(off_data), curr_off)) goto read_err;
        curr_off += sizeof(off_data);
        meta.p1_offset = off_data[0];
        meta.p2_offset = off_data[1];

        if (is_chunk) {
            uint32_t size_crc[6];
            if (!robust_pread(fd, size_crc, sizeof(size_crc), curr_off)) goto read_err;
            curr_off += sizeof(size_crc);
            meta.p1_comp_bytes = size_crc[0];
            meta.p1_raw_bytes = size_crc[1];
            meta.p1_crc32 = size_crc[2];
            meta.p2_comp_bytes = size_crc[3];
            meta.p2_raw_bytes = size_crc[4];
            meta.p2_crc32 = size_crc[5];
        } else {
            uint32_t size_crc[4];
            if (!robust_pread(fd, size_crc, sizeof(size_crc), curr_off)) goto read_err;
            curr_off += sizeof(size_crc);
            meta.p1_comp_bytes = size_crc[0];
            meta.p1_raw_bytes = size_crc[0];
            meta.p1_crc32 = size_crc[1];
            meta.p2_comp_bytes = size_crc[2];
            meta.p2_raw_bytes = size_crc[2];
            meta.p2_crc32 = size_crc[3];
        }

        meta.s1.resize(meta.M);
        meta.s2.resize(meta.M);
        if (!robust_pread(fd, meta.s1.data(), meta.M * sizeof(float), curr_off)) goto read_err;
        curr_off += meta.M * sizeof(float);
        if (!robust_pread(fd, meta.s2.data(), meta.M * sizeof(float), curr_off)) goto read_err;
        curr_off += meta.M * sizeof(float);
    }

    if (out_status) *out_status = PHASE4_SUCCESS;
    return model;

read_err:
    delete model;
    if (out_status) *out_status = PHASE4_ERR_READ;
    return nullptr;
}

void free_phase4_model(NativePhase4Model* model) {
    if (model) delete model;
}

size_t phase4_get_active_weight_ram(NativePhase4Model* model) {
    if (!model) return 0;
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
    return embs + scales + norms_biases + model->scratchpad_allocated_bytes;
}

size_t phase4_get_kv_cache_ram(NativePhase4Model* model) {
    if (!model) return 0;
    if (model->kv_mode == PHASE4_KV_INT8) {
        // 4 blocks * (5 heads * 128 ctx * 32 int8 + 5 * 128 float scale) * 2 (K+V) = 184,320 B
        return model->n_blocks * (sizeof(model->blocks[0].k_cache_i8) + sizeof(model->blocks[0].k_cache_scale) +
                                  sizeof(model->blocks[0].v_cache_i8) + sizeof(model->blocks[0].v_cache_scale));
    } else {
        // 4 blocks * (5 heads * 128 ctx * 32 float) * 2 (K+V) = 655,360 B
        return model->n_blocks * (sizeof(model->blocks[0].k_cache_fp32) + sizeof(model->blocks[0].v_cache_fp32));
    }
}

size_t phase4_get_total_model_ram(NativePhase4Model* model) {
    if (!model) return 0;
    size_t weights = phase4_get_active_weight_ram(model);
    size_t kv_cache = phase4_get_kv_cache_ram(model);
    size_t activations = sizeof(model->x) + sizeof(model->x_norm1) + sizeof(model->q_buf) +
                         sizeof(model->k_buf) + sizeof(model->v_buf) + sizeof(model->att_scores) +
                         sizeof(model->y_attn) + sizeof(model->x_norm2) + sizeof(model->h_mlp) +
                         sizeof(model->out_mlp) + sizeof(model->x_f);
    return weights + kv_cache + activations;
}

void phase4_get_telemetry_stats(
    NativePhase4Model* model,
    uint64_t* out_pread_calls,
    uint64_t* out_bytes_streamed,
    uint64_t* out_decomp_calls,
    uint64_t* out_decomp_bytes
) {
    if (!model) return;
    if (out_pread_calls) *out_pread_calls = model->pread_calls;
    if (out_bytes_streamed) *out_bytes_streamed = model->bytes_streamed;
    if (out_decomp_calls) *out_decomp_calls = model->decomp_calls;
    if (out_decomp_bytes) *out_decomp_bytes = model->decomp_bytes;
}

void phase4_reset_telemetry_stats(NativePhase4Model* model) {
    if (!model) return;
    model->pread_calls = 0;
    model->bytes_streamed = 0;
    model->decomp_calls = 0;
    model->decomp_bytes = 0;
}

int phase4_forward_step(NativePhase4Model* model, int token_id, int pos, float* logits_out) {
    if (!model || !logits_out) return PHASE4_ERR_PARAM;
    if (!model->forward_token(token_id, pos, logits_out)) return PHASE4_ERR_READ;
    return PHASE4_SUCCESS;
}

int phase4_generate_tokens(
    NativePhase4Model* model,
    const int32_t* prompt,
    int prompt_len,
    int max_new_tokens,
    int32_t* out_tokens,
    double* out_stats
) {
    if (!model || !prompt || !out_tokens) return PHASE4_ERR_PARAM;
    if (prompt_len <= 0 || prompt_len + max_new_tokens > model->max_context) return -1;

    model->clear_kv_caches();
    alignas(64) float logits[256];

    // Prefill
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int p = 0; p < prompt_len; ++p) {
        if (!model->forward_token(prompt[p], p, logits)) return PHASE4_ERR_READ;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Decode
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
        if (!model->forward_token(next_token, curr_pos, logits)) return PHASE4_ERR_READ;
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

int phase4_verify_integrity(NativePhase4Model* model) {
    if (!model || model->fd < 0) return PHASE4_ERR_PARAM;

    std::vector<uint8_t> check_comp;
    std::vector<uint8_t> check_raw;

    for (const auto& meta : model->layer_indices) {
        if (check_raw.size() < std::max(meta.p1_raw_bytes, meta.p2_raw_bytes)) {
            check_raw.resize(std::max(meta.p1_raw_bytes, meta.p2_raw_bytes));
        }

        if (model->container_format == 1) {
            if (check_comp.size() < std::max(meta.p1_comp_bytes, meta.p2_comp_bytes)) {
                check_comp.resize(std::max(meta.p1_comp_bytes, meta.p2_comp_bytes));
            }

            // Verify p1
            if (!robust_pread(model->fd, check_comp.data(), meta.p1_comp_bytes, meta.p1_offset)) return PHASE4_ERR_READ;
            uLongf dest_len1 = meta.p1_raw_bytes;
            if (::uncompress(check_raw.data(), &dest_len1, check_comp.data(), meta.p1_comp_bytes) != Z_OK) return PHASE4_ERR_DECOMP;
            if (compute_crc32(check_raw.data(), meta.p1_raw_bytes) != meta.p1_crc32) return PHASE4_ERR_CRC;

            // Verify p2
            if (!robust_pread(model->fd, check_comp.data(), meta.p2_comp_bytes, meta.p2_offset)) return PHASE4_ERR_READ;
            uLongf dest_len2 = meta.p2_raw_bytes;
            if (::uncompress(check_raw.data(), &dest_len2, check_comp.data(), meta.p2_comp_bytes) != Z_OK) return PHASE4_ERR_DECOMP;
            if (compute_crc32(check_raw.data(), meta.p2_raw_bytes) != meta.p2_crc32) return PHASE4_ERR_CRC;
        } else if (model->container_format == 2) {
            if (check_comp.size() < std::max(meta.p1_comp_bytes, meta.p2_comp_bytes)) {
                check_comp.resize(std::max(meta.p1_comp_bytes, meta.p2_comp_bytes));
            }

            // Verify p1
            if (!robust_pread(model->fd, check_comp.data(), meta.p1_comp_bytes, meta.p1_offset)) return PHASE4_ERR_READ;
            fast_stre_decompress(check_comp.data(), meta.p1_comp_bytes, check_raw.data(), meta.p1_raw_bytes);
            if (compute_crc32(check_raw.data(), meta.p1_raw_bytes) != meta.p1_crc32) return PHASE4_ERR_CRC;

            // Verify p2
            if (!robust_pread(model->fd, check_comp.data(), meta.p2_comp_bytes, meta.p2_offset)) return PHASE4_ERR_READ;
            fast_stre_decompress(check_comp.data(), meta.p2_comp_bytes, check_raw.data(), meta.p2_raw_bytes);
            if (compute_crc32(check_raw.data(), meta.p2_raw_bytes) != meta.p2_crc32) return PHASE4_ERR_CRC;
        } else {
            // Verify raw p1
            if (!robust_pread(model->fd, check_raw.data(), meta.p1_raw_bytes, meta.p1_offset)) return PHASE4_ERR_READ;
            if (compute_crc32(check_raw.data(), meta.p1_raw_bytes) != meta.p1_crc32) return PHASE4_ERR_CRC;

            // Verify raw p2
            if (!robust_pread(model->fd, check_raw.data(), meta.p2_raw_bytes, meta.p2_offset)) return PHASE4_ERR_READ;
            if (compute_crc32(check_raw.data(), meta.p2_raw_bytes) != meta.p2_crc32) return PHASE4_ERR_CRC;
        }
    }
    return PHASE4_SUCCESS;
}

} // extern "C"
