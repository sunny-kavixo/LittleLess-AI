#include "native_streaming_transformer.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <chrono>
#include <algorithm>
#include <arm_neon.h>

// 256-entry lookup tables
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

// ARM NEON Vectorized Direct 2-Bit Packed GEMV
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

// Row-Aligned Direct Radix-3 GEMV
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

// LayerNorm
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

// Fast GELU
static inline void gelu_inplace(float* __restrict__ x, int size) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    for (int i = 0; i < size; ++i) {
        float val = x[i];
        float inner = sqrt_2_over_pi * (val + 0.044715f * val * val * val);
        x[i] = 0.5f * val * (1.0f + std::tanh(inner));
    }
}

// Registered external streaming layer metadata
struct StreamingLayerMeta {
    const uint8_t* p1_source = nullptr;
    const uint8_t* p2_source = nullptr;
    int byte_count = 0;
    std::vector<float> s1;
    std::vector<float> s2;
    int M = 0;
    int N = 0;
};

struct StreamingBlockMeta {
    std::vector<float> ln1_w, ln1_b;
    std::vector<float> ln2_w, ln2_b;
    std::vector<float> fc1_b, fc2_b;
    StreamingLayerMeta layers[6]; // 0=q, 1=k, 2=v, 3=out, 4=fc1, 5=fc2

    // Persistent KV Cache in RAM
    alignas(64) float k_cache[5][128][32];
    alignas(64) float v_cache[5][128][32];

    void clear_cache() {
        std::memset(k_cache, 0, sizeof(k_cache));
        std::memset(v_cache, 0, sizeof(v_cache));
    }
};

struct NativeStreamingModel {
    int pack_mode = STREAM_PACK_2BIT;
    int stream_level = STREAM_LEVEL_LAYER; // 0=Block, 1=Layer

    int d_model = 160;
    int n_heads = 5;
    int d_head = 32;
    int n_blocks = 4;
    int max_context = 128;
    int vocab_size = 256;

    // Persistent Parameters in RAM
    std::vector<int8_t> wte_q;
    float wte_s = 0.0f;
    std::vector<int8_t> wpe_q;
    float wpe_s = 0.0f;

    std::vector<StreamingBlockMeta> blocks;

    std::vector<float> ln_f_w, ln_f_b;
    StreamingLayerMeta lm_head_meta;

    // TRANSIENT REUSABLE SCRATCHPAD BUFFERS
    // For Layer streaming: max layer is mlp_fc1 (512x160): 20,480 B (p1) + 20,480 B (p2) = 40,960 B
    // For Block streaming: 133,120 B (p1+p2 for all 6 layers of 1 block)
    alignas(64) std::vector<uint8_t> transient_p1_scratch;
    alignas(64) std::vector<uint8_t> transient_p2_scratch;

    // Scratchpad capacity in bytes
    size_t scratchpad_allocated_bytes = 0;

    // Active inference activation vectors
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

    NativeStreamingModel(int p_mode, int s_level) : pack_mode(p_mode), stream_level(s_level) {
        blocks.resize(n_blocks);
        init_2bit_trit_lut();
        init_radix3_lut();

        // Calculate and allocate exactly the bounded scratchpad
        if (stream_level == STREAM_LEVEL_BLOCK) {
            // Block scratchpad: holds 1 block (6 layers)
            // 2-bit: 133,120 B total (66,560 B p1 + 66,560 B p2)
            // Radix-3: 106,688 B total (53,344 B p1 + 53,344 B p2)
            size_t per_plane = (pack_mode == STREAM_PACK_RADIX3) ? 53344 : 66560;
            transient_p1_scratch.resize(per_plane);
            transient_p2_scratch.resize(per_plane);
            scratchpad_allocated_bytes = per_plane * 2;
        } else {
            // Layer scratchpad: holds only the largest single layer (mlp_fc1: 512x160)
            // 2-bit: 20,480 B per plane = 40,960 B total
            // Radix-3: 16,480 B per plane = 32,960 B total
            size_t per_plane = (pack_mode == STREAM_PACK_RADIX3) ? 16480 : 20480;
            transient_p1_scratch.resize(per_plane);
            transient_p2_scratch.resize(per_plane);
            scratchpad_allocated_bytes = per_plane * 2;
        }
    }

    void clear_kv_caches() {
        for (auto& b : blocks) b.clear_cache();
    }

    // Streams a single linear projection Just-In-Time into transient scratchpad and evaluates GEMV
    void execute_streaming_linear(
        const StreamingLayerMeta& meta,
        const float* in_vec,
        float* out_vec,
        const float* opt_bias = nullptr
    ) {
        // Stream / copy the active layer's weights into transient scratchpad
        std::memcpy(transient_p1_scratch.data(), meta.p1_source, meta.byte_count);
        std::memcpy(transient_p2_scratch.data(), meta.p2_source, meta.byte_count);

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
    }

    void forward_token(int token_id, int pos, float* out_logits) {
        // 1. Embeddings (Persistent)
        const int8_t* wte_row = wte_q.data() + token_id * d_model;
        const int8_t* wpe_row = wpe_q.data() + pos * d_model;
        for (int i = 0; i < d_model; ++i) {
            x[i] = (float)wte_row[i] * wte_s + (float)wpe_row[i] * wpe_s;
        }

        const float scale_factor = 1.0f / std::sqrt((float)d_head);

        // 2. Sequential Block Execution with JIT Weight Streaming
        for (int bi = 0; bi < n_blocks; ++bi) {
            auto& b = blocks[bi];

            // Attention LN1
            layer_norm(x_norm1, x, b.ln1_w.data(), b.ln1_b.data(), d_model);

            // Stream and evaluate Q, K, V
            execute_streaming_linear(b.layers[0], x_norm1, q_buf);
            execute_streaming_linear(b.layers[1], x_norm1, k_buf);
            execute_streaming_linear(b.layers[2], x_norm1, v_buf);

            // Update persistent KV Cache
            for (int h = 0; h < n_heads; ++h) {
                std::memcpy(&b.k_cache[h][pos][0], k_buf + (h * d_head), sizeof(float) * d_head);
                std::memcpy(&b.v_cache[h][pos][0], v_buf + (h * d_head), sizeof(float) * d_head);
            }

            // Multi-head self attention dot-product
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
            execute_streaming_linear(b.layers[3], y_attn, x_norm2);
            for (int i = 0; i < d_model; ++i) x[i] += x_norm2[i];

            // MLP LN2
            layer_norm(x_norm2, x, b.ln2_w.data(), b.ln2_b.data(), d_model);

            // Stream FC1 (Layer 4) with bias
            execute_streaming_linear(b.layers[4], x_norm2, h_mlp, b.fc1_b.data());
            gelu_inplace(h_mlp, 512);

            // Stream FC2 (Layer 5) with bias
            execute_streaming_linear(b.layers[5], h_mlp, out_mlp, b.fc2_b.data());
            for (int i = 0; i < d_model; ++i) x[i] += out_mlp[i];
        }

        // 3. Final LN & LM Head (Streamed through transient scratchpad)
        layer_norm(x_f, x, ln_f_w.data(), ln_f_b.data(), d_model);
        execute_streaming_linear(lm_head_meta, x_f, out_logits);
    }
};

extern "C" {

NativeStreamingModel* create_native_streaming_model(int pack_mode, int stream_level) {
    return new NativeStreamingModel(pack_mode, stream_level);
}

void free_native_streaming_model(NativeStreamingModel* model) {
    if (model) delete model;
}

void streaming_set_embeddings(
    NativeStreamingModel* model,
    const int8_t* wte_q, float wte_s,
    const int8_t* wpe_q, float wpe_s
) {
    model->wte_q.assign(wte_q, wte_q + model->vocab_size * model->d_model);
    model->wte_s = wte_s;
    model->wpe_q.assign(wpe_q, wpe_q + model->max_context * model->d_model);
    model->wpe_s = wpe_s;
}

void streaming_set_block_params(
    NativeStreamingModel* model,
    int block_idx,
    const float* ln1_w, const float* ln1_b,
    const float* ln2_w, const float* ln2_b,
    const float* fc1_b, const float* fc2_b
) {
    auto& b = model->blocks[block_idx];
    b.ln1_w.assign(ln1_w, ln1_w + model->d_model);
    b.ln1_b.assign(ln1_b, ln1_b + model->d_model);
    b.ln2_w.assign(ln2_w, ln2_w + model->d_model);
    b.ln2_b.assign(ln2_b, ln2_b + model->d_model);
    b.fc1_b.assign(fc1_b, fc1_b + 512);
    b.fc2_b.assign(fc2_b, fc2_b + model->d_model);
}

void streaming_set_layer_scales(
    NativeStreamingModel* model,
    int block_idx,
    int layer_type,
    const float* s1,
    const float* s2,
    int M,
    int N
) {
    auto& meta = model->blocks[block_idx].layers[layer_type];
    meta.M = M;
    meta.N = N;
    meta.s1.assign(s1, s1 + M);
    meta.s2.assign(s2, s2 + M);
}

void streaming_register_layer_weights(
    NativeStreamingModel* model,
    int block_idx,
    int layer_type,
    const uint8_t* p1_source,
    const uint8_t* p2_source,
    int byte_count
) {
    auto& meta = model->blocks[block_idx].layers[layer_type];
    meta.p1_source = p1_source;
    meta.p2_source = p2_source;
    meta.byte_count = byte_count;
}

void streaming_set_final_head(
    NativeStreamingModel* model,
    const float* ln_f_w, const float* ln_f_b,
    const float* head_s1, const float* head_s2,
    const uint8_t* head_p1, const uint8_t* head_p2,
    int M, int N
) {
    model->ln_f_w.assign(ln_f_w, ln_f_w + model->d_model);
    model->ln_f_b.assign(ln_f_b, ln_f_b + model->d_model);

    model->lm_head_meta.M = M;
    model->lm_head_meta.N = N;
    model->lm_head_meta.s1.assign(head_s1, head_s1 + M);
    model->lm_head_meta.s2.assign(head_s2, head_s2 + M);
    model->lm_head_meta.p1_source = head_p1;
    model->lm_head_meta.p2_source = head_p2;

    int bytes_per_plane = (model->pack_mode == STREAM_PACK_RADIX3) ? ((N + 4) / 5) * M : (N / 4) * M;
    model->lm_head_meta.byte_count = bytes_per_plane;
}

size_t streaming_get_active_weight_ram(NativeStreamingModel* model) {
    // Exact persistent model weight bytes + transient scratchpad buffer
    size_t embs = model->wte_q.size() * sizeof(int8_t) + model->wpe_q.size() * sizeof(int8_t);
    
    size_t scales = 0;
    for (int bi = 0; bi < model->n_blocks; ++bi) {
        for (int li = 0; li < 6; ++li) {
            scales += model->blocks[bi].layers[li].s1.size() * sizeof(float);
            scales += model->blocks[bi].layers[li].s2.size() * sizeof(float);
        }
    }
    scales += model->lm_head_meta.s1.size() * sizeof(float);
    scales += model->lm_head_meta.s2.size() * sizeof(float);

    size_t norms_biases = model->ln_f_w.size() * sizeof(float) + model->ln_f_b.size() * sizeof(float);
    for (int bi = 0; bi < model->n_blocks; ++bi) {
        norms_biases += model->blocks[bi].ln1_w.size() * sizeof(float) + model->blocks[bi].ln1_b.size() * sizeof(float);
        norms_biases += model->blocks[bi].ln2_w.size() * sizeof(float) + model->blocks[bi].ln2_b.size() * sizeof(float);
        norms_biases += model->blocks[bi].fc1_b.size() * sizeof(float) + model->blocks[bi].fc2_b.size() * sizeof(float);
    }

    size_t scratchpad = model->scratchpad_allocated_bytes;
    return embs + scales + norms_biases + scratchpad;
}

size_t streaming_get_total_model_ram(NativeStreamingModel* model) {
    size_t weights = streaming_get_active_weight_ram(model);
    // KV Cache: 4 blocks * 5 heads * 128 context * 32 head_dim * 4 bytes * 2 (K+V)
    size_t kv_cache = model->n_blocks * sizeof(model->blocks[0].k_cache) * 2;
    // Activations
    size_t activations = sizeof(model->x) + sizeof(model->x_norm1) + sizeof(model->q_buf) +
                         sizeof(model->k_buf) + sizeof(model->v_buf) + sizeof(model->att_scores) +
                         sizeof(model->y_attn) + sizeof(model->x_norm2) + sizeof(model->h_mlp) +
                         sizeof(model->out_mlp) + sizeof(model->x_f);
    return weights + kv_cache + activations;
}

void streaming_forward_step(
    NativeStreamingModel* model,
    int token_id,
    int pos,
    float* logits_out
) {
    model->forward_token(token_id, pos, logits_out);
}

int streaming_generate_tokens(
    NativeStreamingModel* model,
    const int32_t* prompt,
    int prompt_len,
    int max_new_tokens,
    int32_t* out_tokens,
    double* out_stats
) {
    if (prompt_len <= 0 || prompt_len + max_new_tokens > model->max_context) return -1;

    model->clear_kv_caches();
    alignas(64) float logits[256];

    // Prefill Phase
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int p = 0; p < prompt_len; ++p) {
        model->forward_token(prompt[p], p, logits);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Decode Phase (1 token at a time)
    int curr_pos = prompt_len;
    int32_t next_token = 0;

    // Argmax for first generated token
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
        model->forward_token(next_token, curr_pos, logits);
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

} // extern "C"
