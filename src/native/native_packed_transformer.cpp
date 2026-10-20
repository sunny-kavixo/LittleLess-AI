#include "native_packed_transformer.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <chrono>
#include <algorithm>
#include <arm_neon.h>

// 256-entry lookup table mapping 2-bit packed byte -> 4 trits in {-1.0f, 0.0f, 1.0f}
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

// 256-entry lookup table mapping Radix-3 packed byte -> 5 trits in {-1.0f, 0.0f, 1.0f}
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
            float32x4_t vx = vld1q_f32(x + j);
            float32x4_t vt1 = vld1q_f32(TRIT_2BIT_FLOAT_LUT[row_p1[jb]]);
            float32x4_t vt2 = vld1q_f32(TRIT_2BIT_FLOAT_LUT[row_p2[jb]]);
            vacc1 = vmlaq_f32(vacc1, vx, vt1);
            vacc2 = vmlaq_f32(vacc2, vx, vt2);
        }

        float sum1 = vaddvq_f32(vacc1);
        float sum2 = vaddvq_f32(vacc2);

        y[i] = s1[i] * sum1 + s2[i] * sum2;
    }
}

// Direct Radix-3 5-Trit Packed GEMV
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

            // Dot product with 5 floats
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

struct PackedLinear {
    std::vector<uint8_t> p1;
    std::vector<uint8_t> p2;
    std::vector<float> s1;
    std::vector<float> s2;
    std::vector<float> bias;
    int M = 0;
    int N = 0;
    bool has_bias = false;
    int pack_mode = PACK_MODE_2BIT;

    void forward(const float* x, float* y) const {
        if (pack_mode == PACK_MODE_RADIX3) {
            direct_radix3_gemv(p1.data(), p2.data(), s1.data(), s2.data(), x, y, M, N);
        } else {
            direct_packed_gemv_neon(p1.data(), p2.data(), s1.data(), s2.data(), x, y, M, N);
        }
        if (has_bias) {
            for (int i = 0; i < M; ++i) y[i] += bias[i];
        }
    }
};

struct TransformerBlock {
    std::vector<float> ln1_w, ln1_b;
    std::vector<float> ln2_w, ln2_b;
    PackedLinear q_proj;
    PackedLinear k_proj;
    PackedLinear v_proj;
    PackedLinear out_proj;
    PackedLinear mlp_fc1;
    PackedLinear mlp_fc2;

    alignas(64) float k_cache[5][128][32];
    alignas(64) float v_cache[5][128][32];

    void clear_cache() {
        std::memset(k_cache, 0, sizeof(k_cache));
        std::memset(v_cache, 0, sizeof(v_cache));
    }
};

struct NativePackedModel {
    int pack_mode = PACK_MODE_2BIT;
    int d_model = 160;
    int n_heads = 5;
    int d_head = 32;
    int n_blocks = 4;
    int max_context = 128;
    int vocab_size = 256;

    std::vector<int8_t> wte_q;
    float wte_s = 0.0f;
    std::vector<int8_t> wpe_q;
    float wpe_s = 0.0f;

    std::vector<TransformerBlock> blocks;

    std::vector<float> ln_f_w, ln_f_b;
    PackedLinear lm_head;

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

    NativePackedModel(int mode) : pack_mode(mode) {
        blocks.resize(n_blocks);
        init_2bit_trit_lut();
        init_radix3_lut();
        lm_head.pack_mode = mode;
        for (auto& b : blocks) {
            b.q_proj.pack_mode = mode;
            b.k_proj.pack_mode = mode;
            b.v_proj.pack_mode = mode;
            b.out_proj.pack_mode = mode;
            b.mlp_fc1.pack_mode = mode;
            b.mlp_fc2.pack_mode = mode;
        }
    }

    void clear_kv_caches() {
        for (auto& b : blocks) b.clear_cache();
    }

    void forward_token(int token_id, int pos, float* out_logits) {
        const int8_t* wte_row = wte_q.data() + token_id * d_model;
        const int8_t* wpe_row = wpe_q.data() + pos * d_model;
        for (int i = 0; i < d_model; ++i) {
            x[i] = (float)wte_row[i] * wte_s + (float)wpe_row[i] * wpe_s;
        }

        const float scale_factor = 1.0f / std::sqrt((float)d_head);

        for (int bi = 0; bi < n_blocks; ++bi) {
            auto& b = blocks[bi];

            layer_norm(x_norm1, x, b.ln1_w.data(), b.ln1_b.data(), d_model);

            b.q_proj.forward(x_norm1, q_buf);
            b.k_proj.forward(x_norm1, k_buf);
            b.v_proj.forward(x_norm1, v_buf);

            for (int h = 0; h < n_heads; ++h) {
                std::memcpy(&b.k_cache[h][pos][0], k_buf + (h * d_head), sizeof(float) * d_head);
                std::memcpy(&b.v_cache[h][pos][0], v_buf + (h * d_head), sizeof(float) * d_head);
            }

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

            float out_proj_buf[160];
            b.out_proj.forward(y_attn, out_proj_buf);

            for (int i = 0; i < d_model; ++i) {
                x[i] += out_proj_buf[i];
            }

            layer_norm(x_norm2, x, b.ln2_w.data(), b.ln2_b.data(), d_model);

            b.mlp_fc1.forward(x_norm2, h_mlp);
            gelu_inplace(h_mlp, 512);

            b.mlp_fc2.forward(h_mlp, out_mlp);

            for (int i = 0; i < d_model; ++i) {
                x[i] += out_mlp[i];
            }
        }

        layer_norm(x_f, x, ln_f_w.data(), ln_f_b.data(), d_model);
        lm_head.forward(x_f, out_logits);
    }
};

extern "C" {

NativePackedModel* create_native_packed_model(int pack_mode) {
    return new NativePackedModel(pack_mode);
}

void free_native_packed_model(NativePackedModel* model) {
    if (model) delete model;
}

void set_embeddings(
    NativePackedModel* model,
    const int8_t* wte_q, float wte_s,
    const int8_t* wpe_q, float wpe_s
) {
    model->wte_q.assign(wte_q, wte_q + 256 * 160);
    model->wte_s = wte_s;
    model->wpe_q.assign(wpe_q, wpe_q + 128 * 160);
    model->wpe_s = wpe_s;
}

void set_block_norms(
    NativePackedModel* model,
    int block_idx,
    const float* ln1_w, const float* ln1_b,
    const float* ln2_w, const float* ln2_b
) {
    auto& b = model->blocks[block_idx];
    b.ln1_w.assign(ln1_w, ln1_w + 160);
    b.ln1_b.assign(ln1_b, ln1_b + 160);
    b.ln2_w.assign(ln2_w, ln2_w + 160);
    b.ln2_b.assign(ln2_b, ln2_b + 160);
}

void set_block_linear(
    NativePackedModel* model,
    int block_idx,
    int layer_type,
    const uint8_t* p1,
    const uint8_t* p2,
    const float* s1,
    const float* s2,
    const float* bias,
    int M,
    int N
) {
    auto& b = model->blocks[block_idx];
    PackedLinear* target = nullptr;
    if (layer_type == 0) target = &b.q_proj;
    else if (layer_type == 1) target = &b.k_proj;
    else if (layer_type == 2) target = &b.v_proj;
    else if (layer_type == 3) target = &b.out_proj;
    else if (layer_type == 4) target = &b.mlp_fc1;
    else if (layer_type == 5) target = &b.mlp_fc2;

    if (!target) return;
    target->M = M;
    target->N = N;
    target->pack_mode = model->pack_mode;
    int bytes_per_row = (model->pack_mode == PACK_MODE_RADIX3) ? ((N + 4) / 5) : (N / 4);
    target->p1.assign(p1, p1 + M * bytes_per_row);
    target->p2.assign(p2, p2 + M * bytes_per_row);
    target->s1.assign(s1, s1 + M);
    target->s2.assign(s2, s2 + M);
    if (bias != nullptr) {
        target->bias.assign(bias, bias + M);
        target->has_bias = true;
    } else {
        target->has_bias = false;
    }
}

void set_final_head(
    NativePackedModel* model,
    const float* ln_f_w, const float* ln_f_b,
    const uint8_t* p1,
    const uint8_t* p2,
    const float* s1,
    const float* s2,
    int M,
    int N
) {
    model->ln_f_w.assign(ln_f_w, ln_f_w + 160);
    model->ln_f_b.assign(ln_f_b, ln_f_b + 160);
    model->lm_head.M = M;
    model->lm_head.N = N;
    model->lm_head.pack_mode = model->pack_mode;
    int bytes_per_row = (model->pack_mode == PACK_MODE_RADIX3) ? ((N + 4) / 5) : (N / 4);
    model->lm_head.p1.assign(p1, p1 + M * bytes_per_row);
    model->lm_head.p2.assign(p2, p2 + M * bytes_per_row);
    model->lm_head.s1.assign(s1, s1 + M);
    model->lm_head.s2.assign(s2, s2 + M);
    model->lm_head.has_bias = false;
}

void native_forward_step(
    NativePackedModel* model,
    int token_id,
    int pos,
    float* logits_out
) {
    model->forward_token(token_id, pos, logits_out);
}

int native_generate_tokens(
    NativePackedModel* model,
    const int32_t* prompt,
    int prompt_len,
    int max_new_tokens,
    int32_t* out_tokens,
    double* out_stats
) {
    model->clear_kv_caches();
    float logits[256];

    auto t_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < prompt_len; ++i) {
        model->forward_token(prompt[i], i, logits);
    }
    auto t_prefill_done = std::chrono::high_resolution_clock::now();

    int next_tok = 0;
    float max_log = logits[0];
    for (int v = 1; v < 256; ++v) {
        if (logits[v] > max_log) {
            max_log = logits[v];
            next_tok = v;
        }
    }
    out_tokens[0] = next_tok;

    auto t_decode_start = std::chrono::high_resolution_clock::now();
    for (int step = 1; step < max_new_tokens; ++step) {
        int pos = prompt_len + step - 1;
        if (pos >= model->max_context) break;

        model->forward_token(next_tok, pos, logits);

        max_log = logits[0];
        next_tok = 0;
        for (int v = 1; v < 256; ++v) {
            if (logits[v] > max_log) {
                max_log = logits[v];
                next_tok = v;
            }
        }
        out_tokens[step] = next_tok;
    }
    auto t_decode_done = std::chrono::high_resolution_clock::now();

    double prefill_ms = std::chrono::duration<double, std::milli>(t_prefill_done - t_start).count();
    double decode_ms = std::chrono::duration<double, std::milli>(t_decode_done - t_decode_start).count();
    double total_ms = prefill_ms + decode_ms;
    double tok_per_sec = (max_new_tokens > 1 && decode_ms > 0) ? ((double)(max_new_tokens - 1) / (decode_ms / 1000.0)) : 0.0;

    if (out_stats) {
        out_stats[0] = prefill_ms;
        out_stats[1] = decode_ms;
        out_stats[2] = total_ms;
        out_stats[3] = tok_per_sec;
    }

    return max_new_tokens;
}

}
