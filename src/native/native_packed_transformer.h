#ifndef NATIVE_PACKED_TRANSFORMER_H
#define NATIVE_PACKED_TRANSFORMER_H

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NativePackedModel NativePackedModel;

#define PACK_MODE_2BIT 0
#define PACK_MODE_RADIX3 1

/**
 * Creates and initializes the native packed transformer model.
 * pack_mode: 0 = 2-bit (4 trits/byte), 1 = Radix-3 (5 trits/byte)
 */
NativePackedModel* create_native_packed_model(int pack_mode);

/**
 * Frees model memory.
 */
void free_native_packed_model(NativePackedModel* model);

/**
 * Sets embedding tables.
 */
void set_embeddings(
    NativePackedModel* model,
    const int8_t* wte_q, float wte_s,
    const int8_t* wpe_q, float wpe_s
);

/**
 * Sets LayerNorm parameters for a block.
 */
void set_block_norms(
    NativePackedModel* model,
    int block_idx,
    const float* ln1_w, const float* ln1_b,
    const float* ln2_w, const float* ln2_b
);

/**
 * Sets packed dual-basis linear weights for a layer in a block.
 * layer_type: 0=q, 1=k, 2=v, 3=out, 4=fc1, 5=fc2
 */
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
);

/**
 * Sets final LayerNorm and LM Head weights.
 */
void set_final_head(
    NativePackedModel* model,
    const float* ln_f_w, const float* ln_f_b,
    const uint8_t* p1,
    const uint8_t* p2,
    const float* s1,
    const float* s2,
    int M,
    int N
);

/**
 * Evaluates single-step forward pass returning logits for verification.
 * logits_out: destination buffer of size 256
 */
void native_forward_step(
    NativePackedModel* model,
    int token_id,
    int pos,
    float* logits_out
);

/**
 * Pure native autoregressive token generation.
 * Executes the entire multi-token decode loop in native C++ with persistent KV caching!
 * 
 * prompt: array of input token IDs [0, 255]
 * prompt_len: number of prompt tokens
 * max_new_tokens: number of tokens to generate
 * out_tokens: destination buffer for generated tokens (size >= max_new_tokens)
 * out_stats: [prefill_time_ms, decode_time_ms, total_time_ms, tokens_per_sec]
 */
int native_generate_tokens(
    NativePackedModel* model,
    const int32_t* prompt,
    int prompt_len,
    int max_new_tokens,
    int32_t* out_tokens,
    double* out_stats
);

#ifdef __cplusplus
}
#endif

#endif // NATIVE_PACKED_TRANSFORMER_H
