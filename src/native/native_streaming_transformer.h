#ifndef NATIVE_STREAMING_TRANSFORMER_H
#define NATIVE_STREAMING_TRANSFORMER_H

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NativeStreamingModel NativeStreamingModel;

#define STREAM_PACK_2BIT 0
#define STREAM_PACK_RADIX3 1

#define STREAM_LEVEL_BLOCK 0
#define STREAM_LEVEL_LAYER 1

/**
 * Creates and initializes the native memory-bounded streaming transformer.
 * pack_mode: 0 = 2-bit (4 trits/byte), 1 = Radix-3 (5 trits/byte)
 * stream_level: 0 = Block-level (133 KB scratchpad), 1 = Layer-level (41 KB scratchpad)
 */
NativeStreamingModel* create_native_streaming_model(int pack_mode, int stream_level);

/**
 * Frees streaming model memory.
 */
void free_native_streaming_model(NativeStreamingModel* model);

/**
 * Sets persistent embedding tables.
 */
void streaming_set_embeddings(
    NativeStreamingModel* model,
    const int8_t* wte_q, float wte_s,
    const int8_t* wpe_q, float wpe_s
);

/**
 * Sets persistent LayerNorm and bias parameters for a block.
 */
void streaming_set_block_params(
    NativeStreamingModel* model,
    int block_idx,
    const float* ln1_w, const float* ln1_b,
    const float* ln2_w, const float* ln2_b,
    const float* fc1_b, const float* fc2_b
);

/**
 * Sets persistent per-channel scales for a layer in a block.
 * layer_type: 0=q, 1=k, 2=v, 3=out, 4=fc1, 5=fc2
 */
void streaming_set_layer_scales(
    NativeStreamingModel* model,
    int block_idx,
    int layer_type,
    const float* s1,
    const float* s2,
    int M,
    int N
);

/**
 * Registers compressed / external streaming source pointers for a layer.
 * In a memory-mapped file, these pointers refer to mapped storage.
 */
void streaming_register_layer_weights(
    NativeStreamingModel* model,
    int block_idx,
    int layer_type,
    const uint8_t* p1_source,
    const uint8_t* p2_source,
    int byte_count
);

/**
 * Sets final LayerNorm and LM Head weights.
 */
void streaming_set_final_head(
    NativeStreamingModel* model,
    const float* ln_f_w, const float* ln_f_b,
    const float* head_s1, const float* head_s2,
    const uint8_t* head_p1, const uint8_t* head_p2,
    int M, int N
);

/**
 * Gets exact active weight allocations in RAM (excluding KV cache and general buffers).
 */
size_t streaming_get_active_weight_ram(NativeStreamingModel* model);

/**
 * Gets exact total model allocations including persistent params, KV cache, and scratch buffers.
 */
size_t streaming_get_total_model_ram(NativeStreamingModel* model);

/**
 * Evaluates single-step forward pass returning logits.
 */
void streaming_forward_step(
    NativeStreamingModel* model,
    int token_id,
    int pos,
    float* logits_out
);

/**
 * Pure native autoregressive token generation with Just-In-Time weight streaming.
 */
int streaming_generate_tokens(
    NativeStreamingModel* model,
    const int32_t* prompt,
    int prompt_len,
    int max_new_tokens,
    int32_t* out_tokens,
    double* out_stats
);

#ifdef __cplusplus
}
#endif

#endif // NATIVE_STREAMING_TRANSFORMER_H
