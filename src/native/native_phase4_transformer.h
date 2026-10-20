#ifndef NATIVE_PHASE4_TRANSFORMER_H
#define NATIVE_PHASE4_TRANSFORMER_H

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NativePhase4Model NativePhase4Model;

#define PHASE4_PACK_2BIT 0
#define PHASE4_PACK_RADIX3 1

#define PHASE4_STREAM_BLOCK 0
#define PHASE4_STREAM_LAYER 1

#define PHASE4_KV_FP32 0
#define PHASE4_KV_INT8 1

// Error codes
#define PHASE4_SUCCESS 0
#define PHASE4_ERR_OPEN -1
#define PHASE4_ERR_MAGIC -2
#define PHASE4_ERR_VERSION -3
#define PHASE4_ERR_READ -4
#define PHASE4_ERR_CRC -5
#define PHASE4_ERR_PARAM -6
#define PHASE4_ERR_DECOMP -7

/**
 * Creates and initializes the native Phase 4.0 transformer.
 * Automatically detects whether container_path is raw (.llstream) or chunk-compressed (.llchunk).
 * kv_mode: 0 = FP32 KV cache (640 KB), 1 = Int8 dynamic per-head quantized KV cache (180 KB).
 */
NativePhase4Model* create_phase4_model(
    const char* container_path,
    int stream_level,
    int kv_mode,
    int* out_status
);

/**
 * Frees model allocations and closes underlying file descriptor.
 */
void free_phase4_model(NativePhase4Model* model);

/**
 * Gets exact active model weight memory in RAM (persistent params + transient scratchpads).
 */
size_t phase4_get_active_weight_ram(NativePhase4Model* model);

/**
 * Gets exact KV cache memory in RAM.
 */
size_t phase4_get_kv_cache_ram(NativePhase4Model* model);

/**
 * Gets total native heap allocations (weights + KV cache + activations).
 */
size_t phase4_get_total_model_ram(NativePhase4Model* model);

/**
 * Returns I/O and decompression statistics.
 */
void phase4_get_telemetry_stats(
    NativePhase4Model* model,
    uint64_t* out_pread_calls,
    uint64_t* out_bytes_streamed,
    uint64_t* out_decomp_calls,
    uint64_t* out_decomp_bytes
);

/**
 * Resets telemetry counters to zero.
 */
void phase4_reset_telemetry_stats(NativePhase4Model* model);

/**
 * Pure native autoregressive token generation with Phase 4.0 streaming engine.
 */
int phase4_generate_tokens(
    NativePhase4Model* model,
    const int32_t* prompt,
    int prompt_len,
    int max_new_tokens,
    int32_t* out_tokens,
    double* out_stats
);

/**
 * Evaluates single token forward step returning logits (256 floats).
 */
int phase4_forward_step(
    NativePhase4Model* model,
    int token_id,
    int pos,
    float* logits_out
);

/**
 * Validates integrity of all streamed blocks in container using CRC32.
 */
int phase4_verify_integrity(NativePhase4Model* model);

#ifdef __cplusplus
}
#endif

#endif // NATIVE_PHASE4_TRANSFORMER_H
