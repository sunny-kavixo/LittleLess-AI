#ifndef NATIVE_FILE_STREAMING_TRANSFORMER_H
#define NATIVE_FILE_STREAMING_TRANSFORMER_H

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NativeFileStreamingModel NativeFileStreamingModel;

#define STREAM_PACK_2BIT 0
#define STREAM_PACK_RADIX3 1

#define STREAM_LEVEL_BLOCK 0
#define STREAM_LEVEL_LAYER 1

// Error codes
#define STREAM_SUCCESS 0
#define STREAM_ERR_OPEN -1
#define STREAM_ERR_MAGIC -2
#define STREAM_ERR_VERSION -3
#define STREAM_ERR_READ -4
#define STREAM_ERR_CRC -5
#define STREAM_ERR_PARAM -6

/**
 * Creates and initializes the native file-backed streaming transformer directly from disk.
 * Opens the binary container, reads and validates headers and persistent non-block weights,
 * and initializes transient scratchpads.
 * Returns nullptr on failure.
 */
NativeFileStreamingModel* create_file_streaming_model(
    const char* container_path,
    int stream_level,
    int* out_status
);

/**
 * Closes the underlying file descriptor and frees all model memory.
 */
void free_file_streaming_model(NativeFileStreamingModel* model);

/**
 * Gets exact active model weight memory in RAM (persistent non-block params + scratchpad).
 * Note: Does not include KV cache or activations.
 */
size_t file_streaming_get_active_weight_ram(NativeFileStreamingModel* model);

/**
 * Gets total native heap allocations including persistent params, transient scratchpad,
 * KV cache (for context=128), and static activation vectors.
 */
size_t file_streaming_get_total_model_ram(NativeFileStreamingModel* model);

/**
 * Returns I/O statistics: total pread calls and total bytes streamed from disk.
 */
void file_streaming_get_io_stats(
    NativeFileStreamingModel* model,
    uint64_t* out_pread_calls,
    uint64_t* out_bytes_read
);

/**
 * Resets I/O counters to zero.
 */
void file_streaming_reset_io_stats(NativeFileStreamingModel* model);

/**
 * Evaluates a single token forward step returning logits (256 floats).
 */
int file_streaming_forward_step(
    NativeFileStreamingModel* model,
    int token_id,
    int pos,
    float* logits_out
);

/**
 * Pure native autoregressive token generation with true file-backed weight streaming.
 */
int file_streaming_generate_tokens(
    NativeFileStreamingModel* model,
    const int32_t* prompt,
    int prompt_len,
    int max_new_tokens,
    int32_t* out_tokens,
    double* out_stats
);

/**
 * Validates integrity of all streamed blocks in the file container using CRC32.
 * Returns STREAM_SUCCESS (0) if all checksums match, or error code on failure.
 */
int file_streaming_verify_integrity(NativeFileStreamingModel* model);

#ifdef __cplusplus
}
#endif

#endif // NATIVE_FILE_STREAMING_TRANSFORMER_H
