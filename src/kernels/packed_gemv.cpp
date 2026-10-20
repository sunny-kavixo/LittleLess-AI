#include <cstdint>
#include <vector>
#include <cmath>
#include <cstring>
#include <arm_neon.h>

// 256-entry lookup table mapping packed byte -> 4 trits in {-1.0f, 0.0f, 1.0f}
alignas(64) static float TRIT_FLOAT_LUT[256][4];
static bool lut_initialized = false;

static void init_lut() {
    if (lut_initialized) return;
    for (int b = 0; b < 256; ++b) {
        int c0 = (((b >> 0) & 1) << 1) | ((b >> 1) & 1);
        int c1 = (((b >> 2) & 1) << 1) | ((b >> 3) & 1);
        int c2 = (((b >> 4) & 1) << 1) | ((b >> 5) & 1);
        int c3 = (((b >> 6) & 1) << 1) | ((b >> 7) & 1);
        TRIT_FLOAT_LUT[b][0] = static_cast<float>(c0 - 1);
        TRIT_FLOAT_LUT[b][1] = static_cast<float>(c1 - 1);
        TRIT_FLOAT_LUT[b][2] = static_cast<float>(c2 - 1);
        TRIT_FLOAT_LUT[b][3] = static_cast<float>(c3 - 1);
    }
    lut_initialized = true;
}

extern "C" {

/**
 * Direct Packed-Weight Matrix-Vector Multiplication with ARM NEON SIMD.
 * Computes: y = s1 * (T1 * x) + s2 * (T2 * x)
 * WITHOUT ever allocating or storing dense floating-point weight matrices in memory.
 * 
 * p1, p2: [M, N / 4] packed uint8 bitplanes
 * s1, s2: [M] float scales
 * x:      [N] float input activation vector
 * y:      [M] float output vector
 */
void direct_packed_gemv(
    const uint8_t* __restrict__ p1,
    const uint8_t* __restrict__ p2,
    const float* __restrict__ s1,
    const float* __restrict__ s2,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int N
) {
    init_lut();
    const int bytes_per_row = N / 4;

    for (int i = 0; i < M; ++i) {
        const uint8_t* __restrict__ row_p1 = p1 + i * bytes_per_row;
        const uint8_t* __restrict__ row_p2 = p2 + i * bytes_per_row;

        float32x4_t vacc1 = vdupq_n_f32(0.0f);
        float32x4_t vacc2 = vdupq_n_f32(0.0f);

        int jb = 0;
        // Unroll 2 bytes (8 trits / 8 floats) per iteration
        for (; jb + 2 <= bytes_per_row; jb += 2) {
            int j = jb * 4;
            
            // Chunk 1
            float32x4_t vx0 = vld1q_f32(x + j);
            float32x4_t vt1_0 = vld1q_f32(TRIT_FLOAT_LUT[row_p1[jb]]);
            float32x4_t vt2_0 = vld1q_f32(TRIT_FLOAT_LUT[row_p2[jb]]);
            vacc1 = vmlaq_f32(vacc1, vx0, vt1_0);
            vacc2 = vmlaq_f32(vacc2, vx0, vt2_0);

            // Chunk 2
            float32x4_t vx1 = vld1q_f32(x + j + 4);
            float32x4_t vt1_1 = vld1q_f32(TRIT_FLOAT_LUT[row_p1[jb + 1]]);
            float32x4_t vt2_1 = vld1q_f32(TRIT_FLOAT_LUT[row_p2[jb + 1]]);
            vacc1 = vmlaq_f32(vacc1, vx1, vt1_1);
            vacc2 = vmlaq_f32(vacc2, vx1, vt2_1);
        }

        // Remainder
        for (; jb < bytes_per_row; ++jb) {
            int j = jb * 4;
            float32x4_t vx = vld1q_f32(x + j);
            float32x4_t vt1 = vld1q_f32(TRIT_FLOAT_LUT[row_p1[jb]]);
            float32x4_t vt2 = vld1q_f32(TRIT_FLOAT_LUT[row_p2[jb]]);
            vacc1 = vmlaq_f32(vacc1, vx, vt1);
            vacc2 = vmlaq_f32(vacc2, vx, vt2);
        }

        float sum1 = vaddvq_f32(vacc1);
        float sum2 = vaddvq_f32(vacc2);

        y[i] = s1[i] * sum1 + s2[i] * sum2;
    }
}

/**
 * Baseline: Unpack-once dense GEMV for fair hardware latency comparison on CPU.
 */
void dense_fp32_gemv(
    const float* __restrict__ W,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int N
) {
    for (int i = 0; i < M; ++i) {
        const float* row_w = W + i * N;
        int j = 0;
        
        float32x4_t vacc = vdupq_n_f32(0.0f);
        for (; j + 4 <= N; j += 4) {
            float32x4_t vw = vld1q_f32(row_w + j);
            float32x4_t vx = vld1q_f32(x + j);
            vacc = vmlaq_f32(vacc, vw, vx);
        }
        float acc = vaddvq_f32(vacc);
        
        for (; j < N; ++j) {
            acc += row_w[j] * x[j];
        }
        y[i] = acc;
    }
}

}
