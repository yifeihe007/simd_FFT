#ifndef UTILS_H
#define UTILS_H

#include <immintrin.h>

enum class FFT_type { C2C, R2C, C2R };

#if defined(__AVX__)

void avx2_c2c_gather(int fft_size, int batch_size, __m256 *simd_arr,
                     float *in_data);
void avx2_c2c_scatter(int fft_size, int batch_size, __m256 *simd_arr,
                      float *out_data);
void AVX2DFTc2c(const __m256 *realInput, const __m256 *imaginaryInput,
                __m256 *realOutput, __m256 *imaginaryOutput, int inputstride,
                int outputStride, int batch_size, int batchStrideIn,
                int batchStrideOut, int fft_size);

#endif
#if defined(__AVX512F__)
void avx512_c2c_gather(int fft_size, int batch_size, __m512 *simd_arr,
                       float *in_data);
void avx512_c2c_scatter(int fft_size, int batch_size, __m512 *simd_arr,
                        float *out_data);
void AVX2DFTc2c(const __m512 *realInput, const __m512 *imaginaryInput,
                __m512 *realOutput, __m512 *imaginaryOutput, int inputstride,
                int outputStride, int batch_size, int batchStrideIn,
                int batchStrideOut, int fft_size);
#endif
#endif