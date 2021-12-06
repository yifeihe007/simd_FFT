#ifndef UTILS_H
#define UTILS_H

#include <immintrin.h>

enum class FFT_type { C2C, R2C, C2R };

void avx2_c2c_gather(int fft_size, int batch_size, __m256 *simd_arr,
                     float *in_data);
void avx2_c2c_scatter(int fft_size, int batch_size, __m256 *simd_arr,
                      float *out_data);
#if defined(__AVX512F__)
void avx512_c2c_gather(int fft_size, int batch_size, __m512 *simd_arr,
                       float *in_data);
void avx512_c2c_scatter(int fft_size, int batch_size, __m512 *simd_arr,
                        float *out_data);
#endif
#endif