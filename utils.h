#ifndef UTILS_H
#define UTILS_H

#include <immintrin.h>

enum class FFT_type {
    C2C, R2C, C2R
};

void avx512_gather(int fft_size, int batch_size, __m512* simd_arr, float* in_data);
void avx512_scatter(int fft_size, int batch_size, __m512* simd_arr, float* out_data);

#endif
