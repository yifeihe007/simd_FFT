#include "utils.h"

#include <immintrin.h>

constexpr int nvec_512 = 16;

void avx512_gather(int fft_size, int batch_size, __m512* simd_arr, float* in_data) {
    int num_floats = 2 * fft_size;

  __m512i vIdx = _mm512_set_epi32(
      num_floats * 15, num_floats * 14, num_floats * 13, num_floats * 12, num_floats * 11, num_floats * 10, num_floats * 9,
      num_floats * 8, num_floats * 7, num_floats * 6, num_floats * 5, num_floats * 4, num_floats * 3, num_floats * 2, num_floats, 0);
    for (unsigned i = 0; i < batch_size / nvec_512; i++)
        for (unsigned j = 0; j < num_floats; j++) {
            float* ptr = in_data + j + i * num_floats * nvec_512;
            simd_arr[j + i * num_floats] = _mm512_i32gather_ps(
                vIdx, static_cast<void*>(ptr), 4);
    }
}

void avx512_scatter(int fft_size, int batch_size, __m512* simd_arr, float* out_data) {
    int num_floats = 2 * fft_size;
    for (unsigned i = 0; i < fft_size / nvec_512; i++)
        for (unsigned j = 0; j < num_floats; j++)
            for (unsigned k = 0; k < nvec_512; k++) {
                out_data[i * num_floats * nvec_512 + k * num_floats + j] =
                    simd_arr[j + i * num_floats][k];
            }
}