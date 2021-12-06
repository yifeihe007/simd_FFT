#include <fftw3.h>
#include <cstring>
#include <gtest/gtest.h>
#include <immintrin.h>
#include <iostream>
#include <omp.h>
#include <random>
#include <stdlib.h>
#include <sys/time.h>

#include "dft_c2cf_32.c"
#include "utils.h"

int main(int argc, char *argv[])
{
  // c2c
  const int fft_size = 32;
  const int batch_size = 16;
  __m512 *xt = nullptr;
  __m512 *xf = nullptr;
  constexpr size_t byte_size = 2 * fft_size * batch_size * sizeof(float);

  ::posix_memalign((void **)(&xt), 64, byte_size);
  ::posix_memalign((void **)(&xf), 64, byte_size);

  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_real_distribution<float> dist(-10.0, 10.0);
  // Store values in vector
  std::vector<float> values(2 * fft_size * batch_size);
  for (int i = 0; i < values.size(); ++i) {
    values[i] = dist(generator);
  }
  fftwf_complex *xt_fftw = fftwf_alloc_complex(fft_size * batch_size);
  fftwf_complex *xf_fftw = fftwf_alloc_complex(fft_size * batch_size);

  fftwf_plan plan = fftwf_plan_many_dft(
      1, &fft_size, batch_size, xt_fftw, &fft_size, 1, fft_size, xf_fftw,
      &fft_size, 1, fft_size, FFTW_FORWARD, FFTW_MEASURE);

  for (int i = 0; i < values.size(); i += 2) {
    xt_fftw[i / 2][0] = values[i];
    xt_fftw[i / 2][1] = values[i + 1];
  }

  fftwf_execute(plan);

  std::vector<float> out_array(2 * fft_size * batch_size);
  avx512_gather(fft_size, batch_size, xt, &values[0]);
  dft_codelet_c2cf_32(xt, xt + 1, xf, xf + 1, 2, 2, batch_size / 16,
                            (2 * fft_size), (2 * fft_size));

  avx512_scatter(fft_size, batch_size, xf, &out_array[0]);

  for (int i = 0; i < 2 * fft_size * batch_size; i += 2) {
    assert( std::abs(out_array[i] - xf_fftw[i / 2][0]) < 1e-4 && std::abs(out_array[i + 1] - xf_fftw[i / 2][1]) < 1e-4 );
    std::cerr << "ours[" << i / 2 << "]\t Real: " << out_array[i]
              << ", complex: " << out_array[i + 1] << "\n";
    std::cerr << "FFTW[" << i / 2 << "]\t Real: " << xf_fftw[i / 2][0]
              << ", complex: " << xf_fftw[i / 2][1] << "\n\n";
  }

  return 0;
}
