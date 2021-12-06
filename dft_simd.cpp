/*

   dft_simd/dft_simd.cpp -- Stephen Fegan -- 2018-02-19

   Test drive for FFTW speed tests and for SIMD genfft codelets

   Copyright 2018, Stephen Fegan <sfegan@llr.in2p3.fr>
   LLR, Ecole Polytechnique, CNRS/IN2P3

   This file is part of "dft_simd"

   "dft_simd" is free software: you can redistribute it and/or modify it
   under the terms of the GNU General Public License version 2 or
   later, as published by the Free Software Foundation.

   "dft_simd" is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   General Public License for more details.

*/

#include <chrono>
#include <fftw3.h>
#include <gtest/gtest.h>
#include <immintrin.h>
#include <iostream>
#include <omp.h>
#include <random>
#include <stdlib.h>
#include <sys/time.h>

double cpuSecond() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

constexpr int nvec = 8;
constexpr int nvec_512 = 16;
constexpr int Iter = 1000;

TEST(TestDFT, manyc2cFFTW_Aligned_One) {
  char *nsampVar;
  char *nloopVar;
  nsampVar = getenv("NSAMP");
  nloopVar = getenv("NLOOP");
  int nsamp = atoi(nsampVar);
  int nloop = atoi(nloopVar);
  int ompT = omp_get_max_threads();

  fftwf_init_threads();
  fftwf_plan_with_nthreads(ompT);

  fftwf_complex *xt = fftwf_alloc_complex(nsamp * nloop);
  fftwf_complex *xf = fftwf_alloc_complex(nsamp * nloop);

  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_real_distribution<float> dist(-10.0, 10.0);
  // Store values in vector
  std::vector<float> values(2 * nsamp * nloop);
  for (int i = 0; i < values.size(); ++i) {
    values[i] = dist(generator);
  }
  for (int i = 0; i < values.size(); i += 2) {
    xt[i / 2][0] = values[i];
    xt[i / 2][1] = values[i + 1];
  }

  fftwf_plan plan =
      fftwf_plan_many_dft(1, &nsamp, nloop, xt, nullptr, 1, nsamp, xf, nullptr,
                          1, nsamp, FFTW_FORWARD, FFTW_MEASURE);

  using namespace std::chrono;
  high_resolution_clock::time_point iStart = high_resolution_clock::now();

  for (int i = 0; i < Iter; i++)
    fftwf_execute(plan);

  high_resolution_clock::time_point iFinished = high_resolution_clock::now();
  duration<double, std::milli> iElaps = iFinished - iStart;

  std::cout << "Ec2cFFTW : nsamp : " << nsamp << " nloop : " << nloop
            << " ompT : " << ompT << " iElaps : " << iElaps.count() << " .\n";

  fftwf_destroy_plan(plan);
  fftwf_cleanup_threads();

  free(xt);
  free(xf);

  // omp_destroy_lock(&writelock);
}
TEST(TestDFT, manyr2cFFTW_Aligned_One) {

  char *nsampVar;
  char *nloopVar;
  nsampVar = getenv("NSAMP");
  nloopVar = getenv("NLOOP");
  int nsamp = atoi(nsampVar);
  int nloop = atoi(nloopVar);
  int ompT = omp_get_max_threads();
  fftwf_init_threads();
  fftwf_plan_with_nthreads(ompT);

  // omp_lock_t writelock;
  // omp_init_lock(&writelock);

  float *xt = fftwf_alloc_real((nsamp / 2 + 1) * 2 * nloop);
  float *xf = fftwf_alloc_real((nsamp / 2 + 1) * 2 * nloop);

  fftwf_plan plan = fftwf_plan_many_dft_r2c(
      1, &nsamp, nloop, xt, nullptr, 1, (nsamp / 2 + 1) * 2,
      (fftwf_complex *)xf, nullptr, 1, nsamp / 2 + 1, FFTW_MEASURE);

  double iStart = cpuSecond();

  for (int i = 0; i < Iter; i++)
    fftwf_execute(plan);
  double iElaps = cpuSecond() - iStart;
  printf("Er2cFFTW : nsamp : %d nloop : %d ompT : %d iElaps : %f \n", nsamp,
         nloop, ompT, iElaps * 1000);

  fftwf_destroy_plan(plan);
  fftwf_cleanup_threads();
  fftwf_free(xf);
  fftwf_free(xt);

  // omp_destroy_lock(&writelock);
}

TEST(TestDFT, manyc2rFFTW_Aligned_One) {

  char *nsampVar;
  char *nloopVar;
  nsampVar = getenv("NSAMP");
  nloopVar = getenv("NLOOP");
  int nsamp = atoi(nsampVar);
  int nloop = atoi(nloopVar);
  int ompT = omp_get_max_threads();
  fftwf_init_threads();
  fftwf_plan_with_nthreads(ompT);

  // omp_lock_t writelock;
  // omp_init_lock(&writelock);

  float *xt = fftwf_alloc_real((nsamp / 2 + 1) * 2 * nloop);
  float *xf = fftwf_alloc_real((nsamp / 2 + 1) * 2 * nloop);

  fftwf_plan plan = fftwf_plan_many_dft_c2r(
      1, &nsamp, nloop, (fftwf_complex *)xt, nullptr, 1, (nsamp / 2 + 1), xf,
      nullptr, 1, (nsamp / 2 + 1) * 2, FFTW_MEASURE);

  double iStart = cpuSecond();

  for (int i = 0; i < Iter; i++)
    fftwf_execute(plan);
  double iElaps = cpuSecond() - iStart;
  printf("Ec2rFFTW : nsamp : %d nloop : %d ompT : %d iElaps : %f \n", nsamp,
         nloop, ompT, iElaps * 1000);

  fftwf_destroy_plan(plan);
  fftwf_cleanup_threads();
  fftwf_free(xf);
  fftwf_free(xt);

  // omp_destroy_lock(&writelock);
}

#if defined(__AVX__)

#include "avx2_intrinsics.h"
inline void MAKE_VOLATILE_STRIDE(int a, int b) {}

namespace m256 {

using E = __m256;
using R = __m256;

inline int WS(const stride s, const stride i) { return s * i; }

#include "dft_c2cf_1024.c"
#include "dft_c2cf_128.c"
#include "dft_c2cf_256.c"
#include "dft_c2cf_32.c"
#include "dft_c2cf_512.c"
#include "dft_c2cf_64.c"
#include "dft_c2r_1024.c"
#include "dft_c2r_128.c"
#include "dft_c2r_256.c"
#include "dft_c2r_32.c"
#include "dft_c2r_512.c"
#include "dft_c2r_64.c"
#include "dft_r2cf_1024.c"
#include "dft_r2cf_128.c"
#include "dft_r2cf_256.c"
#include "dft_r2cf_32.c"
#include "dft_r2cf_512.c"
#include "dft_r2cf_64.c"

} // namespace m256

TEST(TestDFT, AVX2r2c) {

  char *nsampVar;
  char *nloopVar;
  nsampVar = getenv("NSAMP");
  nloopVar = getenv("NLOOP");
  int nsamp = atoi(nsampVar);
  int nloop = atoi(nloopVar);
  int ompT = omp_get_max_threads();

  int num = (nsamp / 2 + 1) * 2;

  __m256 *xt = nullptr;
  __m256 *xf = nullptr;
  float *in_data = fftwf_alloc_real(num * nloop);
  float *out_data = fftwf_alloc_real(num * nloop);
  ::posix_memalign((void **)&xt, 32,
                   nloop * 2 * (nsamp / 2 + 1) * sizeof(float));
  ::posix_memalign((void **)&xf, 32,
                   nloop * 2 * (nsamp / 2 + 1) * sizeof(float));

  double iStart = cpuSecond();

  __m256i vIdx = _mm256_set_epi32(num * 7, num * 6, num * 5, num * 4, num * 3,
                                  num * 2, num, 0);

  for (unsigned i = 0; i < nloop / nvec; i++)

    for (unsigned j = 0; j < num; j++) {

      xt[j + i * num] = _mm256_i32gather_ps(
          (static_cast<float *>(in_data) + j + i * num * nvec), vIdx, 4);
    }

  double afterGather = cpuSecond();

  for (int i = 0; i < Iter; i++) {
    switch (nsamp) {
    case 32:
      m256::dft_codelet_r2cf_32(xt, xt + 1, xf, xf + 1, 2, 2, 2, nloop / nvec,
                                num, num);
      break;
    case 64:
      m256::dft_codelet_r2cf_64(xt, xt + 1, xf, xf + 1, 2, 2, 2, nloop / nvec,
                                num, num);
      break;
    case 128:
      m256::dft_codelet_r2cf_128(xt, xt + 1, xf, xf + 1, 2, 2, 2, nloop / nvec,
                                 num, num);
      break;
    case 256:
      m256::dft_codelet_r2cf_256(xt, xt + 1, xf, xf + 1, 2, 2, 2, nloop / nvec,
                                 num, num);
      break;
    case 512:
      m256::dft_codelet_r2cf_512(xt, xt + 1, xf, xf + 1, 2, 2, 2, nloop / nvec,
                                 num, num);
      break;
    case 1024:
      m256::dft_codelet_r2cf_1024(xt, xt + 1, xf, xf + 1, 2, 2, 2, nloop / nvec,
                                  num, num);
      break;
    }
  }
  double afterCodelet = cpuSecond();

  for (unsigned i = 0; i < nloop / nvec; i++)
    for (unsigned j = 0; j < num; j++)
      for (unsigned k = 0; k < nvec; k++) {
        static_cast<float *>(out_data)[i * num * nvec + k * num + j] =
            xf[j + i * num][k];
      }
  double afterScatter = cpuSecond();
  double gather = afterGather - iStart;
  double codelet = afterCodelet - afterGather;
  double scatter = afterScatter - afterCodelet;
  printf("EAVX2r2c : nsamp : %d nloop : %d ompT : %d gather : %f codelet : %f "
         "scatter : %f \n",
         nsamp, nloop, ompT, gather * 1000, codelet * 1000, scatter * 1000);

  ::free(xf);
  ::free(xt);
  fftwf_free(in_data);
  fftwf_free(out_data);
}
TEST(TestDFT, AVX2c2r) {
  char *nsampVar;
  char *nloopVar;
  nsampVar = getenv("NSAMP");
  nloopVar = getenv("NLOOP");
  int nsamp = atoi(nsampVar);
  int nloop = atoi(nloopVar);
  int ompT = omp_get_max_threads();
  int num = (nsamp / 2 + 1) * 2;

  __m256 *xt = nullptr;
  __m256 *xf = nullptr;
  float *in_data = fftwf_alloc_real(num * nloop);
  float *out_data = fftwf_alloc_real(num * nloop);
  ::posix_memalign((void **)&xt, 32,
                   nloop * 2 * (nsamp / 2 + 1) * sizeof(float));
  ::posix_memalign((void **)&xf, 32,
                   nloop * 2 * (nsamp / 2 + 1) * sizeof(float));

  double iStart = cpuSecond();

  __m256i vIdx = _mm256_set_epi32(num * 7, num * 6, num * 5, num * 4, num * 3,
                                  num * 2, num, 0);

  for (unsigned i = 0; i < nloop / nvec; i++)

    for (unsigned j = 0; j < num; j++) {

      xt[j + i * num] = _mm256_i32gather_ps(
          (static_cast<float *>(in_data) + j + i * num * nvec), vIdx, 4);
    }

  double afterGather = cpuSecond();

  for (int i = 0; i < Iter; i++) {
    switch (nsamp) {
    case 32:
      m256::dft_codelet_c2r_32(xt, xt + 1, xf, xf + 1, 2, 2, 2, nloop / nvec,
                               num, num);
      break;
    case 64:
      m256::dft_codelet_c2r_64(xt, xt + 1, xf, xf + 1, 2, 2, 2, nloop / nvec,
                               num, num);
      break;
    case 128:
      m256::dft_codelet_c2r_128(xt, xt + 1, xf, xf + 1, 2, 2, 2, nloop / nvec,
                                num, num);
      break;
    case 256:
      m256::dft_codelet_c2r_256(xt, xt + 1, xf, xf + 1, 2, 2, 2, nloop / nvec,
                                num, num);
      break;
    case 512:
      m256::dft_codelet_c2r_512(xt, xt + 1, xf, xf + 1, 2, 2, 2, nloop / nvec,
                                num, num);
      break;
    case 1024:
      m256::dft_codelet_c2r_1024(xt, xt + 1, xf, xf + 1, 2, 2, 2, nloop / nvec,
                                 num, num);
      break;
    }
  }
  double afterCodelet = cpuSecond();
  for (unsigned i = 0; i < nloop / nvec; i++)
    for (unsigned j = 0; j < num; j++)
      for (unsigned k = 0; k < nvec; k++) {
        static_cast<float *>(out_data)[i * num * nvec + k * num + j] =
            xf[j + i * num][k];
      }

  double afterScatter = cpuSecond();
  double gather = afterGather - iStart;
  double codelet = afterCodelet - afterGather;
  double scatter = afterScatter - afterCodelet;
  printf("EAVX2c2r : nsamp : %d nloop : %d ompT : %d gather : %f codelet : %f "
         "scatter : %f \n",
         nsamp, nloop, ompT, gather * 1000, codelet * 1000, scatter * 1000);
  ::free(xf);
  ::free(xt);
  fftwf_free(in_data);
  fftwf_free(out_data);
}
TEST(TestDFT, AVX2c2c) {
  char *nsampVar;
  char *nloopVar;
  nsampVar = getenv("NSAMP");
  nloopVar = getenv("NLOOP");
  int nsamp = atoi(nsampVar);
  int nloop = atoi(nloopVar);
  int ompT = omp_get_max_threads();
  int num = nsamp * 2;

  // for (int i = 0; i < Iter; i++) {
  __m256 *xt = nullptr;
  __m256 *xf = nullptr;
  ::posix_memalign((void **)&xt, 32, nloop * num * sizeof(float));
  ::posix_memalign((void **)&xf, 32, nloop * num * sizeof(float));

  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_real_distribution<float> dist(-10.0, 10.0);
  // Store values in vector
  std::vector<float> values(2 * nsamp * nloop);
  std::vector<float> out_array(2 * nsamp * nloop);
  for (int i = 0; i < values.size(); ++i) {
    values[i] = dist(generator);
  }

  using namespace std::chrono;
  high_resolution_clock::time_point iStart = high_resolution_clock::now();

  __m256i vIdx = _mm256_set_epi32(num * 7, num * 6, num * 5, num * 4, num * 3,
                                  num * 2, num, 0);
  for (unsigned i = 0; i < nloop / nvec; i++)
    for (unsigned j = 0; j < num; j++) {
      float *ptr = &values[0] + j + i * num * nvec;
      xt[j + i * num] = _mm256_i32gather_ps(ptr, vIdx, 4);
    }

  high_resolution_clock::time_point afterGather = high_resolution_clock::now();

  for (int i = 0; i < Iter; i++) {
    switch (nsamp) {
    case 32:
      m256::dft_codelet_c2cf_32(xt, xt + 1, xf, xf + 1, 2, 2, nloop / nvec, num,
                                num);
      break;
    case 64:
      m256::dft_codelet_c2cf_64(xt, xt + 1, xf, xf + 1, 2, 2, nloop / nvec, num,
                                num);
      break;
    case 128:
      m256::dft_codelet_c2cf_128(xt, xt + 1, xf, xf + 1, 2, 2, nloop / nvec,
                                 num, num);
      break;
    case 256:
      m256::dft_codelet_c2cf_256(xt, xt + 1, xf, xf + 1, 2, 2, nloop / nvec,
                                 num, num);
      break;
    case 512:
      m256::dft_codelet_c2cf_512(xt, xt + 1, xf, xf + 1, 2, 2, nloop / nvec,
                                 num, num);
      break;
    case 1024:
      m256::dft_codelet_c2cf_1024(xt, xt + 1, xf, xf + 1, 2, 2, nloop / nvec,
                                  num, num);
      break;
    }
  }
  high_resolution_clock::time_point afterCodelet = high_resolution_clock::now();

  for (unsigned i = 0; i < nloop / nvec; i++)
    for (unsigned j = 0; j < num; j++)
      for (unsigned k = 0; k < nvec; k++) {
        out_array[i * num * nvec + k * num + j] = xf[j + i * num][k];
      }
  high_resolution_clock::time_point afterScatter = high_resolution_clock::now();

  duration<double, std::milli> gather = afterGather - iStart;
  duration<double, std::milli> codelet = afterCodelet - afterGather;
  duration<double, std::milli> scatter = afterScatter - afterCodelet;

  std::cout << "EAVX2c2c : nsamp : " << nsamp << " nloop : " << nloop
            << " ompT : " << ompT << " gather : " << gather.count()
            << " codelet : " << codelet.count()
            << " scatter : " << scatter.count() << " .\n ";

  ::free(xf);
  ::free(xt);
}
#undef DK
#endif

#if defined(__AVX512F__)

#include "avx512_intrinsics.h"

namespace m512 {

using E = __m512;
using R = __m512;

inline int WS(const stride s, const stride i) { return s * i; }

#include "dft_c2cf_1024.c"
#include "dft_c2cf_128.c"
#include "dft_c2cf_256.c"
#include "dft_c2cf_32.c"
#include "dft_c2cf_512.c"
#include "dft_c2cf_64.c"
#include "dft_c2r_1024.c"
#include "dft_c2r_128.c"
#include "dft_c2r_256.c"
#include "dft_c2r_32.c"
#include "dft_c2r_512.c"
#include "dft_c2r_64.c"
#include "dft_r2cf_1024.c"
#include "dft_r2cf_128.c"
#include "dft_r2cf_256.c"
#include "dft_r2cf_32.c"
#include "dft_r2cf_512.c"
#include "dft_r2cf_64.c"

} // namespace m512

TEST(TestDFT, AVX512r2c) {

  char *nsampVar;
  char *nloopVar;
  nsampVar = getenv("NSAMP");
  nloopVar = getenv("NLOOP");
  int nsamp = atoi(nsampVar);
  int nloop = atoi(nloopVar);
  // for (int i = 0; i < Iter; i++) {
  __m512 *xt = nullptr;
  __m512 *xf = nullptr;
  ::posix_memalign((void **)&xt, 64,
                   nloop * 2 * (nsamp / 2 + 1) * sizeof(float));
  ::posix_memalign((void **)&xf, 64,
                   nloop * 2 * (nsamp / 2 + 1) * sizeof(float));

  int ompT = omp_get_max_threads();

  double iStart = cpuSecond();

  for (int i = 0; i < Iter; i++) {
    switch (nsamp) {
    case 32:
      m512::dft_codelet_r2cf_32(xt, xt + 1, xf, xf + 1, 1, 1, 1, nloop / 16,
                                (nsamp / 2 + 1) * 2, (nsamp / 2 + 1) * 2);
      break;
    case 64:
      m512::dft_codelet_r2cf_64(xt, xt + 1, xf, xf + 1, 1, 1, 1, nloop / 16,
                                (nsamp / 2 + 1) * 2, (nsamp / 2 + 1) * 2);
      break;
    case 128:
      m512::dft_codelet_r2cf_128(xt, xt + 1, xf, xf + 1, 1, 1, 1, nloop / 16,
                                 (nsamp / 2 + 1) * 2, (nsamp / 2 + 1) * 2);
      break;
    case 256:
      m512::dft_codelet_r2cf_256(xt, xt + 1, xf, xf + 1, 1, 1, 1, nloop / 16,
                                 (nsamp / 2 + 1) * 2, (nsamp / 2 + 1) * 2);
      break;
    case 512:
      m512::dft_codelet_r2cf_512(xt, xt + 1, xf, xf + 1, 1, 1, 1, nloop / 16,
                                 (nsamp / 2 + 1) * 2, (nsamp / 2 + 1) * 2);
      break;
    case 1024:
      m512::dft_codelet_r2cf_1024(xt, xt + 1, xf, xf + 1, 1, 1, 1, nloop / 16,
                                  (nsamp / 2 + 1) * 2, (nsamp / 2 + 1) * 2);
      break;
    }
  }
  double iElaps = cpuSecond() - iStart;
  printf("EAVX512r2c : nsamp : %d nloop : %d ompT : %d iElaps : %f \n", nsamp,
         nloop, ompT, iElaps * 1000);
  /*  if (iloop == 0)
  {
    for (int ifreq = 0; ifreq < 2 * (nsamp / 2 + 1); ifreq++)
    {
      std::cout << xf[ifreq][0] << ' ';
    }
    std::cout << '\n';
  }
//}*/
  ::free(xf);
  ::free(xt);
}

TEST(TestDFT, AVX512c2r) {
  char *nsampVar;
  char *nloopVar;
  nsampVar = getenv("NSAMP");
  nloopVar = getenv("NLOOP");
  int nsamp = atoi(nsampVar);
  int nloop = atoi(nloopVar);
  // for (int i = 0; i < Iter; i++) {
  __m512 *xt = nullptr;
  __m512 *xf = nullptr;
  ::posix_memalign((void **)&xt, 64,
                   nloop * 2 * (nsamp / 2 + 1) * sizeof(float));
  ::posix_memalign((void **)&xf, 64,
                   nloop * 2 * (nsamp / 2 + 1) * sizeof(float));

  int ompT = omp_get_max_threads();

  double iStart = cpuSecond();
  for (int i = 0; i < Iter; i++) {
    switch (nsamp) {
    case 32:
      m512::dft_codelet_c2r_32(xt, xt + 1, xf, xf + 1, 1, 1, 1, nloop / 16,
                               (nsamp / 2 + 1) * 2, (nsamp / 2 + 1) * 2);
      break;
    case 64:
      m512::dft_codelet_c2r_64(xt, xt + 1, xf, xf + 1, 1, 1, 1, nloop / 16,
                               (nsamp / 2 + 1) * 2, (nsamp / 2 + 1) * 2);
      break;
    case 128:
      m512::dft_codelet_c2r_128(xt, xt + 1, xf, xf + 1, 1, 1, 1, nloop / 16,
                                (nsamp / 2 + 1) * 2, (nsamp / 2 + 1) * 2);
      break;
    case 256:
      m512::dft_codelet_c2r_256(xt, xt + 1, xf, xf + 1, 1, 1, 1, nloop / 16,
                                (nsamp / 2 + 1) * 2, (nsamp / 2 + 1) * 2);
      break;
    case 512:
      m512::dft_codelet_c2r_512(xt, xt + 1, xf, xf + 1, 1, 1, 1, nloop / 16,
                                (nsamp / 2 + 1) * 2, (nsamp / 2 + 1) * 2);
      break;
    case 1024:
      m512::dft_codelet_c2r_1024(xt, xt + 1, xf, xf + 1, 1, 1, 1, nloop / 16,
                                 (nsamp / 2 + 1) * 2, (nsamp / 2 + 1) * 2);
      break;
    }
  }
  double iElaps = cpuSecond() - iStart;
  printf("EAVX512c2r : nsamp : %d nloop : %d ompT : %d iElaps : %f \n", nsamp,
         nloop, ompT, iElaps * 1000);

  ::free(xf);
  ::free(xt);
}
TEST(TestDFT, AVX512c2c) {
  char *nsampVar;
  char *nloopVar;
  nsampVar = getenv("NSAMP");
  nloopVar = getenv("NLOOP");
  int nsamp = atoi(nsampVar);
  int nloop = atoi(nloopVar);
  int num = nsamp * 2;
  // for (int i = 0; i < Iter; i++) {
  __m512 *xt = nullptr;
  __m512 *xf = nullptr;
  ::posix_memalign((void **)&xt, 64, nloop * 2 * nsamp * sizeof(float));
  ::posix_memalign((void **)&xf, 64, nloop * 2 * nsamp * sizeof(float));

  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_real_distribution<float> dist(-10.0, 10.0);
  // Store values in vector
  std::vector<float> values(2 * nsamp * nloop);
  std::vector<float> out_array(2 * nsamp * nloop);
  for (int i = 0; i < values.size(); ++i) {
    values[i] = dist(generator);
  }

  int ompT = omp_get_max_threads();

  using namespace std::chrono;
  high_resolution_clock::time_point iStart = high_resolution_clock::now();

  __m512i vIdx = _mm512_set_epi32(
      num * 15, num * 14, num * 13, num * 12, num * 11, num * 10, num * 9,
      num * 8, num * 7, num * 6, num * 5, num * 4, num * 3, num * 2, num, 0);
  for (unsigned i = 0; i < nloop / nvec_512; i++)

    for (unsigned j = 0; j < num; j++) {
      float *ptr = &values[0] + j + i * num * nvec_512;
      xt[j + i * num] = _mm512_i32gather_ps(vIdx, static_cast<void *>(ptr), 4);
    }

  high_resolution_clock::time_point afterGather = high_resolution_clock::now();

  for (int i = 0; i < Iter; i++) {
    switch (nsamp) {
    case 32:
      m512::dft_codelet_c2cf_32(xt, xt + 1, xf, xf + 1, 2, 2, nloop / nvec_512,
                                num, num);
      break;
    case 64:
      m512::dft_codelet_c2cf_64(xt, xt + 1, xf, xf + 1, 2, 2, nloop / nvec_512,
                                num, num);
      break;
    case 128:
      m512::dft_codelet_c2cf_128(xt, xt + 1, xf, xf + 1, 2, 2, nloop / nvec_512,
                                 num, num);
      break;
    case 256:
      m512::dft_codelet_c2cf_256(xt, xt + 1, xf, xf + 1, 2, 2, nloop / nvec_512,
                                 num, num);
      break;
    case 512:
      m512::dft_codelet_c2cf_512(xt, xt + 1, xf, xf + 1, 2, 2, nloop / nvec_512,
                                 num, num);
      break;
    case 1024:
      m512::dft_codelet_c2cf_1024(xt, xt + 1, xf, xf + 1, 2, 2,
                                  nloop / nvec_512, num, num);
      break;
    }
  }
  high_resolution_clock::time_point afterCodelet = high_resolution_clock::now();
  for (unsigned i = 0; i < nloop / nvec_512; i++)
    for (unsigned j = 0; j < num; j++)
      for (unsigned k = 0; k < nvec_512; k++) {
        out_array[i * num * nvec_512 + k * num + j] = xf[j + i * num][k];
      }
  high_resolution_clock::time_point afterScatter = high_resolution_clock::now();

  duration<double, std::milli> gather = afterGather - iStart;
  duration<double, std::milli> codelet = afterCodelet - afterGather;
  duration<double, std::milli> scatter = afterScatter - afterCodelet;

  std::cout << "EAVX512c2c : nsamp : " << nsamp << " nloop : " << nloop
            << " ompT : " << ompT << " gather : " << gather.count()
            << " codelet : " << codelet.count()
            << " scatter : " << scatter.count() << " .\n ";

  ::free(xf);
  ::free(xt);
}
#undef DK
#endif

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
