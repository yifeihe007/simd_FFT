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

#include <fftw3.h>
#include <cstring>
#include <gtest/gtest.h>
#include <immintrin.h>
#include <iostream>
#include <omp.h>
#include <random>
#include <stdlib.h>
#include <sys/time.h>

#include "utils.h"

double cpuSecond() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}


constexpr int nvec = 8;
constexpr int nvec_512 = 16;
constexpr int Iter = 1000;

#if 0
TEST(TestDFT, manyc2cFFTW_Aligned_One) {
  char *nsampVar;
  char *nloopVar;
  nsampVar = getenv("NSAMP");
  nloopVar = getenv("NLOOP");
  int nsamp = atoi(nsampVar);
  int nloop = atoi(nloopVar);
  int ompT = omp_get_max_threads();

  fftw_init_threads();
  fftw_plan_with_nthreads(ompT);

  float *xt = fftwf_alloc_real((nsamp * 2 + 2) * nloop);
  float *xf = fftwf_alloc_real((nsamp * 2 + 2) * nloop);

  fftwf_plan plan = fftwf_plan_many_dft(
      1, &nsamp, nloop, (fftwf_complex *)xt, nullptr, 1, (nsamp + 1),
      (fftwf_complex *)xf, nullptr, 1, nsamp + 1, FFTW_FORWARD, FFTW_MEASURE);

  double iStart = cpuSecond();

  for (int i = 0; i < Iter; i++)
    fftwf_execute(plan);
  double iElaps = cpuSecond() - iStart;
  printf("Ec2cFFTW : nsamp : %d nloop : %d ompT : %d iElaps : %f \n", nsamp,
         nloop, ompT, iElaps * 1000);

  fftwf_destroy_plan(plan);
  fftw_cleanup_threads();

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
  fftw_init_threads();
  fftw_plan_with_nthreads(ompT);

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
  fftw_cleanup_threads();
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
  fftw_init_threads();
  fftw_plan_with_nthreads(ompT);

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
  fftw_cleanup_threads();
  fftwf_free(xf);
  fftwf_free(xt);

  // omp_destroy_lock(&writelock);
}
#endif
#if defined(__AVX__)

using INT = int;
using stride = int;

inline __m256 ADD(const __m256 &a, const __m256 &b) {
  return _mm256_add_ps(a, b);
}
inline __m256 SUB(const __m256 &a, const __m256 &b) {
  return _mm256_sub_ps(a, b);
}
inline __m256 MUL(const __m256 &a, const __m256 &b) {
  return _mm256_mul_ps(a, b);
}

// inline __m256 NEG(const __m256& a) { return
// _mm256_sub_ps(_mm256_setzero_ps(),a); }
inline __m256 NEG(const __m256 &a) {
  return _mm256_xor_ps(a, _mm256_set1_ps(-0.0));
}

#if defined(__FMA__)
inline __m256 FMA(const __m256 &a, const __m256 &b, const __m256 &c) {
  return _mm256_fmadd_ps(a, b, c);
}
inline __m256 FMS(const __m256 &a, const __m256 &b, const __m256 &c) {
  return _mm256_fmsub_ps(a, b, c);
}
// Note: inconsistency between FFTW and Intel intrinsics definitions of FNMA/S
inline __m256 FNMA(const __m256 &a, const __m256 &b, const __m256 &c) {
  return _mm256_fnmsub_ps(a, b, c);
}
inline __m256 FNMS(const __m256 &a, const __m256 &b, const __m256 &c) {
  return _mm256_fnmadd_ps(a, b, c);
}
#else
inline __m256 FMA(const __m256 &a, const __m256 &b, const __m256 &c) {
  return _mm256_add_ps(_mm256_mul_ps(a, b), c);
}
inline __m256 FMS(const __m256 &a, const __m256 &b, const __m256 &c) {
  return _mm256_sub_ps(_mm256_mul_ps(a, b), c);
}
inline __m256 FNMA(const __m256 &a, const __m256 &b, const __m256 &c) {
  return _mm256_sub_ps(NEG(_mm256_mul_ps(a, b)), c);
}
inline __m256 FNMS(const __m256 &a, const __m256 &b, const __m256 &c) {
  return _mm256_add_ps(NEG(_mm256_mul_ps(a, b)), c);
}
#endif

inline std::pair<__m256, __m256> ADD(const std::pair<__m256, __m256> &a,
                                     const std::pair<__m256, __m256> &b) {
  return {ADD(a.first, b.first), ADD(a.second, b.second)};
}
inline std::pair<__m256, __m256> SUB(const std::pair<__m256, __m256> &a,
                                     const std::pair<__m256, __m256> &b) {
  return {SUB(a.first, b.first), SUB(a.second, b.second)};
}
inline std::pair<__m256, __m256> MUL(const std::pair<__m256, __m256> &a,
                                     const std::pair<__m256, __m256> &b) {
  return {MUL(a.first, b.first), MUL(a.second, b.second)};
}

inline std::pair<__m256, __m256> NEG(const std::pair<__m256, __m256> &a) {
  return {NEG(a.first), NEG(a.second)};
}

inline std::pair<__m256, __m256> FMA(const std::pair<__m256, __m256> &a,
                                     const std::pair<__m256, __m256> &b,
                                     const std::pair<__m256, __m256> &c) {
  return {FMA(a.first, b.first, c.first), FMA(a.second, b.second, c.second)};
}
inline std::pair<__m256, __m256> FMS(const std::pair<__m256, __m256> &a,
                                     const std::pair<__m256, __m256> &b,
                                     const std::pair<__m256, __m256> &c) {
  return {FMS(a.first, b.first, c.first), FMS(a.second, b.second, c.second)};
}
// Note: inconsistency between FFTW and Intel intrinsics definitions of FNMA/S
inline std::pair<__m256, __m256> FNMA(const std::pair<__m256, __m256> &a,
                                      const std::pair<__m256, __m256> &b,
                                      const std::pair<__m256, __m256> &c) {
  return {FNMA(a.first, b.first, c.first), FNMA(a.second, b.second, c.second)};
}
inline std::pair<__m256, __m256> FNMS(const std::pair<__m256, __m256> &a,
                                      const std::pair<__m256, __m256> &b,
                                      const std::pair<__m256, __m256> &c) {
  return {FNMS(a.first, b.first, c.first), FNMS(a.second, b.second, c.second)};
}

inline std::pair<__m256, __m256> MUL(const __m256 &a,
                                     const std::pair<__m256, __m256> &b) {
  return {MUL(a, b.first), MUL(a, b.second)};
}
inline std::pair<__m256, __m256> FMA(const __m256 &a,
                                     const std::pair<__m256, __m256> &b,
                                     const std::pair<__m256, __m256> &c) {
  return {FMA(a, b.first, c.first), FMA(a, b.second, c.second)};
}
inline std::pair<__m256, __m256> FMS(const __m256 &a,
                                     const std::pair<__m256, __m256> &b,
                                     const std::pair<__m256, __m256> &c) {
  return {FMS(a, b.first, c.first), FMS(a, b.second, c.second)};
}
inline std::pair<__m256, __m256> FNMA(const __m256 &a,
                                      const std::pair<__m256, __m256> &b,
                                      const std::pair<__m256, __m256> &c) {
  return {FNMA(a, b.first, c.first), FNMA(a, b.second, c.second)};
}
inline std::pair<__m256, __m256> FNMS(const __m256 &a,
                                      const std::pair<__m256, __m256> &b,
                                      const std::pair<__m256, __m256> &c) {
  return {FNMS(a, b.first, c.first), FNMS(a, b.second, c.second)};
}

#define DK(name, val)                                                          \
  static const __m256 name = {(val), (val), (val), (val),                      \
                              (val), (val), (val), (val)}

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
  printf("EAVX2r2c : nsamp : %d nloop : %d ompT : %d gather : %f codelet : %f "
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
  float *in_data = fftwf_alloc_real(num * nloop);
  float *out_data = fftwf_alloc_real(num * nloop);
  ::posix_memalign((void **)&xt, 32, nloop * num * sizeof(float));
  ::posix_memalign((void **)&xf, 32, nloop * num * sizeof(float));

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
      m256::dft_codelet_c2cf_32(xt, xt + 1, xf, xf + 1, 2, 2, nloop / nvec,
                               num, num);
      break;
    case 64:
      m256::dft_codelet_c2cf_64(xt, xt + 1, xf, xf + 1, 2, 2, nloop / nvec,
                                num, num);
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
  double afterCodelet = cpuSecond();

  for (unsigned i = 0; i < nsamp / nvec; i++)
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
#endif


using INT = int;
using stride = int;

inline __m512 ADD(const __m512 &a, const __m512 &b) {
  return _mm512_add_ps(a, b);
}
inline __m512 SUB(const __m512 &a, const __m512 &b) {
  return _mm512_sub_ps(a, b);
}
inline __m512 MUL(const __m512 &a, const __m512 &b) {
  return _mm512_mul_ps(a, b);
}

// inline __m512 NEG(const __m512 & a) { return
//_mm512_sub_ps(_mm256_setzero_ps(), a);
//}
inline __m512 NEG(const __m512 &a) {
  return _mm512_xor_ps(a, _mm512_set1_ps(-0.0));
}

#if defined(__FMA__)
inline __m512 FMA(const __m512 &a, const __m512 &b, const __m512 &c) {
  return _mm512_fmadd_ps(a, b, c);
}
inline __m512 FMS(const __m512 &a, const __m512 &b, const __m512 &c) {
  return _mm512_fmsub_ps(a, b, c);
}
// Note: inconsistency between FFTW and Intel intrinsics definitions of FNMA/S
inline __m512 FNMA(const __m512 &a, const __m512 &b, const __m512 &c) {
  return _mm512_fnmsub_ps(a, b, c);
}
inline __m512 FNMS(const __m512 &a, const __m512 &b, const __m512 &c) {
  return _mm512_fnmadd_ps(a, b, c);
}
#else
inline __m512 FMA(const __m512 &a, const __m512 &b, const __m512 &c) {
  return _mm512_add_ps(_mm512_mul_ps(a, b), c);
}
inline __m512 FMS(const __m512 &a, const __m512 &b, const __m512 &c) {
  return _mm512_sub_ps(_mm512_mul_ps(a, b), c);
}
inline __m512 FNMA(const __m512 &a, const __m512 &b, const __m512 &c) {
  return _mm512_sub_ps(NEG(_mm512_mul_ps(a, b)), c);
}
inline __m512 FNMS(const __m512 &a, const __m512 &b, const __m512 &c) {
  return _mm512_add_ps(NEG(_mm512_mul_ps(a, b)), c);
}
#endif

inline std::pair<__m512, __m512> ADD(const std::pair<__m512, __m512> &a,
                                     const std::pair<__m512, __m512> &b) {
  return {ADD(a.first, b.first), ADD(a.second, b.second)};
}
inline std::pair<__m512, __m512> SUB(const std::pair<__m512, __m512> &a,
                                     const std::pair<__m512, __m512> &b) {
  return {SUB(a.first, b.first), SUB(a.second, b.second)};
}
inline std::pair<__m512, __m512> MUL(const std::pair<__m512, __m512> &a,
                                     const std::pair<__m512, __m512> &b) {
  return {MUL(a.first, b.first), MUL(a.second, b.second)};
}

inline std::pair<__m512, __m512> NEG(const std::pair<__m512, __m512> &a) {
  return {NEG(a.first), NEG(a.second)};
}

inline std::pair<__m512, __m512> FMA(const std::pair<__m512, __m512> &a,
                                     const std::pair<__m512, __m512> &b,
                                     const std::pair<__m512, __m512> &c) {
  return {FMA(a.first, b.first, c.first), FMA(a.second, b.second, c.second)};
}
inline std::pair<__m512, __m512> FMS(const std::pair<__m512, __m512> &a,
                                     const std::pair<__m512, __m512> &b,
                                     const std::pair<__m512, __m512> &c) {
  return {FMS(a.first, b.first, c.first), FMS(a.second, b.second, c.second)};
}
// Note: inconsistency between FFTW and Intel intrinsics definitions of FNMA/S
inline std::pair<__m512, __m512> FNMA(const std::pair<__m512, __m512> &a,
                                      const std::pair<__m512, __m512> &b,
                                      const std::pair<__m512, __m512> &c) {
  return {FNMA(a.first, b.first, c.first), FNMA(a.second, b.second, c.second)};
}
inline std::pair<__m512, __m512> FNMS(const std::pair<__m512, __m512> &a,
                                      const std::pair<__m512, __m512> &b,
                                      const std::pair<__m512, __m512> &c) {
  return {FNMS(a.first, b.first, c.first), FNMS(a.second, b.second, c.second)};
}

inline std::pair<__m512, __m512> MUL(const __m512 &a,
                                     const std::pair<__m512, __m512> &b) {
  return {MUL(a, b.first), MUL(a, b.second)};
}
inline std::pair<__m512, __m512> FMA(const __m512 &a,
                                     const std::pair<__m512, __m512> &b,
                                     const std::pair<__m512, __m512> &c) {
  return {FMA(a, b.first, c.first), FMA(a, b.second, c.second)};
}
inline std::pair<__m512, __m512> FMS(const __m512 &a,
                                     const std::pair<__m512, __m512> &b,
                                     const std::pair<__m512, __m512> &c) {
  return {FMS(a, b.first, c.first), FMS(a, b.second, c.second)};
}
inline std::pair<__m512, __m512> FNMA(const __m512 &a,
                                      const std::pair<__m512, __m512> &b,
                                      const std::pair<__m512, __m512> &c) {
  return {FNMA(a, b.first, c.first), FNMA(a, b.second, c.second)};
}
inline std::pair<__m512, __m512> FNMS(const __m512 &a,
                                      const std::pair<__m512, __m512> &b,
                                      const std::pair<__m512, __m512> &c) {
  return {FNMS(a, b.first, c.first), FNMS(a, b.second, c.second)};
}

#define DK(name, val)                                                          \
  static const __m512 name = {(val), (val), (val), (val),                      \
                              (val), (val), (val), (val)}

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
  float *in_data = fftwf_alloc_real(num * nloop);
  float *out_data = fftwf_alloc_real(num * nloop);

  int ompT = omp_get_max_threads();

  double iStart = cpuSecond();

  __m512i vIdx = _mm512_set_epi32(
      num * 15, num * 14, num * 13, num * 12, num * 11, num * 10, num * 9,
      num * 8, num * 7, num * 6, num * 5, num * 4, num * 3, num * 2, num, 0);
  for (unsigned i = 0; i < nloop / nvec_512; i++)

    for (unsigned j = 0; j < num; j++) {

      xt[j + i * num] = _mm512_i32gather_ps(
          vIdx, static_cast<void*>(in_data) + j + i * num * nvec_512, 4);
    }

  double afterGather = cpuSecond();

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
      m512::dft_codelet_c2cf_1024(xt, xt + 1, xf, xf + 1, 2, 2, nloop / nvec_512,
                                  num, num);
      break;
    }
  }
 double afterCodelet = cpuSecond();

  for (unsigned i = 0; i < nsamp / nvec_512; i++)
    for (unsigned j = 0; j < num; j++)
      for (unsigned k = 0; k < nvec_512; k++) {
        static_cast<float *>(out_data)[i * num * nvec_512 + k * num + j] =
            xf[j + i * num][k];
      }
}

TEST(TestDFT, correctness_ones) {
  //c2c
  constexpr int fft_size = 32;
  constexpr int batch_size = 32;
  __m512* xt = nullptr;
  __m512* xf = nullptr;
  constexpr size_t byte_size = 2 * fft_size * batch_size * sizeof(float);

  ::posix_memalign((void**)(&xt), 64, byte_size);
  ::posix_memalign((void**)(&xf), 64, byte_size);

  //Store values in vector
  std::vector<float> values(2 * fft_size * batch_size);
  for (int i = 0; i < values.size(); ++i)
    values[i] = 1.0f;

/*
  for (int i = 0; i < (byte_size / sizeof(__m512)); ++i) {
    xt[i] = _mm512_loadu_ps((const void*) &values[i * 16]);
  }
*/

  avx512_gather(fft_size, batch_size, xt, &values[0]);

  m512::dft_codelet_c2cf_32(xt, xt + 1, xf, xf + 1, 2, 2, batch_size / 16, (2 * fft_size), (2 * fft_size));

  std::vector<float> out_array(2 * fft_size * batch_size);
/*
  for (int i = 0; i < (byte_size / sizeof(__m512)); ++i) {
    _mm512_storeu_ps((void*) &out_array[i * 16], xf[i]);
  }
*/
  avx512_scatter(fft_size, batch_size, xf, &out_array[0]);
  for (int i = 0; i < out_array.size(); i += 2) {
    std::cerr << "Real: " << out_array[i] << ", complex: " << out_array[i + 1] << "\n";
  }
}

TEST(TestDFT, correctness_fftw) {
  //c2c
  const int fft_size = 32;
  const int batch_size = 32;
  __m512* xt = nullptr;
  __m512* xf = nullptr;
  constexpr size_t byte_size = 2 * fft_size * batch_size * sizeof(float);

  ::posix_memalign((void**)(&xt), 64, byte_size);
  ::posix_memalign((void**)(&xf), 64, byte_size);

  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_real_distribution<float> dist(-10.0, 10.0);
  //Store values in vector
  std::vector<float> values(2 * fft_size * batch_size);
  for (int i = 0; i < values.size(); ++i) {
    values[i] = dist(generator);
  }
  fftwf_complex* xt_fftw = fftwf_alloc_complex(fft_size * batch_size);
  fftwf_complex* xf_fftw = fftwf_alloc_complex(fft_size * batch_size);

  fftwf_plan plan = fftwf_plan_many_dft(
      1, &fft_size, batch_size, xt_fftw, &fft_size, 1, fft_size,
      xf_fftw, &fft_size, 1, fft_size, FFTW_FORWARD, FFTW_MEASURE);

  for (int i = 0; i < values.size(); i += 2) {
    xt_fftw[i / 2][0] = values[i];
    xt_fftw[i / 2][1] = values[i + 1];
  }

  fftwf_execute(plan);

  avx512_gather(fft_size, batch_size, xt, &values[0]);
  m512::dft_codelet_c2cf_32(xt, xt + 1, xf, xf + 1, 2, 2, batch_size / 16, (2 * fft_size), (2 * fft_size));

  std::vector<float> out_array(2 * fft_size * batch_size);
  avx512_scatter(fft_size, batch_size, xf, &out_array[0]);

  for (int i = 0; i < 2 * fft_size * batch_size; i += 2) {
    std::cerr << "ours[" << i / 2 << "]\t Real: " << out_array[i] << ", complex: " << out_array[i + 1] << "\n";
    std::cerr << "FFTW[" << i / 2 << "]\t Real: " << xf_fftw[i/2][0] << ", complex: " << xf_fftw[i/2][1] << "\n\n";
  }
}
/*

#include "dft_r2cb_60.c"
#include "dft_r2cf_60.c"

} // namespace m256

TEST(TestDFT, AVX2_512)
{
  __m512 *xt = nullptr;
  __m512 *xf;

  ::posix_memalign((void **)&xt, 64, nvec_512 * nsamp * sizeof(float));
  ::posix_memalign((void **)&xf, 64, nvec_512 * 2 * (nsamp / 2 + 1) *
sizeof(float)); for (unsigned i = 0; i < 2 * (nsamp / 2 + 1); i++) xf[i] =
_mm512_setzero_ps(); std::mt19937 core(12345);
  std::uniform_real_distribution<float> gen(0.0, 1.0);
  for (int ivec = 0; ivec < nvec_512; ivec++)
  {
    for (int isamp = 0; isamp < nsamp; isamp++)
    {
      xt[isamp][ivec] = gen(core);
    }
  }
  for (int iloop = 0; iloop < nloop; iloop++)
  {
    m512::dft_codelet_r2cf_60(xt, xt + 1, xf, xf + 1, 2, 2, 2, 1, 0, 0);
    if (iloop == 0)
    {
      for (int ifreq = 0; ifreq < 2 * (nsamp / 2 + 1); ifreq++)
      {
        std::cout << xf[ifreq][0] << ' ';
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

namespace m512_Unroll2_FixedStride
{

  using E = std::pair<__m512, __m512>;
  using R = std::pair<__m512, __m512>;

  inline int WS(const stride s, const stride i) { return 2 * i; }

#include "dft_r2cb_60.c"
#include "dft_r2cf_60.c"

} // namespace m256

TEST(TestDFT, AVX512_Unroll2_FixedStride)
{
  std::pair<__m512, __m512> *xt = nullptr;
  std::pair<__m512, __m512> *xf;
  ::posix_memalign((void **)&xt, 64, 2 * nvec_512 * nsamp * sizeof(float));
  ::posix_memalign((void **)&xf, 64, 2 * nvec_512 * 2 * (nsamp / 2 + 1) *
sizeof(float)); for (unsigned i = 0; i < 2 * (nsamp / 2 + 1); i++) xf[i].first =
xf[i].second = _mm512_setzero_ps(); std::mt19937 core(12345);
  std::uniform_real_distribution<float> gen(0.0, 1.0);
  for (int ivec = 0; ivec < nvec_512; ivec++)
  {
    for (int isamp = 0; isamp < nsamp; isamp++)
    {
      xt[isamp].first[ivec] = gen(core);
    }
    for (int isamp = 0; isamp < nsamp; isamp++)
    {
      xt[isamp].second[ivec] = gen(core);
    }
  }
  for (int iloop = 0; iloop < nloop / 2; iloop++)
  {
    m512_Unroll2_FixedStride::dft_codelet_r2cf_60(xt, xt + 1, xf, xf + 1, 0, 0,
0, 1, 0, 0); if (iloop == 0)
    {
      for (int ifreq = 0; ifreq < 2 * (nsamp / 2 + 1); ifreq++)
      {
        std::cout << xf[ifreq].first[0] << ' ';
      }
      std::cout << '\n';
    }
  }
  ::free(xf);
  ::free(xt);
}*/



int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
