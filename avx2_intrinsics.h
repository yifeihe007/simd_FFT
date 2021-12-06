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