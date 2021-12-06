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
  static const __m512 name = {(val), (val), (val), (val), (val), (val),        \
                              (val), (val), (val), (val), (val), (val),        \
                              (val), (val), (val), (val)}
