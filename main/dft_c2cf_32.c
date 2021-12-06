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
inline void MAKE_VOLATILE_STRIDE(int a, int b) {}


using E = __m512;
using R = __m512;

inline int WS(const stride s, const stride i) { return s * i; }


/* Generated by: ./gen_notw.native -n 32 -standalone -fma -generic-arith -compact -name dft_codelet_c2cf_32 */

/*
 * This function contains 372 FP additions, 136 FP multiplications,
 * (or, 236 additions, 0 multiplications, 136 fused multiply/add),
 * 100 stack variables, 7 constants, and 128 memory accesses
 */
void dft_codelet_c2cf_32(const R * ri, const R * ii, R * ro, R * io, stride is, stride os, INT v, INT ivs, INT ovs)
{
DK(KP980785280, +0.980785280403230449126182236134239036973933731);
DK(KP198912367, +0.198912367379658006911597622644676228597850501);
DK(KP831469612, +0.831469612302545237078788377617905756738560812);
DK(KP668178637, +0.668178637919298919997757686523080761552472251);
DK(KP923879532, +0.923879532511286756128183189396788286822416626);
DK(KP707106781, +0.707106781186547524400844362104849039284835938);
DK(KP414213562, +0.414213562373095048801688724209698078569671875);
{
INT i;
#pragma omp parallel for
for (i = v; i > 0; i = i - 1){
E T7, T4r, T4Z, T18, T1z, T3t, T3T, T2T, Te, T1f, T50, T4s, T2W, T3u, T1G;
E T3U, Tm, T1n, T1O, T2Z, T3y, T3X, T4w, T53, Tt, T1u, T1V, T2Y, T3B, T3W;
E T4z, T52, T2t, T3L, T3O, T2K, TR, TY, T5F, T5G, T5H, T5I, T4R, T5k, T2E;
E T3M, T4W, T5j, T2N, T3P, T22, T3E, T3H, T2j, TC, TJ, T5A, T5B, T5C, T5D;
E T4G, T5h, T2d, T3F, T4L, T5g, T2m, T3I;
{
E T3, T1x, T14, T2R, T6, T2S, T17, T1y;
{
E T1, T2, T12, T13;
T1 = ri[0];
T2 = ri[WS(is, 16)];
T3 = ADD(T1, T2);
T1x = SUB(T1, T2);
T12 = ii[0];
T13 = ii[WS(is, 16)];
T14 = ADD(T12, T13);
T2R = SUB(T12, T13);
}
{
E T4, T5, T15, T16;
T4 = ri[WS(is, 8)];
T5 = ri[WS(is, 24)];
T6 = ADD(T4, T5);
T2S = SUB(T4, T5);
T15 = ii[WS(is, 8)];
T16 = ii[WS(is, 24)];
T17 = ADD(T15, T16);
T1y = SUB(T15, T16);
}
T7 = ADD(T3, T6);
T4r = SUB(T3, T6);
T4Z = SUB(T14, T17);
T18 = ADD(T14, T17);
T1z = ADD(T1x, T1y);
T3t = SUB(T1x, T1y);
T3T = ADD(T2S, T2R);
T2T = SUB(T2R, T2S);
}
{
E Ta, T1A, T1b, T1B, Td, T1D, T1e, T1E;
{
E T8, T9, T19, T1a;
T8 = ri[WS(is, 4)];
T9 = ri[WS(is, 20)];
Ta = ADD(T8, T9);
T1A = SUB(T8, T9);
T19 = ii[WS(is, 4)];
T1a = ii[WS(is, 20)];
T1b = ADD(T19, T1a);
T1B = SUB(T19, T1a);
}
{
E Tb, Tc, T1c, T1d;
Tb = ri[WS(is, 28)];
Tc = ri[WS(is, 12)];
Td = ADD(Tb, Tc);
T1D = SUB(Tb, Tc);
T1c = ii[WS(is, 28)];
T1d = ii[WS(is, 12)];
T1e = ADD(T1c, T1d);
T1E = SUB(T1c, T1d);
}
Te = ADD(Ta, Td);
T1f = ADD(T1b, T1e);
T50 = SUB(Td, Ta);
T4s = SUB(T1b, T1e);
{
E T2U, T2V, T1C, T1F;
T2U = SUB(T1B, T1A);
T2V = ADD(T1D, T1E);
T2W = ADD(T2U, T2V);
T3u = SUB(T2U, T2V);
T1C = ADD(T1A, T1B);
T1F = SUB(T1D, T1E);
T1G = ADD(T1C, T1F);
T3U = SUB(T1F, T1C);
}
}
{
E Ti, T1L, T1j, T1I, Tl, T1J, T1m, T1M, T1K, T1N;
{
E Tg, Th, T1h, T1i;
Tg = ri[WS(is, 2)];
Th = ri[WS(is, 18)];
Ti = ADD(Tg, Th);
T1L = SUB(Tg, Th);
T1h = ii[WS(is, 2)];
T1i = ii[WS(is, 18)];
T1j = ADD(T1h, T1i);
T1I = SUB(T1h, T1i);
}
{
E Tj, Tk, T1k, T1l;
Tj = ri[WS(is, 10)];
Tk = ri[WS(is, 26)];
Tl = ADD(Tj, Tk);
T1J = SUB(Tj, Tk);
T1k = ii[WS(is, 10)];
T1l = ii[WS(is, 26)];
T1m = ADD(T1k, T1l);
T1M = SUB(T1k, T1l);
}
Tm = ADD(Ti, Tl);
T1n = ADD(T1j, T1m);
T1K = SUB(T1I, T1J);
T1N = ADD(T1L, T1M);
T1O = FNMS(KP414213562, T1N, T1K);
T2Z = FMA(KP414213562, T1K, T1N);
{
E T3w, T3x, T4u, T4v;
T3w = ADD(T1J, T1I);
T3x = SUB(T1L, T1M);
T3y = FMA(KP414213562, T3x, T3w);
T3X = FNMS(KP414213562, T3w, T3x);
T4u = SUB(T1j, T1m);
T4v = SUB(Ti, Tl);
T4w = SUB(T4u, T4v);
T53 = ADD(T4v, T4u);
}
}
{
E Tp, T1S, T1q, T1P, Ts, T1Q, T1t, T1T, T1R, T1U;
{
E Tn, To, T1o, T1p;
Tn = ri[WS(is, 30)];
To = ri[WS(is, 14)];
Tp = ADD(Tn, To);
T1S = SUB(Tn, To);
T1o = ii[WS(is, 30)];
T1p = ii[WS(is, 14)];
T1q = ADD(T1o, T1p);
T1P = SUB(T1o, T1p);
}
{
E Tq, Tr, T1r, T1s;
Tq = ri[WS(is, 6)];
Tr = ri[WS(is, 22)];
Ts = ADD(Tq, Tr);
T1Q = SUB(Tq, Tr);
T1r = ii[WS(is, 6)];
T1s = ii[WS(is, 22)];
T1t = ADD(T1r, T1s);
T1T = SUB(T1r, T1s);
}
Tt = ADD(Tp, Ts);
T1u = ADD(T1q, T1t);
T1R = SUB(T1P, T1Q);
T1U = ADD(T1S, T1T);
T1V = FMA(KP414213562, T1U, T1R);
T2Y = FNMS(KP414213562, T1R, T1U);
{
E T3z, T3A, T4x, T4y;
T3z = ADD(T1Q, T1P);
T3A = SUB(T1S, T1T);
T3B = FNMS(KP414213562, T3A, T3z);
T3W = FMA(KP414213562, T3z, T3A);
T4x = SUB(Tp, Ts);
T4y = SUB(T1q, T1t);
T4z = ADD(T4x, T4y);
T52 = SUB(T4x, T4y);
}
}
{
E TN, T2G, T2r, T4N, TQ, T2s, T2J, T4O, TU, T2x, T2w, T4T, TX, T2z, T2C;
E T4U;
{
E TL, TM, T2p, T2q;
TL = ri[WS(is, 31)];
TM = ri[WS(is, 15)];
TN = ADD(TL, TM);
T2G = SUB(TL, TM);
T2p = ii[WS(is, 31)];
T2q = ii[WS(is, 15)];
T2r = SUB(T2p, T2q);
T4N = ADD(T2p, T2q);
}
{
E TO, TP, T2H, T2I;
TO = ri[WS(is, 7)];
TP = ri[WS(is, 23)];
TQ = ADD(TO, TP);
T2s = SUB(TO, TP);
T2H = ii[WS(is, 7)];
T2I = ii[WS(is, 23)];
T2J = SUB(T2H, T2I);
T4O = ADD(T2H, T2I);
}
{
E TS, TT, T2u, T2v;
TS = ri[WS(is, 3)];
TT = ri[WS(is, 19)];
TU = ADD(TS, TT);
T2x = SUB(TS, TT);
T2u = ii[WS(is, 3)];
T2v = ii[WS(is, 19)];
T2w = SUB(T2u, T2v);
T4T = ADD(T2u, T2v);
}
{
E TV, TW, T2A, T2B;
TV = ri[WS(is, 27)];
TW = ri[WS(is, 11)];
TX = ADD(TV, TW);
T2z = SUB(TV, TW);
T2A = ii[WS(is, 27)];
T2B = ii[WS(is, 11)];
T2C = SUB(T2A, T2B);
T4U = ADD(T2A, T2B);
}
T2t = SUB(T2r, T2s);
T3L = SUB(T2G, T2J);
T3O = ADD(T2s, T2r);
T2K = ADD(T2G, T2J);
TR = ADD(TN, TQ);
TY = ADD(TU, TX);
T5F = SUB(TR, TY);
{
E T4P, T4Q, T2y, T2D;
T5G = ADD(T4N, T4O);
T5H = ADD(T4T, T4U);
T5I = SUB(T5G, T5H);
T4P = SUB(T4N, T4O);
T4Q = SUB(TX, TU);
T4R = SUB(T4P, T4Q);
T5k = ADD(T4Q, T4P);
T2y = SUB(T2w, T2x);
T2D = ADD(T2z, T2C);
T2E = ADD(T2y, T2D);
T3M = SUB(T2D, T2y);
{
E T4S, T4V, T2L, T2M;
T4S = SUB(TN, TQ);
T4V = SUB(T4T, T4U);
T4W = SUB(T4S, T4V);
T5j = ADD(T4S, T4V);
T2L = ADD(T2x, T2w);
T2M = SUB(T2z, T2C);
T2N = ADD(T2L, T2M);
T3P = SUB(T2L, T2M);
}
}
}
{
E Ty, T2f, T20, T4C, TB, T21, T2i, T4D, TF, T26, T25, T4I, TI, T28, T2b;
E T4J;
{
E Tw, Tx, T1Y, T1Z;
Tw = ri[WS(is, 1)];
Tx = ri[WS(is, 17)];
Ty = ADD(Tw, Tx);
T2f = SUB(Tw, Tx);
T1Y = ii[WS(is, 1)];
T1Z = ii[WS(is, 17)];
T20 = SUB(T1Y, T1Z);
T4C = ADD(T1Y, T1Z);
}
{
E Tz, TA, T2g, T2h;
Tz = ri[WS(is, 9)];
TA = ri[WS(is, 25)];
TB = ADD(Tz, TA);
T21 = SUB(Tz, TA);
T2g = ii[WS(is, 9)];
T2h = ii[WS(is, 25)];
T2i = SUB(T2g, T2h);
T4D = ADD(T2g, T2h);
}
{
E TD, TE, T23, T24;
TD = ri[WS(is, 5)];
TE = ri[WS(is, 21)];
TF = ADD(TD, TE);
T26 = SUB(TD, TE);
T23 = ii[WS(is, 5)];
T24 = ii[WS(is, 21)];
T25 = SUB(T23, T24);
T4I = ADD(T23, T24);
}
{
E TG, TH, T29, T2a;
TG = ri[WS(is, 29)];
TH = ri[WS(is, 13)];
TI = ADD(TG, TH);
T28 = SUB(TG, TH);
T29 = ii[WS(is, 29)];
T2a = ii[WS(is, 13)];
T2b = SUB(T29, T2a);
T4J = ADD(T29, T2a);
}
T22 = SUB(T20, T21);
T3E = SUB(T2f, T2i);
T3H = ADD(T21, T20);
T2j = ADD(T2f, T2i);
TC = ADD(Ty, TB);
TJ = ADD(TF, TI);
T5A = SUB(TC, TJ);
{
E T4E, T4F, T27, T2c;
T5B = ADD(T4C, T4D);
T5C = ADD(T4I, T4J);
T5D = SUB(T5B, T5C);
T4E = SUB(T4C, T4D);
T4F = SUB(TI, TF);
T4G = SUB(T4E, T4F);
T5h = ADD(T4F, T4E);
T27 = SUB(T25, T26);
T2c = ADD(T28, T2b);
T2d = ADD(T27, T2c);
T3F = SUB(T2c, T27);
{
E T4H, T4K, T2k, T2l;
T4H = SUB(Ty, TB);
T4K = SUB(T4I, T4J);
T4L = SUB(T4H, T4K);
T5g = ADD(T4H, T4K);
T2k = ADD(T26, T25);
T2l = SUB(T28, T2b);
T2m = ADD(T2k, T2l);
T3I = SUB(T2k, T2l);
}
}
}
{
E T4B, T5b, T5a, T5c, T4Y, T56, T55, T57;
{
E T4t, T4A, T58, T59;
T4t = SUB(T4r, T4s);
T4A = SUB(T4w, T4z);
T4B = FMA(KP707106781, T4A, T4t);
T5b = FNMS(KP707106781, T4A, T4t);
T58 = FMA(KP414213562, T4R, T4W);
T59 = FNMS(KP414213562, T4G, T4L);
T5a = SUB(T58, T59);
T5c = ADD(T59, T58);
}
{
E T4M, T4X, T51, T54;
T4M = FMA(KP414213562, T4L, T4G);
T4X = FNMS(KP414213562, T4W, T4R);
T4Y = SUB(T4M, T4X);
T56 = ADD(T4M, T4X);
T51 = SUB(T4Z, T50);
T54 = SUB(T52, T53);
T55 = FNMS(KP707106781, T54, T51);
T57 = FMA(KP707106781, T54, T51);
}
ro[WS(os, 22)] = FNMS(KP923879532, T4Y, T4B);
io[WS(os, 22)] = FNMS(KP923879532, T5a, T57);
ro[WS(os, 6)] = FMA(KP923879532, T4Y, T4B);
io[WS(os, 6)] = FMA(KP923879532, T5a, T57);
io[WS(os, 14)] = FNMS(KP923879532, T56, T55);
ro[WS(os, 14)] = FNMS(KP923879532, T5c, T5b);
io[WS(os, 30)] = FMA(KP923879532, T56, T55);
ro[WS(os, 30)] = FMA(KP923879532, T5c, T5b);
}
{
E T5f, T5r, T5u, T5w, T5m, T5q, T5p, T5v;
{
E T5d, T5e, T5s, T5t;
T5d = ADD(T4r, T4s);
T5e = ADD(T53, T52);
T5f = FMA(KP707106781, T5e, T5d);
T5r = FNMS(KP707106781, T5e, T5d);
T5s = FNMS(KP414213562, T5g, T5h);
T5t = FMA(KP414213562, T5j, T5k);
T5u = SUB(T5s, T5t);
T5w = ADD(T5s, T5t);
}
{
E T5i, T5l, T5n, T5o;
T5i = FMA(KP414213562, T5h, T5g);
T5l = FNMS(KP414213562, T5k, T5j);
T5m = ADD(T5i, T5l);
T5q = SUB(T5l, T5i);
T5n = ADD(T50, T4Z);
T5o = ADD(T4w, T4z);
T5p = FNMS(KP707106781, T5o, T5n);
T5v = FMA(KP707106781, T5o, T5n);
}
ro[WS(os, 18)] = FNMS(KP923879532, T5m, T5f);
io[WS(os, 18)] = FNMS(KP923879532, T5w, T5v);
ro[WS(os, 2)] = FMA(KP923879532, T5m, T5f);
io[WS(os, 2)] = FMA(KP923879532, T5w, T5v);
io[WS(os, 26)] = FNMS(KP923879532, T5q, T5p);
ro[WS(os, 26)] = FNMS(KP923879532, T5u, T5r);
io[WS(os, 10)] = FMA(KP923879532, T5q, T5p);
ro[WS(os, 10)] = FMA(KP923879532, T5u, T5r);
}
{
E T5z, T5P, T5S, T5U, T5K, T5O, T5N, T5T;
{
E T5x, T5y, T5Q, T5R;
T5x = SUB(T7, Te);
T5y = SUB(T1n, T1u);
T5z = ADD(T5x, T5y);
T5P = SUB(T5x, T5y);
T5Q = SUB(T5D, T5A);
T5R = ADD(T5F, T5I);
T5S = SUB(T5Q, T5R);
T5U = ADD(T5Q, T5R);
}
{
E T5E, T5J, T5L, T5M;
T5E = ADD(T5A, T5D);
T5J = SUB(T5F, T5I);
T5K = ADD(T5E, T5J);
T5O = SUB(T5J, T5E);
T5L = SUB(T18, T1f);
T5M = SUB(Tt, Tm);
T5N = SUB(T5L, T5M);
T5T = ADD(T5M, T5L);
}
ro[WS(os, 20)] = FNMS(KP707106781, T5K, T5z);
io[WS(os, 20)] = FNMS(KP707106781, T5U, T5T);
ro[WS(os, 4)] = FMA(KP707106781, T5K, T5z);
io[WS(os, 4)] = FMA(KP707106781, T5U, T5T);
io[WS(os, 28)] = FNMS(KP707106781, T5O, T5N);
ro[WS(os, 28)] = FNMS(KP707106781, T5S, T5P);
io[WS(os, 12)] = FMA(KP707106781, T5O, T5N);
ro[WS(os, 12)] = FMA(KP707106781, T5S, T5P);
}
{
E Tv, T5V, T5Y, T60, T10, T11, T1w, T5Z;
{
E Tf, Tu, T5W, T5X;
Tf = ADD(T7, Te);
Tu = ADD(Tm, Tt);
Tv = ADD(Tf, Tu);
T5V = SUB(Tf, Tu);
T5W = ADD(T5B, T5C);
T5X = ADD(T5G, T5H);
T5Y = SUB(T5W, T5X);
T60 = ADD(T5W, T5X);
}
{
E TK, TZ, T1g, T1v;
TK = ADD(TC, TJ);
TZ = ADD(TR, TY);
T10 = ADD(TK, TZ);
T11 = SUB(TZ, TK);
T1g = ADD(T18, T1f);
T1v = ADD(T1n, T1u);
T1w = SUB(T1g, T1v);
T5Z = ADD(T1g, T1v);
}
ro[WS(os, 16)] = SUB(Tv, T10);
io[WS(os, 16)] = SUB(T5Z, T60);
ro[0] = ADD(Tv, T10);
io[0] = ADD(T5Z, T60);
io[WS(os, 8)] = ADD(T11, T1w);
ro[WS(os, 8)] = ADD(T5V, T5Y);
io[WS(os, 24)] = SUB(T1w, T11);
ro[WS(os, 24)] = SUB(T5V, T5Y);
}
{
E T1X, T37, T31, T33, T2o, T35, T2P, T34;
{
E T1H, T1W, T2X, T30;
T1H = FNMS(KP707106781, T1G, T1z);
T1W = SUB(T1O, T1V);
T1X = FMA(KP923879532, T1W, T1H);
T37 = FNMS(KP923879532, T1W, T1H);
T2X = FNMS(KP707106781, T2W, T2T);
T30 = SUB(T2Y, T2Z);
T31 = FNMS(KP923879532, T30, T2X);
T33 = FMA(KP923879532, T30, T2X);
}
{
E T2e, T2n, T2F, T2O;
T2e = FNMS(KP707106781, T2d, T22);
T2n = FNMS(KP707106781, T2m, T2j);
T2o = FMA(KP668178637, T2n, T2e);
T35 = FNMS(KP668178637, T2e, T2n);
T2F = FNMS(KP707106781, T2E, T2t);
T2O = FNMS(KP707106781, T2N, T2K);
T2P = FNMS(KP668178637, T2O, T2F);
T34 = FMA(KP668178637, T2F, T2O);
}
{
E T2Q, T36, T32, T38;
T2Q = SUB(T2o, T2P);
ro[WS(os, 21)] = FNMS(KP831469612, T2Q, T1X);
ro[WS(os, 5)] = FMA(KP831469612, T2Q, T1X);
T36 = SUB(T34, T35);
io[WS(os, 21)] = FNMS(KP831469612, T36, T33);
io[WS(os, 5)] = FMA(KP831469612, T36, T33);
T32 = ADD(T2o, T2P);
io[WS(os, 13)] = FNMS(KP831469612, T32, T31);
io[WS(os, 29)] = FMA(KP831469612, T32, T31);
T38 = ADD(T35, T34);
ro[WS(os, 13)] = FNMS(KP831469612, T38, T37);
ro[WS(os, 29)] = FMA(KP831469612, T38, T37);
}
}
{
E T3D, T41, T3Z, T45, T3K, T42, T3R, T43;
{
E T3v, T3C, T3V, T3Y;
T3v = FMA(KP707106781, T3u, T3t);
T3C = SUB(T3y, T3B);
T3D = FMA(KP923879532, T3C, T3v);
T41 = FNMS(KP923879532, T3C, T3v);
T3V = FMA(KP707106781, T3U, T3T);
T3Y = SUB(T3W, T3X);
T3Z = FNMS(KP923879532, T3Y, T3V);
T45 = FMA(KP923879532, T3Y, T3V);
}
{
E T3G, T3J, T3N, T3Q;
T3G = FNMS(KP707106781, T3F, T3E);
T3J = FNMS(KP707106781, T3I, T3H);
T3K = FMA(KP668178637, T3J, T3G);
T42 = FNMS(KP668178637, T3G, T3J);
T3N = FNMS(KP707106781, T3M, T3L);
T3Q = FNMS(KP707106781, T3P, T3O);
T3R = FNMS(KP668178637, T3Q, T3N);
T43 = FMA(KP668178637, T3N, T3Q);
}
{
E T3S, T46, T40, T44;
T3S = ADD(T3K, T3R);
ro[WS(os, 19)] = FNMS(KP831469612, T3S, T3D);
ro[WS(os, 3)] = FMA(KP831469612, T3S, T3D);
T46 = ADD(T42, T43);
io[WS(os, 19)] = FNMS(KP831469612, T46, T45);
io[WS(os, 3)] = FMA(KP831469612, T46, T45);
T40 = SUB(T3R, T3K);
io[WS(os, 27)] = FNMS(KP831469612, T40, T3Z);
io[WS(os, 11)] = FMA(KP831469612, T40, T3Z);
T44 = SUB(T42, T43);
ro[WS(os, 27)] = FNMS(KP831469612, T44, T41);
ro[WS(os, 11)] = FMA(KP831469612, T44, T41);
}
}
{
E T49, T4p, T4j, T4l, T4c, T4n, T4f, T4m;
{
E T47, T48, T4h, T4i;
T47 = FNMS(KP707106781, T3u, T3t);
T48 = ADD(T3X, T3W);
T49 = FNMS(KP923879532, T48, T47);
T4p = FMA(KP923879532, T48, T47);
T4h = FNMS(KP707106781, T3U, T3T);
T4i = ADD(T3y, T3B);
T4j = FMA(KP923879532, T4i, T4h);
T4l = FNMS(KP923879532, T4i, T4h);
}
{
E T4a, T4b, T4d, T4e;
T4a = FMA(KP707106781, T3I, T3H);
T4b = FMA(KP707106781, T3F, T3E);
T4c = FMA(KP198912367, T4b, T4a);
T4n = FNMS(KP198912367, T4a, T4b);
T4d = FMA(KP707106781, T3P, T3O);
T4e = FMA(KP707106781, T3M, T3L);
T4f = FNMS(KP198912367, T4e, T4d);
T4m = FMA(KP198912367, T4d, T4e);
}
{
E T4g, T4o, T4k, T4q;
T4g = SUB(T4c, T4f);
ro[WS(os, 23)] = FNMS(KP980785280, T4g, T49);
ro[WS(os, 7)] = FMA(KP980785280, T4g, T49);
T4o = SUB(T4m, T4n);
io[WS(os, 23)] = FNMS(KP980785280, T4o, T4l);
io[WS(os, 7)] = FMA(KP980785280, T4o, T4l);
T4k = ADD(T4c, T4f);
io[WS(os, 15)] = FNMS(KP980785280, T4k, T4j);
io[WS(os, 31)] = FMA(KP980785280, T4k, T4j);
T4q = ADD(T4n, T4m);
ro[WS(os, 15)] = FNMS(KP980785280, T4q, T4p);
ro[WS(os, 31)] = FMA(KP980785280, T4q, T4p);
}
}
{
E T3b, T3n, T3l, T3r, T3e, T3o, T3h, T3p;
{
E T39, T3a, T3j, T3k;
T39 = FMA(KP707106781, T1G, T1z);
T3a = ADD(T2Z, T2Y);
T3b = FMA(KP923879532, T3a, T39);
T3n = FNMS(KP923879532, T3a, T39);
T3j = FMA(KP707106781, T2W, T2T);
T3k = ADD(T1O, T1V);
T3l = FNMS(KP923879532, T3k, T3j);
T3r = FMA(KP923879532, T3k, T3j);
}
{
E T3c, T3d, T3f, T3g;
T3c = FMA(KP707106781, T2m, T2j);
T3d = FMA(KP707106781, T2d, T22);
T3e = FMA(KP198912367, T3d, T3c);
T3o = FNMS(KP198912367, T3c, T3d);
T3f = FMA(KP707106781, T2N, T2K);
T3g = FMA(KP707106781, T2E, T2t);
T3h = FNMS(KP198912367, T3g, T3f);
T3p = FMA(KP198912367, T3f, T3g);
}
{
E T3i, T3s, T3m, T3q;
T3i = ADD(T3e, T3h);
ro[WS(os, 17)] = FNMS(KP980785280, T3i, T3b);
ro[WS(os, 1)] = FMA(KP980785280, T3i, T3b);
T3s = ADD(T3o, T3p);
io[WS(os, 17)] = FNMS(KP980785280, T3s, T3r);
io[WS(os, 1)] = FMA(KP980785280, T3s, T3r);
T3m = SUB(T3h, T3e);
io[WS(os, 25)] = FNMS(KP980785280, T3m, T3l);
io[WS(os, 9)] = FMA(KP980785280, T3m, T3l);
T3q = SUB(T3o, T3p);
ro[WS(os, 25)] = FNMS(KP980785280, T3q, T3n);
ro[WS(os, 9)] = FMA(KP980785280, T3q, T3n);
}
}
ri = ri + ivs;
ii = ii + ivs;
ro = ro + ovs;
io = io + ovs;
MAKE_VOLATILE_STRIDE(128, is);
MAKE_VOLATILE_STRIDE(128, os);
}
}
}

