 
using INT = int;
using stride = int;

using E = __m512;
using R = __m512;

inline int WS(const stride s, const stride i) { return s * i; }

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

#define DK(name, val)                                                          \
  static const __m512 name = {(val), (val), (val), (val),                      \
                              (val), (val), (val), (val)}


inline void MAKE_VOLATILE_STRIDE(int a, int b) {}


/* Generated by: ./fftw3/genfft/gen_notw.native -n 12 -standalone -fma -generic-arith -compact -name dft_c2cf_12 */

/*
 * This function contains 96 FP additions, 24 FP multiplications,
 * (or, 72 additions, 0 multiplications, 24 fused multiply/add),
 * 43 stack variables, 2 constants, and 48 memory accesses
 */
void dft_c2cf_12(const R * ri, const R * ii, R * ro, R * io, stride is, stride os, INT v, INT ivs, INT ovs)
{
DK(KP866025403, +0.866025403784438646763723170752936183471402627);
DK(KP500000000, +0.500000000000000000000000000000000000000000000);
{
INT i;
for (i = v; i > 0; i = i - 1, ri = ri + ivs, ii = ii + ivs, ro = ro + ovs, io = io + ovs, MAKE_VOLATILE_STRIDE(48, is), MAKE_VOLATILE_STRIDE(48, os)){
E T5, TR, TA, Ts, TS, Tz, Ta, TU, TD, Tx, TV, TC, Tg, T1d, TG;
E TJ, T1u, T1c, Tl, T1i, TL, TO, T1v, T1h;
{
E T1, T2, T3, T4;
T1 = ri[0];
T2 = ri[WS(is, 4)];
T3 = ri[WS(is, 8)];
T4 = ADD(T2, T3);
T5 = ADD(T1, T4);
TR = FNMS(KP500000000, T4, T1);
TA = SUB(T3, T2);
}
{
E To, Tp, Tq, Tr;
To = ii[0];
Tp = ii[WS(is, 4)];
Tq = ii[WS(is, 8)];
Tr = ADD(Tp, Tq);
Ts = ADD(To, Tr);
TS = SUB(Tp, Tq);
Tz = FNMS(KP500000000, Tr, To);
}
{
E T6, T7, T8, T9;
T6 = ri[WS(is, 6)];
T7 = ri[WS(is, 10)];
T8 = ri[WS(is, 2)];
T9 = ADD(T7, T8);
Ta = ADD(T6, T9);
TU = FNMS(KP500000000, T9, T6);
TD = SUB(T8, T7);
}
{
E Tt, Tu, Tv, Tw;
Tt = ii[WS(is, 6)];
Tu = ii[WS(is, 10)];
Tv = ii[WS(is, 2)];
Tw = ADD(Tu, Tv);
Tx = ADD(Tt, Tw);
TV = SUB(Tu, Tv);
TC = FNMS(KP500000000, Tw, Tt);
}
{
E Tc, Td, Te, Tf;
Tc = ri[WS(is, 3)];
Td = ri[WS(is, 7)];
Te = ri[WS(is, 11)];
Tf = ADD(Td, Te);
Tg = ADD(Tc, Tf);
T1d = SUB(Te, Td);
TG = FNMS(KP500000000, Tf, Tc);
}
{
E T1a, TH, TI, T1b;
T1a = ii[WS(is, 3)];
TH = ii[WS(is, 7)];
TI = ii[WS(is, 11)];
T1b = ADD(TH, TI);
TJ = SUB(TH, TI);
T1u = ADD(T1a, T1b);
T1c = FNMS(KP500000000, T1b, T1a);
}
{
E Th, Ti, Tj, Tk;
Th = ri[WS(is, 9)];
Ti = ri[WS(is, 1)];
Tj = ri[WS(is, 5)];
Tk = ADD(Ti, Tj);
Tl = ADD(Th, Tk);
T1i = SUB(Tj, Ti);
TL = FNMS(KP500000000, Tk, Th);
}
{
E T1f, TM, TN, T1g;
T1f = ii[WS(is, 9)];
TM = ii[WS(is, 1)];
TN = ii[WS(is, 5)];
T1g = ADD(TM, TN);
TO = SUB(TM, TN);
T1v = ADD(T1f, T1g);
T1h = FNMS(KP500000000, T1g, T1f);
}
{
E Tb, Tm, T1t, T1w;
Tb = ADD(T5, Ta);
Tm = ADD(Tg, Tl);
ro[WS(os, 6)] = SUB(Tb, Tm);
ro[0] = ADD(Tb, Tm);
{
E T1x, T1y, Tn, Ty;
T1x = ADD(Ts, Tx);
T1y = ADD(T1u, T1v);
io[WS(os, 6)] = SUB(T1x, T1y);
io[0] = ADD(T1x, T1y);
Tn = SUB(Tg, Tl);
Ty = SUB(Ts, Tx);
io[WS(os, 3)] = ADD(Tn, Ty);
io[WS(os, 9)] = SUB(Ty, Tn);
}
T1t = SUB(T5, Ta);
T1w = SUB(T1u, T1v);
ro[WS(os, 3)] = SUB(T1t, T1w);
ro[WS(os, 9)] = ADD(T1t, T1w);
{
E T11, T1l, T1k, T1m, T14, T18, T17, T19;
{
E TZ, T10, T1e, T1j;
TZ = FMA(KP866025403, TA, Tz);
T10 = FMA(KP866025403, TD, TC);
T11 = SUB(TZ, T10);
T1l = ADD(TZ, T10);
T1e = FMA(KP866025403, T1d, T1c);
T1j = FMA(KP866025403, T1i, T1h);
T1k = SUB(T1e, T1j);
T1m = ADD(T1e, T1j);
}
{
E T12, T13, T15, T16;
T12 = FMA(KP866025403, TJ, TG);
T13 = FMA(KP866025403, TO, TL);
T14 = SUB(T12, T13);
T18 = ADD(T12, T13);
T15 = FMA(KP866025403, TS, TR);
T16 = FMA(KP866025403, TV, TU);
T17 = ADD(T15, T16);
T19 = SUB(T15, T16);
}
io[WS(os, 1)] = SUB(T11, T14);
ro[WS(os, 1)] = ADD(T19, T1k);
io[WS(os, 7)] = ADD(T11, T14);
ro[WS(os, 7)] = SUB(T19, T1k);
ro[WS(os, 10)] = SUB(T17, T18);
io[WS(os, 10)] = SUB(T1l, T1m);
ro[WS(os, 4)] = ADD(T17, T18);
io[WS(os, 4)] = ADD(T1l, T1m);
}
{
E TF, T1r, T1q, T1s, TQ, TY, TX, T1n;
{
E TB, TE, T1o, T1p;
TB = FNMS(KP866025403, TA, Tz);
TE = FNMS(KP866025403, TD, TC);
TF = SUB(TB, TE);
T1r = ADD(TB, TE);
T1o = FNMS(KP866025403, T1d, T1c);
T1p = FNMS(KP866025403, T1i, T1h);
T1q = SUB(T1o, T1p);
T1s = ADD(T1o, T1p);
}
{
E TK, TP, TT, TW;
TK = FNMS(KP866025403, TJ, TG);
TP = FNMS(KP866025403, TO, TL);
TQ = SUB(TK, TP);
TY = ADD(TK, TP);
TT = FNMS(KP866025403, TS, TR);
TW = FNMS(KP866025403, TV, TU);
TX = ADD(TT, TW);
T1n = SUB(TT, TW);
}
io[WS(os, 5)] = SUB(TF, TQ);
ro[WS(os, 5)] = ADD(T1n, T1q);
io[WS(os, 11)] = ADD(TF, TQ);
ro[WS(os, 11)] = SUB(T1n, T1q);
ro[WS(os, 2)] = SUB(TX, TY);
io[WS(os, 2)] = SUB(T1r, T1s);
ro[WS(os, 8)] = ADD(TX, TY);
io[WS(os, 8)] = ADD(T1r, T1s);
}
}
}
}
}

