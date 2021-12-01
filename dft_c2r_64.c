/* Generated by: ./gen_r2cb.native -n 64 -standalone -fma -generic-arith -compact -name dft_codelet_c2r_64 */

/*
 * This function contains 394 FP additions, 216 FP multiplications,
 * (or, 178 additions, 0 multiplications, 216 fused multiply/add),
 * 109 stack variables, 18 constants, and 128 memory accesses
 */
void dft_codelet_c2r_64(R * R0, R * R1, R * Cr, R * Ci, stride rs, stride csr, stride csi, INT v, INT ivs, INT ovs)
{
DK(KP1_546020906, +1.546020906725473921621813219516939601942082586);
DK(KP820678790, +0.820678790828660330972281985331011598767386482);
DK(KP1_990369453, +1.990369453344393772489673906218959843150949737);
DK(KP098491403, +0.098491403357164253077197521291327432293052451);
DK(KP1_763842528, +1.763842528696710059425513727320776699016885241);
DK(KP534511135, +0.534511135950791641089685961295362908582039528);
DK(KP1_913880671, +1.913880671464417729871595773960539938965698411);
DK(KP303346683, +0.303346683607342391675883946941299872384187453);
DK(KP923879532, +0.923879532511286756128183189396788286822416626);
DK(KP1_662939224, +1.662939224605090474157576755235811513477121624);
DK(KP668178637, +0.668178637919298919997757686523080761552472251);
DK(KP1_961570560, +1.961570560806460898252364472268478073947867462);
DK(KP198912367, +0.198912367379658006911597622644676228597850501);
DK(KP707106781, +0.707106781186547524400844362104849039284835938);
DK(KP1_847759065, +1.847759065022573512256366378793576573644833252);
DK(KP414213562, +0.414213562373095048801688724209698078569671875);
DK(KP1_414213562, +1.414213562373095048801688724209698078569671875);
DK(KP2_000000000, +2.000000000000000000000000000000000000000000000);
{
INT i;
#pragma omp parallel for
for (i = v; i > 0; i = i - 1){
E T9, T5H, T4p, T5j, T1b, T2T, T3j, T3Z, Tg, T5I, T1m, T2U, T3m, T40, T4u;
E T5k, T1s, T3o, T3r, T1J, To, Tv, T5K, T5L, T5M, T5N, T4A, T5n, T1D, T3s;
E T4F, T5m, T1M, T3p, T1U, T3w, T3H, T2z, TM, T5Q, T5Y, T6f, T4M, T5q, T25;
E T3I, T53, T5t, T2C, T3x, T11, T5V, T4W, T55, T5T, T6g, T2h, T2E, T2s, T2F;
E T3E, T3K, T4R, T54, T3B, T3L;
{
E T4, T14, T3, T13, T8, T16, T19, T4o, T1, T2, T5, T4n;
T4 = Cr[WS(csr, 16)];
T14 = Ci[WS(csi, 16)];
T1 = Cr[0];
T2 = Cr[WS(csr, 32)];
T3 = ADD(T1, T2);
T13 = SUB(T1, T2);
{
E T6, T7, T17, T18;
T6 = Cr[WS(csr, 8)];
T7 = Cr[WS(csr, 24)];
T8 = ADD(T6, T7);
T16 = SUB(T6, T7);
T17 = Ci[WS(csi, 8)];
T18 = Ci[WS(csi, 24)];
T19 = ADD(T17, T18);
T4o = SUB(T17, T18);
}
T5 = FMA(KP2_000000000, T4, T3);
T9 = FMA(KP2_000000000, T8, T5);
T5H = FNMS(KP2_000000000, T8, T5);
T4n = FNMS(KP2_000000000, T4, T3);
T4p = FMA(KP2_000000000, T4o, T4n);
T5j = FNMS(KP2_000000000, T4o, T4n);
{
E T15, T1a, T3h, T3i;
T15 = FMA(KP2_000000000, T14, T13);
T1a = ADD(T16, T19);
T1b = FMA(KP1_414213562, T1a, T15);
T2T = FNMS(KP1_414213562, T1a, T15);
T3h = FNMS(KP2_000000000, T14, T13);
T3i = SUB(T19, T16);
T3j = FMA(KP1_414213562, T3i, T3h);
T3Z = FNMS(KP1_414213562, T3i, T3h);
}
}
{
E Tc, T1c, T1j, T4r, Tf, T1k, T1f, T4s, T1g, T1l;
{
E Ta, Tb, T1h, T1i;
Ta = Cr[WS(csr, 4)];
Tb = Cr[WS(csr, 28)];
Tc = ADD(Ta, Tb);
T1c = SUB(Ta, Tb);
T1h = Ci[WS(csi, 4)];
T1i = Ci[WS(csi, 28)];
T1j = ADD(T1h, T1i);
T4r = SUB(T1h, T1i);
}
{
E Td, Te, T1d, T1e;
Td = Cr[WS(csr, 20)];
Te = Cr[WS(csr, 12)];
Tf = ADD(Td, Te);
T1k = SUB(Td, Te);
T1d = Ci[WS(csi, 20)];
T1e = Ci[WS(csi, 12)];
T1f = ADD(T1d, T1e);
T4s = SUB(T1d, T1e);
}
Tg = ADD(Tc, Tf);
T5I = ADD(T4s, T4r);
T1g = ADD(T1c, T1f);
T1l = SUB(T1j, T1k);
T1m = FMA(KP414213562, T1l, T1g);
T2U = FNMS(KP414213562, T1g, T1l);
{
E T3k, T3l, T4q, T4t;
T3k = ADD(T1k, T1j);
T3l = SUB(T1c, T1f);
T3m = FMA(KP414213562, T3l, T3k);
T40 = FNMS(KP414213562, T3k, T3l);
T4q = SUB(Tc, Tf);
T4t = SUB(T4r, T4s);
T4u = ADD(T4q, T4t);
T5k = SUB(T4t, T4q);
}
}
{
E Tk, T1o, T1H, T4C, Tn, T1I, T1r, T4D, Tr, T1t, T1w, T4x, Tu, T1y, T1B;
E T4y;
{
E Ti, Tj, T1F, T1G;
Ti = Cr[WS(csr, 2)];
Tj = Cr[WS(csr, 30)];
Tk = ADD(Ti, Tj);
T1o = SUB(Ti, Tj);
T1F = Ci[WS(csi, 2)];
T1G = Ci[WS(csi, 30)];
T1H = ADD(T1F, T1G);
T4C = SUB(T1F, T1G);
}
{
E Tl, Tm, T1p, T1q;
Tl = Cr[WS(csr, 18)];
Tm = Cr[WS(csr, 14)];
Tn = ADD(Tl, Tm);
T1I = SUB(Tl, Tm);
T1p = Ci[WS(csi, 18)];
T1q = Ci[WS(csi, 14)];
T1r = ADD(T1p, T1q);
T4D = SUB(T1p, T1q);
}
{
E Tp, Tq, T1u, T1v;
Tp = Cr[WS(csr, 10)];
Tq = Cr[WS(csr, 22)];
Tr = ADD(Tp, Tq);
T1t = SUB(Tp, Tq);
T1u = Ci[WS(csi, 10)];
T1v = Ci[WS(csi, 22)];
T1w = ADD(T1u, T1v);
T4x = SUB(T1u, T1v);
}
{
E Ts, Tt, T1z, T1A;
Ts = Cr[WS(csr, 6)];
Tt = Cr[WS(csr, 26)];
Tu = ADD(Ts, Tt);
T1y = SUB(Ts, Tt);
T1z = Ci[WS(csi, 6)];
T1A = Ci[WS(csi, 26)];
T1B = ADD(T1z, T1A);
T4y = SUB(T1A, T1z);
}
T1s = ADD(T1o, T1r);
T3o = SUB(T1o, T1r);
T3r = ADD(T1I, T1H);
T1J = SUB(T1H, T1I);
To = ADD(Tk, Tn);
Tv = ADD(Tr, Tu);
T5K = SUB(To, Tv);
{
E T4w, T4z, T1x, T1C;
T5L = ADD(T4D, T4C);
T5M = ADD(T4x, T4y);
T5N = SUB(T5L, T5M);
T4w = SUB(Tk, Tn);
T4z = SUB(T4x, T4y);
T4A = ADD(T4w, T4z);
T5n = SUB(T4w, T4z);
T1x = ADD(T1t, T1w);
T1C = ADD(T1y, T1B);
T1D = ADD(T1x, T1C);
T3s = SUB(T1x, T1C);
{
E T4B, T4E, T1K, T1L;
T4B = SUB(Tu, Tr);
T4E = SUB(T4C, T4D);
T4F = ADD(T4B, T4E);
T5m = SUB(T4E, T4B);
T1K = SUB(T1w, T1t);
T1L = SUB(T1y, T1B);
T1M = ADD(T1K, T1L);
T3p = SUB(T1L, T1K);
}
}
}
{
E TA, T1Q, T2x, T50, TD, T2y, T1T, T51, TH, T1V, T1Y, T4J, TK, T20, T23;
E T4K;
{
E Ty, Tz, T2v, T2w;
Ty = Cr[WS(csr, 1)];
Tz = Cr[WS(csr, 31)];
TA = ADD(Ty, Tz);
T1Q = SUB(Ty, Tz);
T2v = Ci[WS(csi, 1)];
T2w = Ci[WS(csi, 31)];
T2x = ADD(T2v, T2w);
T50 = SUB(T2v, T2w);
}
{
E TB, TC, T1R, T1S;
TB = Cr[WS(csr, 17)];
TC = Cr[WS(csr, 15)];
TD = ADD(TB, TC);
T2y = SUB(TB, TC);
T1R = Ci[WS(csi, 17)];
T1S = Ci[WS(csi, 15)];
T1T = ADD(T1R, T1S);
T51 = SUB(T1R, T1S);
}
{
E TF, TG, T1W, T1X;
TF = Cr[WS(csr, 9)];
TG = Cr[WS(csr, 23)];
TH = ADD(TF, TG);
T1V = SUB(TF, TG);
T1W = Ci[WS(csi, 9)];
T1X = Ci[WS(csi, 23)];
T1Y = ADD(T1W, T1X);
T4J = SUB(T1W, T1X);
}
{
E TI, TJ, T21, T22;
TI = Cr[WS(csr, 7)];
TJ = Cr[WS(csr, 25)];
TK = ADD(TI, TJ);
T20 = SUB(TI, TJ);
T21 = Ci[WS(csi, 7)];
T22 = Ci[WS(csi, 25)];
T23 = ADD(T21, T22);
T4K = SUB(T22, T21);
}
{
E TE, TL, T1Z, T24;
T1U = ADD(T1Q, T1T);
T3w = SUB(T1Q, T1T);
T3H = ADD(T2y, T2x);
T2z = SUB(T2x, T2y);
TE = ADD(TA, TD);
TL = ADD(TH, TK);
TM = ADD(TE, TL);
T5Q = SUB(TE, TL);
{
E T5W, T5X, T4I, T4L;
T5W = ADD(T51, T50);
T5X = ADD(T4J, T4K);
T5Y = SUB(T5W, T5X);
T6f = ADD(T5X, T5W);
T4I = SUB(TA, TD);
T4L = SUB(T4J, T4K);
T4M = ADD(T4I, T4L);
T5q = SUB(T4I, T4L);
}
T1Z = ADD(T1V, T1Y);
T24 = ADD(T20, T23);
T25 = ADD(T1Z, T24);
T3I = SUB(T1Z, T24);
{
E T4Z, T52, T2A, T2B;
T4Z = SUB(TK, TH);
T52 = SUB(T50, T51);
T53 = ADD(T4Z, T52);
T5t = SUB(T52, T4Z);
T2A = SUB(T1Y, T1V);
T2B = SUB(T20, T23);
T2C = ADD(T2A, T2B);
T3x = SUB(T2B, T2A);
}
}
}
{
E TP, T27, T2e, T4O, TS, T2f, T2a, T4P, TW, T2i, T2q, T4T, TZ, T2n, T2l;
E T4U;
{
E TN, TO, T2c, T2d;
TN = Cr[WS(csr, 5)];
TO = Cr[WS(csr, 27)];
TP = ADD(TN, TO);
T27 = SUB(TN, TO);
T2c = Ci[WS(csi, 5)];
T2d = Ci[WS(csi, 27)];
T2e = ADD(T2c, T2d);
T4O = SUB(T2c, T2d);
}
{
E TQ, TR, T28, T29;
TQ = Cr[WS(csr, 21)];
TR = Cr[WS(csr, 11)];
TS = ADD(TQ, TR);
T2f = SUB(TQ, TR);
T28 = Ci[WS(csi, 21)];
T29 = Ci[WS(csi, 11)];
T2a = ADD(T28, T29);
T4P = SUB(T28, T29);
}
{
E TU, TV, T2o, T2p;
TU = Cr[WS(csr, 3)];
TV = Cr[WS(csr, 29)];
TW = ADD(TU, TV);
T2i = SUB(TU, TV);
T2o = Ci[WS(csi, 3)];
T2p = Ci[WS(csi, 29)];
T2q = ADD(T2o, T2p);
T4T = SUB(T2p, T2o);
}
{
E TX, TY, T2j, T2k;
TX = Cr[WS(csr, 13)];
TY = Cr[WS(csr, 19)];
TZ = ADD(TX, TY);
T2n = SUB(TX, TY);
T2j = Ci[WS(csi, 13)];
T2k = Ci[WS(csi, 19)];
T2l = ADD(T2j, T2k);
T4U = SUB(T2j, T2k);
}
{
E TT, T10, T4S, T4V;
TT = ADD(TP, TS);
T10 = ADD(TW, TZ);
T11 = ADD(TT, T10);
T5V = SUB(T10, TT);
T4S = SUB(TW, TZ);
T4V = SUB(T4T, T4U);
T4W = SUB(T4S, T4V);
T55 = ADD(T4S, T4V);
}
{
E T5R, T5S, T2b, T2g;
T5R = ADD(T4P, T4O);
T5S = ADD(T4U, T4T);
T5T = SUB(T5R, T5S);
T6g = ADD(T5R, T5S);
T2b = ADD(T27, T2a);
T2g = SUB(T2e, T2f);
T2h = FMA(KP414213562, T2g, T2b);
T2E = FNMS(KP414213562, T2b, T2g);
}
{
E T2m, T2r, T3C, T3D;
T2m = ADD(T2i, T2l);
T2r = ADD(T2n, T2q);
T2s = FMA(KP414213562, T2r, T2m);
T2F = FNMS(KP414213562, T2m, T2r);
T3C = SUB(T2n, T2q);
T3D = SUB(T2i, T2l);
T3E = FNMS(KP414213562, T3D, T3C);
T3K = FMA(KP414213562, T3C, T3D);
}
{
E T4N, T4Q, T3z, T3A;
T4N = SUB(TP, TS);
T4Q = SUB(T4O, T4P);
T4R = ADD(T4N, T4Q);
T54 = SUB(T4Q, T4N);
T3z = ADD(T2f, T2e);
T3A = SUB(T27, T2a);
T3B = FMA(KP414213562, T3A, T3z);
T3L = FNMS(KP414213562, T3z, T3A);
}
}
{
E T12, T6m, Tx, T6l, Th, Tw;
T12 = ADD(TM, T11);
T6m = ADD(T6g, T6f);
Th = FMA(KP2_000000000, Tg, T9);
Tw = ADD(To, Tv);
Tx = FMA(KP2_000000000, Tw, Th);
T6l = FNMS(KP2_000000000, Tw, Th);
R0[WS(rs, 16)] = FNMS(KP2_000000000, T12, Tx);
R0[WS(rs, 8)] = FMA(KP2_000000000, T6m, T6l);
R0[0] = FMA(KP2_000000000, T12, Tx);
R0[WS(rs, 24)] = FNMS(KP2_000000000, T6m, T6l);
}
{
E T65, T69, T68, T6a;
{
E T63, T64, T66, T67;
T63 = FNMS(KP2_000000000, T5I, T5H);
T64 = SUB(T5N, T5K);
T65 = FNMS(KP1_414213562, T64, T63);
T69 = FMA(KP1_414213562, T64, T63);
T66 = SUB(T5Q, T5T);
T67 = SUB(T5Y, T5V);
T68 = FNMS(KP414213562, T67, T66);
T6a = FMA(KP414213562, T66, T67);
}
R0[WS(rs, 14)] = FNMS(KP1_847759065, T68, T65);
R0[WS(rs, 6)] = FMA(KP1_847759065, T6a, T69);
R0[WS(rs, 30)] = FMA(KP1_847759065, T68, T65);
R0[WS(rs, 22)] = FNMS(KP1_847759065, T6a, T69);
}
{
E T6d, T6j, T6i, T6k;
{
E T6b, T6c, T6e, T6h;
T6b = FNMS(KP2_000000000, Tg, T9);
T6c = ADD(T5M, T5L);
T6d = FMA(KP2_000000000, T6c, T6b);
T6j = FNMS(KP2_000000000, T6c, T6b);
T6e = SUB(TM, T11);
T6h = SUB(T6f, T6g);
T6i = ADD(T6e, T6h);
T6k = SUB(T6h, T6e);
}
R0[WS(rs, 20)] = FNMS(KP1_414213562, T6i, T6d);
R0[WS(rs, 12)] = FMA(KP1_414213562, T6k, T6j);
R0[WS(rs, 4)] = FMA(KP1_414213562, T6i, T6d);
R0[WS(rs, 28)] = FNMS(KP1_414213562, T6k, T6j);
}
{
E T5P, T61, T60, T62;
{
E T5J, T5O, T5U, T5Z;
T5J = FMA(KP2_000000000, T5I, T5H);
T5O = ADD(T5K, T5N);
T5P = FMA(KP1_414213562, T5O, T5J);
T61 = FNMS(KP1_414213562, T5O, T5J);
T5U = ADD(T5Q, T5T);
T5Z = ADD(T5V, T5Y);
T60 = FMA(KP414213562, T5Z, T5U);
T62 = FNMS(KP414213562, T5U, T5Z);
}
R0[WS(rs, 18)] = FNMS(KP1_847759065, T60, T5P);
R0[WS(rs, 10)] = FMA(KP1_847759065, T62, T61);
R0[WS(rs, 2)] = FMA(KP1_847759065, T60, T5P);
R0[WS(rs, 26)] = FNMS(KP1_847759065, T62, T61);
}
{
E T4Y, T5e, T57, T5f, T4H, T59, T5d, T5h, T4X, T56;
T4X = ADD(T4R, T4W);
T4Y = FMA(KP707106781, T4X, T4M);
T5e = FNMS(KP707106781, T4X, T4M);
T56 = ADD(T54, T55);
T57 = FMA(KP707106781, T56, T53);
T5f = FNMS(KP707106781, T56, T53);
{
E T4v, T4G, T5b, T5c;
T4v = FMA(KP1_414213562, T4u, T4p);
T4G = FMA(KP414213562, T4F, T4A);
T4H = FMA(KP1_847759065, T4G, T4v);
T59 = FNMS(KP1_847759065, T4G, T4v);
T5b = FNMS(KP1_414213562, T4u, T4p);
T5c = FNMS(KP414213562, T4A, T4F);
T5d = FNMS(KP1_847759065, T5c, T5b);
T5h = FMA(KP1_847759065, T5c, T5b);
}
{
E T58, T5i, T5a, T5g;
T58 = FMA(KP198912367, T57, T4Y);
R0[WS(rs, 17)] = FNMS(KP1_961570560, T58, T4H);
R0[WS(rs, 1)] = FMA(KP1_961570560, T58, T4H);
T5i = FMA(KP668178637, T5e, T5f);
R0[WS(rs, 21)] = FNMS(KP1_662939224, T5i, T5h);
R0[WS(rs, 5)] = FMA(KP1_662939224, T5i, T5h);
T5a = FNMS(KP198912367, T4Y, T57);
R0[WS(rs, 25)] = FNMS(KP1_961570560, T5a, T59);
R0[WS(rs, 9)] = FMA(KP1_961570560, T5a, T59);
T5g = FNMS(KP668178637, T5f, T5e);
R0[WS(rs, 13)] = FNMS(KP1_662939224, T5g, T5d);
R0[WS(rs, 29)] = FMA(KP1_662939224, T5g, T5d);
}
}
{
E T5s, T5C, T5v, T5D, T5p, T5x, T5B, T5F, T5r, T5u;
T5r = SUB(T55, T54);
T5s = FNMS(KP707106781, T5r, T5q);
T5C = FMA(KP707106781, T5r, T5q);
T5u = SUB(T4R, T4W);
T5v = FNMS(KP707106781, T5u, T5t);
T5D = FMA(KP707106781, T5u, T5t);
{
E T5l, T5o, T5z, T5A;
T5l = FMA(KP1_414213562, T5k, T5j);
T5o = FMA(KP414213562, T5n, T5m);
T5p = FMA(KP1_847759065, T5o, T5l);
T5x = FNMS(KP1_847759065, T5o, T5l);
T5z = FNMS(KP1_414213562, T5k, T5j);
T5A = FNMS(KP414213562, T5m, T5n);
T5B = FMA(KP1_847759065, T5A, T5z);
T5F = FNMS(KP1_847759065, T5A, T5z);
}
{
E T5w, T5G, T5y, T5E;
T5w = FMA(KP668178637, T5v, T5s);
R0[WS(rs, 19)] = FNMS(KP1_662939224, T5w, T5p);
R0[WS(rs, 3)] = FMA(KP1_662939224, T5w, T5p);
T5G = FMA(KP198912367, T5C, T5D);
R0[WS(rs, 23)] = FNMS(KP1_961570560, T5G, T5F);
R0[WS(rs, 7)] = FMA(KP1_961570560, T5G, T5F);
T5y = FNMS(KP668178637, T5s, T5v);
R0[WS(rs, 27)] = FNMS(KP1_662939224, T5y, T5x);
R0[WS(rs, 11)] = FMA(KP1_662939224, T5y, T5x);
T5E = FNMS(KP198912367, T5D, T5C);
R0[WS(rs, 15)] = FNMS(KP1_961570560, T5E, T5B);
R0[WS(rs, 31)] = FMA(KP1_961570560, T5E, T5B);
}
}
{
E T3n, T3R, T3u, T3S, T3G, T3U, T3N, T3V, T3q, T3t;
T3n = FMA(KP1_847759065, T3m, T3j);
T3R = FNMS(KP1_847759065, T3m, T3j);
T3q = FNMS(KP707106781, T3p, T3o);
T3t = FNMS(KP707106781, T3s, T3r);
T3u = FMA(KP668178637, T3t, T3q);
T3S = FNMS(KP668178637, T3q, T3t);
{
E T3y, T3F, T3J, T3M;
T3y = FNMS(KP707106781, T3x, T3w);
T3F = SUB(T3B, T3E);
T3G = FMA(KP923879532, T3F, T3y);
T3U = FNMS(KP923879532, T3F, T3y);
T3J = FNMS(KP707106781, T3I, T3H);
T3M = SUB(T3K, T3L);
T3N = FMA(KP923879532, T3M, T3J);
T3V = FNMS(KP923879532, T3M, T3J);
}
{
E T3v, T3O, T3X, T3Y;
T3v = FMA(KP1_662939224, T3u, T3n);
T3O = FMA(KP303346683, T3N, T3G);
R1[WS(rs, 17)] = FNMS(KP1_913880671, T3O, T3v);
R1[WS(rs, 1)] = FMA(KP1_913880671, T3O, T3v);
T3X = FMA(KP1_662939224, T3S, T3R);
T3Y = FMA(KP534511135, T3U, T3V);
R1[WS(rs, 21)] = FNMS(KP1_763842528, T3Y, T3X);
R1[WS(rs, 5)] = FMA(KP1_763842528, T3Y, T3X);
}
{
E T3P, T3Q, T3T, T3W;
T3P = FNMS(KP1_662939224, T3u, T3n);
T3Q = FNMS(KP303346683, T3G, T3N);
R1[WS(rs, 25)] = FNMS(KP1_913880671, T3Q, T3P);
R1[WS(rs, 9)] = FMA(KP1_913880671, T3Q, T3P);
T3T = FNMS(KP1_662939224, T3S, T3R);
T3W = FNMS(KP534511135, T3V, T3U);
R1[WS(rs, 13)] = FNMS(KP1_763842528, T3W, T3T);
R1[WS(rs, 29)] = FMA(KP1_763842528, T3W, T3T);
}
}
{
E T1n, T2L, T1O, T2M, T2u, T2O, T2H, T2P, T1E, T1N;
T1n = FMA(KP1_847759065, T1m, T1b);
T2L = FNMS(KP1_847759065, T1m, T1b);
T1E = FMA(KP707106781, T1D, T1s);
T1N = FMA(KP707106781, T1M, T1J);
T1O = FMA(KP198912367, T1N, T1E);
T2M = FNMS(KP198912367, T1E, T1N);
{
E T26, T2t, T2D, T2G;
T26 = FMA(KP707106781, T25, T1U);
T2t = ADD(T2h, T2s);
T2u = FMA(KP923879532, T2t, T26);
T2O = FNMS(KP923879532, T2t, T26);
T2D = FMA(KP707106781, T2C, T2z);
T2G = SUB(T2E, T2F);
T2H = FMA(KP923879532, T2G, T2D);
T2P = FNMS(KP923879532, T2G, T2D);
}
{
E T1P, T2I, T2R, T2S;
T1P = FMA(KP1_961570560, T1O, T1n);
T2I = FMA(KP098491403, T2H, T2u);
R1[WS(rs, 16)] = FNMS(KP1_990369453, T2I, T1P);
R1[0] = FMA(KP1_990369453, T2I, T1P);
T2R = FMA(KP1_961570560, T2M, T2L);
T2S = FMA(KP820678790, T2O, T2P);
R1[WS(rs, 20)] = FNMS(KP1_546020906, T2S, T2R);
R1[WS(rs, 4)] = FMA(KP1_546020906, T2S, T2R);
}
{
E T2J, T2K, T2N, T2Q;
T2J = FNMS(KP1_961570560, T1O, T1n);
T2K = FNMS(KP098491403, T2u, T2H);
R1[WS(rs, 24)] = FNMS(KP1_990369453, T2K, T2J);
R1[WS(rs, 8)] = FMA(KP1_990369453, T2K, T2J);
T2N = FNMS(KP1_961570560, T2M, T2L);
T2Q = FNMS(KP820678790, T2P, T2O);
R1[WS(rs, 12)] = FNMS(KP1_546020906, T2Q, T2N);
R1[WS(rs, 28)] = FMA(KP1_546020906, T2Q, T2N);
}
}
{
E T41, T4f, T44, T4g, T48, T4i, T4b, T4j, T42, T43;
T41 = FNMS(KP1_847759065, T40, T3Z);
T4f = FMA(KP1_847759065, T40, T3Z);
T42 = FMA(KP707106781, T3s, T3r);
T43 = FMA(KP707106781, T3p, T3o);
T44 = FMA(KP198912367, T43, T42);
T4g = FNMS(KP198912367, T42, T43);
{
E T46, T47, T49, T4a;
T46 = FMA(KP707106781, T3x, T3w);
T47 = ADD(T3L, T3K);
T48 = FNMS(KP923879532, T47, T46);
T4i = FMA(KP923879532, T47, T46);
T49 = FMA(KP707106781, T3I, T3H);
T4a = ADD(T3B, T3E);
T4b = FNMS(KP923879532, T4a, T49);
T4j = FMA(KP923879532, T4a, T49);
}
{
E T45, T4c, T4l, T4m;
T45 = FMA(KP1_961570560, T44, T41);
T4c = FMA(KP820678790, T4b, T48);
R1[WS(rs, 19)] = FNMS(KP1_546020906, T4c, T45);
R1[WS(rs, 3)] = FMA(KP1_546020906, T4c, T45);
T4l = FNMS(KP1_961570560, T4g, T4f);
T4m = FMA(KP098491403, T4i, T4j);
R1[WS(rs, 23)] = FNMS(KP1_990369453, T4m, T4l);
R1[WS(rs, 7)] = FMA(KP1_990369453, T4m, T4l);
}
{
E T4d, T4e, T4h, T4k;
T4d = FNMS(KP1_961570560, T44, T41);
T4e = FNMS(KP820678790, T48, T4b);
R1[WS(rs, 27)] = FNMS(KP1_546020906, T4e, T4d);
R1[WS(rs, 11)] = FMA(KP1_546020906, T4e, T4d);
T4h = FMA(KP1_961570560, T4g, T4f);
T4k = FNMS(KP098491403, T4j, T4i);
R1[WS(rs, 15)] = FNMS(KP1_990369453, T4k, T4h);
R1[WS(rs, 31)] = FMA(KP1_990369453, T4k, T4h);
}
}
{
E T2V, T39, T2Y, T3a, T32, T3c, T35, T3d, T2W, T2X;
T2V = FMA(KP1_847759065, T2U, T2T);
T39 = FNMS(KP1_847759065, T2U, T2T);
T2W = FNMS(KP707106781, T1M, T1J);
T2X = FNMS(KP707106781, T1D, T1s);
T2Y = FMA(KP668178637, T2X, T2W);
T3a = FNMS(KP668178637, T2W, T2X);
{
E T30, T31, T33, T34;
T30 = FNMS(KP707106781, T25, T1U);
T31 = ADD(T2E, T2F);
T32 = FMA(KP923879532, T31, T30);
T3c = FNMS(KP923879532, T31, T30);
T33 = FNMS(KP707106781, T2C, T2z);
T34 = SUB(T2h, T2s);
T35 = FNMS(KP923879532, T34, T33);
T3d = FMA(KP923879532, T34, T33);
}
{
E T2Z, T36, T3f, T3g;
T2Z = FMA(KP1_662939224, T2Y, T2V);
T36 = FMA(KP534511135, T35, T32);
R1[WS(rs, 18)] = FNMS(KP1_763842528, T36, T2Z);
R1[WS(rs, 2)] = FMA(KP1_763842528, T36, T2Z);
T3f = FNMS(KP1_662939224, T3a, T39);
T3g = FMA(KP303346683, T3c, T3d);
R1[WS(rs, 22)] = FNMS(KP1_913880671, T3g, T3f);
R1[WS(rs, 6)] = FMA(KP1_913880671, T3g, T3f);
}
{
E T37, T38, T3b, T3e;
T37 = FNMS(KP1_662939224, T2Y, T2V);
T38 = FNMS(KP534511135, T32, T35);
R1[WS(rs, 26)] = FNMS(KP1_763842528, T38, T37);
R1[WS(rs, 10)] = FMA(KP1_763842528, T38, T37);
T3b = FMA(KP1_662939224, T3a, T39);
T3e = FNMS(KP303346683, T3d, T3c);
R1[WS(rs, 14)] = FNMS(KP1_913880671, T3e, T3b);
R1[WS(rs, 30)] = FMA(KP1_913880671, T3e, T3b);
}
}
R0 = R0 + ovs;
R1 = R1 + ovs;
Cr = Cr + ivs;
Ci = Ci + ivs;
MAKE_VOLATILE_STRIDE(256, rs);
MAKE_VOLATILE_STRIDE(256, csr);
MAKE_VOLATILE_STRIDE(256, csi);
}
}
}

