#!/bin/bash

lscpu

for i in 1024
do
  for j in 1024
  do
    for k in 1 2 4 6 8 10 12 14
    do
      for l in {1..20}
      do
        export NSAMP=$i
        export NLOOP=$j
        export OMP_NUM_THREADS=$k
        echo "INTER =" $l
        echo "NSAMP =" $NSAMP
        echo "NLOOP =" $NLOOP
        echo "OMP_NUM_THREADS =" $OMP_NUM_THREADS 
        ./build_f/dft_simd --gtest_filter=*manyc2cFFTW_Aligned_One*
done
done
done
done

for i in 1024
do
  for j in 1024
  do
    for k in 1 2 4 6 8 10 12 14
    do
      for l in {1..20}
      do
        export NSAMP=$i
        export NLOOP=$j
        export OMP_NUM_THREADS=$k
        echo "INTER =" $l
        echo "NSAMP =" $NSAMP
        echo "NLOOP =" $NLOOP
        echo "OMP_NUM_THREADS =" $OMP_NUM_THREADS 
        ./build_a2/dft_simd --gtest_filter=*AVX2c2c*
done
done
done
done

for i in 1024
do
  for j in 1024
  do
    for k in 1 2 4 6 8 10 12 14
    do
      for l in {1..20}
      do
        export NSAMP=$i
        export NLOOP=$j
        export OMP_NUM_THREADS=$k
        echo "INTER =" $l
        echo "NSAMP =" $NSAMP
        echo "NLOOP =" $NLOOP
        echo "OMP_NUM_THREADS =" $OMP_NUM_THREADS 
        ./build_a512/dft_simd --gtest_filter=*AVX512c2c*
done
done
done
done

