#!/bin/bash

for i in 32 64 128 256 512 1024
do
  for j in 32 64 128 256 512 1024
  do
    for k in 1 2 4 6 8
    do
      for l in 1 2 3
      do
        export NSAMP=$i
        export NLOOP=$j
        export OMP_NUM_THREADS=$k
        echo "INTER =" $l
        echo "NSAMP =" $NSAMP
        echo "NLOOP =" $NLOOP
        echo "OMP_NUM_THREADS =" $OMP_NUM_THREADS 
        ./build/dft_simd
done
done
done
done
