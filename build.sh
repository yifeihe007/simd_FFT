#!/bin/bash

cmake .. -DCMAKE_BUILD_TYPE=Release   -DFFTW_INCLUDE_DIR=~/fftw/include/ -DFFTW_LIBRARY=~/fftw/lib/libfftw3f.a -DFFTWF_LIBRARY=~/fftw/lib/libfftw3f.a -DFFTWT_LIBRARY=~/fftw/lib/libfftw3f_threads.a
make -j12
