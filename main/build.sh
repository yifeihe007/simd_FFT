#!/bin/bash
g++ -g -Wall -march=skylake-avx512 main.cpp utils.cpp -o main.out -lfftw3f
